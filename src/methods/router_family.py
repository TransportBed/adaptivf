from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import murmurhash3_32

from artifacts import ensure_dir, write_json
from console import banner, info, print_table
from datasets import LearnedDataset, load_learned_dataset
from index_manifest import write_index_manifest
from measurement_contract import (
    AVG_COMPUTATIONS_MODE_ROUTER,
    INDEX_OVERHEAD_MODE,
    QUERY_MEM_MODE_ISOLATED,
    QUERY_MEM_MODE_IN_PROCESS,
    SERVING_FOOTPRINT_MODE,
    recall_at_k,
)
from pq import adc_distances, adc_table, encode_pq_codes, pad_to_m, train_global_pq_codebooks
from presets import DATASETS

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore[assignment]

try:
    import numba  # type: ignore
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    numba = None  # type: ignore[assignment]
    njit = None  # type: ignore[assignment]
    _HAS_NUMBA = False


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            rss_pages = int(handle.readline().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (rss_pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _ensure_tf_graph():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for BLISS/MLP-IVF training.") from exc
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    return tf


@dataclass(frozen=True)
class RouterConfig:
    dataset: str
    method: str
    init_mode: str
    repetitions: int = 4
    partitions: int | None = None
    probe_depth: int = 10
    min_reps: int = 2
    k: int = 10
    hidden: int = 512
    epochs: int = 20
    batch_size: int = 1024
    reassign_batch_size: int = 1024
    query_batch_size: int = 64
    learning_rate: float = 1e-3
    reassign_interval: int = 5
    top_k_reassign: int = 2
    drop_tail: bool = True
    weight_init: str = "truncated_normal"
    clamp_targets: bool = False
    label_smoothing: float = 0.0
    prepare_max_samples: int = 1_000_000
    prepare_k: int = 100
    ivf_train_samples: int = 200_000
    ivf_niter: int = 25
    ivf_permute_reps: bool = True
    pq_enabled: bool = False
    pq_m: int = 16
    pq_bits: int = 8
    pq_train_samples: int = 200_000
    assignment_strategy: str = "single"
    assignment_threshold: float = 0.9
    max_assignments: int = 1
    probing_strategy: str = "fixed"
    m_base: int = 5
    m_max: int = 50
    entropy_scale: float = 10.0
    return_candidates_k: int | None = None
    seed: int = 0

    @property
    def metric(self) -> str:
        return DATASETS[self.dataset].metric

    @property
    def index_partitions(self) -> int:
        return int(self.partitions or DATASETS[self.dataset].partitions)


@dataclass
class RouterWeights:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class InitState:
    assignments: np.ndarray
    ivf_centroids: np.ndarray | None = None
    ivf_perms: np.ndarray | None = None
    ivf_inv_perms: np.ndarray | None = None


@dataclass
class FitState:
    exp_dir: Path
    weights: list[RouterWeights]
    lookups: list[tuple[np.ndarray, np.ndarray]]
    point_to_buckets: np.ndarray
    train_seconds: float
    index_seconds: float
    ivf_centroids: np.ndarray | None = None
    ivf_perms: np.ndarray | None = None
    ivf_inv_perms: np.ndarray | None = None
    ivf_list_ids: np.ndarray | None = None
    pq_codebooks: np.ndarray | None = None
    pq_codes: np.ndarray | None = None


@dataclass
class QueryResult:
    probe_depth: int
    recall_at_10: float
    avg_computations: float
    computation_min: float
    computation_max: float
    qps: float | None
    query_mem_delta_mb: float


def _prepare_batches(n: int, batch_size: int, drop_tail: bool) -> tuple[int, list[int]]:
    train_limit = n - (n % batch_size) if drop_tail else n
    return train_limit, list(range(0, train_limit, batch_size))


def _topk_sorted(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.argmax(logits, axis=1, keepdims=True).astype(np.int32, copy=False)
    part = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(logits, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1).astype(np.int32, copy=False)


def _top1_probabilities(logits: np.ndarray) -> np.ndarray:
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits
    exp_shifted = np.exp(shifted, dtype=np.float32)
    return (np.max(exp_shifted, axis=1) / np.maximum(np.sum(exp_shifted, axis=1), 1e-12)).astype(
        np.float32,
        copy=False,
    )


def _softmax_entropy(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits
    exp_shifted = np.exp(shifted, dtype=np.float32)
    probs = exp_shifted / np.maximum(np.sum(exp_shifted, axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    return entropy.astype(np.float32, copy=False), np.max(probs, axis=1).astype(np.float32, copy=False)


def _router_forward(x: np.ndarray, weights: RouterWeights) -> np.ndarray:
    hidden = np.maximum(x @ weights.W1 + weights.b1, 0.0).astype(np.float32, copy=False)
    return (hidden @ weights.W2 + weights.b2).astype(np.float32, copy=False)


def _normalize_if_cosine(arr: np.ndarray, metric: str) -> np.ndarray:
    if str(metric).lower() != "cosine":
        return arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _exact_scores_for_ids(
    *,
    train_view: np.ndarray,
    train_sq_norms: np.ndarray | None,
    query_vec: np.ndarray,
    ids: np.ndarray,
    metric: str,
) -> np.ndarray:
    metric_key = str(metric).lower()
    if metric_key == "cosine":
        return train_view[ids] @ query_vec
    if metric_key == "l2":
        q_norm = float(np.dot(query_vec, query_vec))
        return -(train_sq_norms[ids] + q_norm - 2.0 * (train_view[ids] @ query_vec))
    return train_view[ids] @ query_vec


def _query_buffer(
    *,
    index_full_dataset: bool,
    n_points: int,
    repetitions: int,
    probe_depth: int,
    partitions: int,
    min_reps: int,
    k: int,
) -> tuple[int, int]:
    raw = int(2 * repetitions * n_points * probe_depth / (partitions * max(min_reps, 1)))
    if index_full_dataset:
        aligned = 1024 * (raw // 1024)
        return raw, max(aligned, k)
    return raw, max(1024, int(raw * 2), k)


def _build_inverted(assign: np.ndarray, partitions: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(assign, minlength=partitions).astype(np.int64, copy=False)
    offsets = np.zeros(partitions + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    ids = np.argsort(assign, kind="stable").astype(np.int32, copy=False)
    return offsets, ids


def _build_inverted_multi(ids: np.ndarray, assign: np.ndarray, partitions: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(assign, minlength=partitions).astype(np.int64, copy=False)
    offsets = np.zeros(partitions + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(assign, kind="stable").astype(np.int32, copy=False)
    return offsets, ids[order].astype(np.int32, copy=False)


def _write_training_artifacts(
    exp_dir: Path,
    cfg: RouterConfig,
    weights: list[RouterWeights],
    lookups: list[tuple[np.ndarray, np.ndarray]],
    point_to_buckets: np.ndarray,
    train_assignments: np.ndarray,
    train_seconds: float,
    index_seconds: float,
    *,
    ivf_centroids: np.ndarray | None = None,
    ivf_perms: np.ndarray | None = None,
    ivf_inv_perms: np.ndarray | None = None,
    ivf_list_ids: np.ndarray | None = None,
    pq_codebooks: np.ndarray | None = None,
    pq_codes: np.ndarray | None = None,
) -> None:
    models_dir = ensure_dir(exp_dir / "models")
    lookups_dir = ensure_dir(exp_dir / "staging" / "lookups")
    staging_dir = ensure_dir(exp_dir / "staging")
    write_json(
        exp_dir / "config.json",
        {
            "method": cfg.method,
            "backend": "tensorflow",
            **asdict(cfg),
        },
    )
    np.savez(
        models_dir / "router_weights.npz",
        W1=np.stack([w.W1 for w in weights], axis=0),
        b1=np.stack([w.b1 for w in weights], axis=0),
        W2=np.stack([w.W2 for w in weights], axis=0),
        b2=np.stack([w.b2 for w in weights], axis=0),
    )
    np.save(staging_dir / "assignments.npy", train_assignments.astype(np.int32, copy=False))
    np.save(staging_dir / "point_to_buckets.npy", point_to_buckets.astype(np.int32, copy=False))
    if ivf_centroids is not None:
        np.save(staging_dir / "ivf_centroids.npy", ivf_centroids.astype(np.float32, copy=False))
    if ivf_perms is not None:
        np.save(staging_dir / "ivf_permutations.npy", ivf_perms.astype(np.int32, copy=False))
    if ivf_inv_perms is not None:
        np.save(staging_dir / "ivf_inv_permutations.npy", ivf_inv_perms.astype(np.int32, copy=False))
    if ivf_list_ids is not None:
        np.save(staging_dir / "ivf_list_ids_index.npy", ivf_list_ids.astype(np.int32, copy=False))
    if pq_codebooks is not None:
        np.save(staging_dir / "pq_codebooks.npy", pq_codebooks.astype(np.float32, copy=False))
    if pq_codes is not None:
        np.save(staging_dir / "pq_codes.npy", pq_codes.astype(np.uint8, copy=False))
    for rep, (offsets, ids) in enumerate(lookups):
        np.save(lookups_dir / f"rep{rep}_offsets.npy", offsets.astype(np.int64, copy=False))
        np.save(lookups_dir / f"rep{rep}_ids.npy", ids.astype(np.int32, copy=False))
    write_json(
        exp_dir / "train_metrics.json",
        {
            "dataset": cfg.dataset,
            "method": cfg.method,
            "train_s": train_seconds,
            "full_index_lookup_s": index_seconds,
        },
    )
    write_index_manifest(exp_dir)


def _load_fit_state(exp_dir: Path, cfg: RouterConfig) -> FitState:
    payload = np.load(exp_dir / "models" / "router_weights.npz")
    weights: list[RouterWeights] = []
    for rep in range(payload["W1"].shape[0]):
        weights.append(
            RouterWeights(
                W1=payload["W1"][rep].astype(np.float32, copy=False),
                b1=payload["b1"][rep].astype(np.float32, copy=False),
                W2=payload["W2"][rep].astype(np.float32, copy=False),
                b2=payload["b2"][rep].astype(np.float32, copy=False),
            )
        )
    lookups: list[tuple[np.ndarray, np.ndarray]] = []
    lookups_dir = exp_dir / "staging" / "lookups"
    for rep in range(cfg.repetitions):
        offsets = np.load(lookups_dir / f"rep{rep}_offsets.npy").astype(np.int64, copy=False)
        ids = np.load(lookups_dir / f"rep{rep}_ids.npy").astype(np.int32, copy=False)
        lookups.append((offsets, ids))
    point_to_buckets = np.load(exp_dir / "staging" / "point_to_buckets.npy").astype(np.int32, copy=False)
    staging_dir = exp_dir / "staging"
    metrics_path = exp_dir / "train_metrics.json"
    train_seconds = 0.0
    index_seconds = 0.0
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            train_seconds = float(payload.get("train_s", 0.0))
            index_seconds = float(payload.get("full_index_lookup_s", 0.0))
        except Exception:
            train_seconds = 0.0
            index_seconds = 0.0
    return FitState(
        exp_dir=exp_dir,
        weights=weights,
        lookups=lookups,
        point_to_buckets=point_to_buckets,
        train_seconds=train_seconds,
        index_seconds=index_seconds,
        ivf_centroids=(
            np.load(staging_dir / "ivf_centroids.npy").astype(np.float32, copy=False)
            if (staging_dir / "ivf_centroids.npy").exists()
            else None
        ),
        ivf_perms=(
            np.load(staging_dir / "ivf_permutations.npy").astype(np.int32, copy=False)
            if (staging_dir / "ivf_permutations.npy").exists()
            else None
        ),
        ivf_inv_perms=(
            np.load(staging_dir / "ivf_inv_permutations.npy").astype(np.int32, copy=False)
            if (staging_dir / "ivf_inv_permutations.npy").exists()
            else None
        ),
        ivf_list_ids=(
            np.load(staging_dir / "ivf_list_ids_index.npy").astype(np.int32, copy=False)
            if (staging_dir / "ivf_list_ids_index.npy").exists()
            else None
        ),
        pq_codebooks=(
            np.load(staging_dir / "pq_codebooks.npy").astype(np.float32, copy=False)
            if (staging_dir / "pq_codebooks.npy").exists()
            else None
        ),
        pq_codes=(
            np.load(staging_dir / "pq_codes.npy").astype(np.uint8, copy=False)
            if (staging_dir / "pq_codes.npy").exists()
            else None
        ),
    )


def _isolated_query_memory(
    *,
    exp_dir: Path,
    dataset: str,
    data_root: Path,
    probe_depth: int,
) -> dict[str, float] | None:
    env = os.environ.copy()
    src_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
    cmd = [
        sys.executable,
        "-m",
        "run_router_query_isolated",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--experiment-dir",
        str(exp_dir),
        "--probe-depth",
        str(probe_depth),
    ]
    proc = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True)
    payload = json.loads(proc.stdout.strip())
    if not isinstance(payload, dict):
        return None
    return {str(k): float(v) for k, v in payload.items() if v is not None}


def _kmeans_cache_paths(data_root: Path, cfg: RouterConfig) -> tuple[Path, Path]:
    dataset_dir = data_root / cfg.dataset
    stem = f"kmeans_labels_B{cfg.index_partitions}_seed{cfg.seed}"
    labels_path = dataset_dir / f"{stem}.npy"
    meta_path = dataset_dir / f"{stem}.meta.json"
    return labels_path, meta_path


def _hash_assignments(train: np.ndarray, cfg: RouterConfig) -> InitState:
    n = train.shape[0]
    assignments = np.empty((cfg.repetitions, n), dtype=np.int32)
    for rep in range(cfg.repetitions):
        seed = int(cfg.seed + rep)
        assignments[rep] = np.fromiter(
            (murmurhash3_32(int(i), seed=seed) % cfg.index_partitions for i in range(n)),
            dtype=np.int32,
            count=n,
        )
    return InitState(assignments=assignments)


def _kmeans_assignments(train: np.ndarray, cfg: RouterConfig, data_root: Path) -> InitState:
    labels_path, meta_path = _kmeans_cache_paths(data_root, cfg)
    expected_meta = {
        "dataset": cfg.dataset,
        "metric": cfg.metric,
        "train_rows": int(train.shape[0]),
        "dim": int(train.shape[1]),
        "B": int(cfg.index_partitions),
        "seed": int(cfg.seed),
    }

    def _fit_labels() -> np.ndarray:
        batch_size = min(10_000, max(1_000, train.shape[0]))
        kmeans = MiniBatchKMeans(
            n_clusters=cfg.index_partitions,
            batch_size=batch_size,
            max_iter=100,
            random_state=cfg.seed,
            verbose=0,
        )
        labels = kmeans.fit_predict(train).astype(np.int32, copy=False)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_path, labels)
        write_json(meta_path, expected_meta)
        return labels

    rebuild = True
    if labels_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            labels = np.load(labels_path).astype(np.int32, copy=False)
            rebuild = (
                meta != expected_meta
                or labels.shape[0] != train.shape[0]
                or labels.min(initial=0) < 0
                or int(labels.max(initial=0)) + 1 > cfg.index_partitions
            )
        except Exception:
            rebuild = True
    if rebuild:
        labels = _fit_labels()

    assignments = np.empty((cfg.repetitions, train.shape[0]), dtype=np.int32)
    rng = np.random.default_rng(cfg.seed)
    for rep in range(cfg.repetitions):
        if cfg.ivf_permute_reps:
            perm = rng.permutation(cfg.index_partitions).astype(np.int32, copy=False)
            assignments[rep] = perm[labels]
        else:
            assignments[rep] = labels
    return InitState(assignments=assignments)


def _ivf_assignments(train: np.ndarray, cfg: RouterConfig) -> InitState:
    n, d = train.shape
    train_used = _normalize_if_cosine(train, cfg.metric)
    sample_n = min(cfg.ivf_train_samples, n)
    rng = np.random.default_rng(cfg.seed)
    if sample_n < n:
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        sample = train_used[sample_idx]
    else:
        sample = train_used

    if faiss is not None:
        kmeans = faiss.Kmeans(
            d,
            cfg.index_partitions,
            niter=cfg.ivf_niter,
            verbose=False,
            seed=cfg.seed,
        )
        kmeans.train(sample.astype(np.float32, copy=False))
        _, list_ids = kmeans.index.search(train_used.astype(np.float32, copy=False), 1)
        base_ids = list_ids.reshape(-1).astype(np.int32, copy=False)
        centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(cfg.index_partitions, d)
    else:
        batch_size = min(10_000, max(1_000, sample_n))
        kmeans = MiniBatchKMeans(
            n_clusters=cfg.index_partitions,
            batch_size=batch_size,
            max_iter=cfg.ivf_niter,
            random_state=cfg.seed,
            verbose=0,
        )
        kmeans.fit(sample)
        base_ids = kmeans.predict(train_used).astype(np.int32, copy=False)
        centroids = kmeans.cluster_centers_.astype(np.float32, copy=False)

    assignments = np.empty((cfg.repetitions, n), dtype=np.int32)
    perms = np.empty((cfg.repetitions, cfg.index_partitions), dtype=np.int32)
    inv_perms = np.empty((cfg.repetitions, cfg.index_partitions), dtype=np.int32)
    for rep in range(cfg.repetitions):
        perm = rng.permutation(cfg.index_partitions).astype(np.int32, copy=False)
        perms[rep] = perm
        inv = np.empty_like(perm)
        inv[perm] = np.arange(cfg.index_partitions, dtype=np.int32)
        inv_perms[rep] = inv
        assignments[rep] = perm[base_ids] if cfg.ivf_permute_reps else base_ids
        if not cfg.ivf_permute_reps:
            perms[rep] = np.arange(cfg.index_partitions, dtype=np.int32)
            inv_perms[rep] = np.arange(cfg.index_partitions, dtype=np.int32)
    return InitState(
        assignments=assignments,
        ivf_centroids=centroids,
        ivf_perms=perms,
        ivf_inv_perms=inv_perms,
    )


def _ivf_list_ids(index_train: np.ndarray, centroids: np.ndarray, metric: str) -> np.ndarray:
    train_used = _normalize_if_cosine(index_train, metric)
    centroids_used = _normalize_if_cosine(centroids, metric)
    if faiss is not None:
        index = faiss.IndexFlatL2(centroids_used.shape[1])
        index.add(centroids_used.astype(np.float32, copy=False))
        _, list_ids = index.search(train_used.astype(np.float32, copy=False), 1)
        return list_ids.reshape(-1).astype(np.int32, copy=False)
    x_norm = np.sum(train_used * train_used, axis=1, keepdims=True)
    c_norm = np.sum(centroids_used * centroids_used, axis=1, keepdims=True).T
    d2 = x_norm + c_norm - 2.0 * (train_used @ centroids_used.T)
    return np.argmin(d2, axis=1).astype(np.int32, copy=False)


def _build_pq_artifacts(
    index_train: np.ndarray,
    *,
    centroids: np.ndarray,
    list_ids: np.ndarray,
    cfg: RouterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    train_used = _normalize_if_cosine(index_train, cfg.metric)
    centroids_used = _normalize_if_cosine(centroids, cfg.metric)
    residuals = train_used - centroids_used[list_ids]
    residuals_pad, d_pad = pad_to_m(residuals.astype(np.float32, copy=False), cfg.pq_m)
    sample_n = min(cfg.pq_train_samples, residuals_pad.shape[0])
    rng = np.random.default_rng(cfg.seed)
    if sample_n < residuals_pad.shape[0]:
        sample_idx = rng.choice(residuals_pad.shape[0], size=sample_n, replace=False)
        sample = residuals_pad[sample_idx]
    else:
        sample = residuals_pad
    codebooks, _ = train_global_pq_codebooks(sample, cfg.pq_m, cfg.pq_bits, cfg.seed)
    codes = encode_pq_codes(residuals_pad, codebooks, d_pad)
    return codebooks.astype(np.float32, copy=False), codes.astype(np.uint8, copy=False)


def _router_weights(tf, d: int, hidden: int, partitions: int, weight_init: str):
    if weight_init == "truncated_normal":
        W1 = tf.compat.v1.get_variable(
            "W1",
            initializer=tf.compat.v1.truncated_normal([d, hidden], stddev=0.05, dtype=tf.float32),
        )
        b1 = tf.compat.v1.get_variable(
            "b1",
            initializer=tf.compat.v1.truncated_normal([hidden], stddev=0.05, dtype=tf.float32),
        )
        W2 = tf.compat.v1.get_variable(
            "W2",
            initializer=tf.compat.v1.truncated_normal([hidden, partitions], stddev=0.05, dtype=tf.float32),
        )
        b2 = tf.compat.v1.get_variable(
            "b2",
            initializer=tf.compat.v1.truncated_normal([partitions], stddev=0.05, dtype=tf.float32),
        )
    else:
        W1 = tf.compat.v1.get_variable(
            "W1", shape=[d, hidden], initializer=tf.compat.v1.initializers.glorot_uniform()
        )
        b1 = tf.compat.v1.get_variable("b1", shape=[hidden], initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(
            "W2", shape=[hidden, partitions], initializer=tf.compat.v1.initializers.glorot_uniform()
        )
        b2 = tf.compat.v1.get_variable("b2", shape=[partitions], initializer=tf.compat.v1.zeros_initializer())
    return W1, b1, W2, b2


def _build_graph(
    tf,
    *,
    d: int,
    hidden: int,
    partitions: int,
    assign: np.ndarray,
    neighbors_per_item: int,
    weight_init: str,
    clamp_targets: bool,
    label_smoothing: float,
    learning_rate: float,
    top_k: int,
):
    graph = tf.Graph()
    with graph.as_default():
        x_pl = tf.compat.v1.placeholder(tf.float32, shape=(None, d), name="x")
        neighbors_pl = tf.compat.v1.placeholder(tf.int64, shape=(None,), name="neighbors")
        lookup_init = tf.constant(assign, dtype=tf.int64)
        lookup_var = tf.compat.v1.get_variable("lookup_assignments", initializer=lookup_init, trainable=False)
        lookup_update_pl = tf.compat.v1.placeholder(tf.int64, shape=(assign.shape[0],), name="lookup_update")
        lookup_assign_op = tf.compat.v1.assign(lookup_var, lookup_update_pl)

        W1, b1, W2, b2 = _router_weights(tf, d, hidden, partitions, weight_init)
        hidden_act = tf.nn.relu(tf.matmul(x_pl, W1) + b1)
        logits = tf.matmul(hidden_act, W2) + b2

        batch_rows = tf.shape(neighbors_pl, out_type=tf.int64)[0] // neighbors_per_item
        row_ids = tf.reshape(tf.tile(tf.range(batch_rows, dtype=tf.int64)[:, None], [1, neighbors_per_item]), [-1])
        bucket_ids = tf.gather(lookup_var, neighbors_pl)
        scatter_indices = tf.stack([row_ids, bucket_ids], axis=1)
        sparse = tf.SparseTensor(
            indices=scatter_indices,
            values=tf.ones_like(row_ids, dtype=tf.float32),
            dense_shape=[batch_rows, partitions],
        )
        targets = tf.compat.v1.sparse_tensor_to_dense(sparse, validate_indices=False)
        if clamp_targets:
            targets = tf.minimum(targets, 1.0)
        if label_smoothing > 0.0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
        optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        topk = tf.nn.top_k(logits, k=top_k, sorted=True)

    return {
        "graph": graph,
        "x_pl": x_pl,
        "neighbors_pl": neighbors_pl,
        "lookup_update_pl": lookup_update_pl,
        "lookup_assign_op": lookup_assign_op,
        "logits": logits,
        "loss": loss,
        "optimiser": optimiser,
        "topk_indices": topk.indices,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }


def _greedy_reassign_python(top_candidates: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n, k = top_candidates.shape
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        best = 0
        best_count = counts[top_candidates[i, 0]]
        for j in range(1, k):
            bucket = top_candidates[i, j]
            bucket_count = counts[bucket]
            if bucket_count < best_count:
                best = j
                best_count = bucket_count
        chosen = top_candidates[i, best]
        out[i] = chosen
        counts[chosen] += 1
    return out


if _HAS_NUMBA:
    _greedy_reassign_numba = njit(cache=True)(_greedy_reassign_python)
else:  # pragma: no cover
    _greedy_reassign_numba = None


def _greedy_reassign(top_candidates: np.ndarray, partitions: int) -> np.ndarray:
    counts = np.zeros(partitions, dtype=np.int64)
    if _greedy_reassign_numba is not None:
        return _greedy_reassign_numba(top_candidates.astype(np.int32, copy=False), counts)
    return _greedy_reassign_python(top_candidates.astype(np.int32, copy=False), counts)


def _greedy_reassign_with_counts(top_candidates: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if _greedy_reassign_numba is not None:
        return _greedy_reassign_numba(top_candidates.astype(np.int32, copy=False), counts)
    return _greedy_reassign_python(top_candidates.astype(np.int32, copy=False), counts)


def _fit_single_rep(
    rep: int,
    train: np.ndarray,
    neighbors: np.ndarray,
    initial_assign: np.ndarray,
    cfg: RouterConfig,
) -> tuple[np.ndarray, RouterWeights]:
    tf = _ensure_tf_graph()
    n, d = train.shape
    train_limit, batch_starts = _prepare_batches(n, cfg.batch_size, cfg.drop_tail)
    assign = initial_assign.astype(np.int64, copy=True)
    tf.compat.v1.set_random_seed(cfg.seed + rep)
    np.random.seed(cfg.seed + rep)
    ops = _build_graph(
        tf,
        d=d,
        hidden=cfg.hidden,
        partitions=cfg.index_partitions,
        assign=assign,
        neighbors_per_item=neighbors.shape[1],
        weight_init=cfg.weight_init,
        clamp_targets=cfg.clamp_targets,
        label_smoothing=cfg.label_smoothing,
        learning_rate=cfg.learning_rate,
        top_k=cfg.top_k_reassign,
    )
    cfg_tf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    cfg_tf.gpu_options.allow_growth = True  # type: ignore[attr-defined]
    intra = os.environ.get("TF_NUM_INTRAOP_THREADS")
    inter = os.environ.get("TF_NUM_INTEROP_THREADS")
    if intra:
        cfg_tf.intra_op_parallelism_threads = int(intra)
    if inter:
        cfg_tf.inter_op_parallelism_threads = int(inter)
    with tf.compat.v1.Session(graph=ops["graph"], config=cfg_tf) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(1, cfg.epochs + 1):
            loss_last = 0.0
            t0 = time.perf_counter()
            for start in batch_starts:
                end = start + cfg.batch_size
                _, loss_last = sess.run(
                    [ops["optimiser"], ops["loss"]],
                    feed_dict={
                        ops["x_pl"]: train[start:end],
                        ops["neighbors_pl"]: neighbors[start:end].reshape(-1),
                    },
                )
            train_s = time.perf_counter() - t0

            moved = 0.0
            reassign_s = 0.0
            if cfg.reassign_interval > 0 and epoch % cfg.reassign_interval == 0:
                t0 = time.perf_counter()
                top_candidates = np.empty((train_limit, cfg.top_k_reassign), dtype=np.int32)
                fill = 0
                for start in range(0, train_limit, cfg.reassign_batch_size):
                    end = min(start + cfg.reassign_batch_size, train_limit)
                    top_batch = sess.run(
                        ops["topk_indices"],
                        feed_dict={ops["x_pl"]: train[start:end]},
                    )
                    top_candidates[fill : fill + top_batch.shape[0]] = top_batch.astype(np.int32, copy=False)
                    fill += top_batch.shape[0]
                new_assign = assign.copy()
                new_assign[:train_limit] = _greedy_reassign(top_candidates, cfg.index_partitions)
                moved = float(np.mean(new_assign != assign))
                assign = new_assign
                sess.run(ops["lookup_assign_op"], feed_dict={ops["lookup_update_pl"]: assign})
                reassign_s = time.perf_counter() - t0
            info(
                f"[{cfg.method} rep {rep + 1}/{cfg.repetitions}] "
                f"epoch={epoch:02d} loss={loss_last:.4f} train_s={train_s:.1f} "
                f"reassign_s={reassign_s:.1f} moved={moved:.3f}"
            )

        W1, b1, W2, b2 = sess.run([ops["W1"], ops["b1"], ops["W2"], ops["b2"]])
    return (
        assign.astype(np.int32, copy=False),
        RouterWeights(
            W1=W1.astype(np.float32, copy=False),
            b1=b1.astype(np.float32, copy=False),
            W2=W2.astype(np.float32, copy=False),
            b2=b2.astype(np.float32, copy=False),
        ),
    )


def _full_index_lookups(
    index_train: np.ndarray,
    weights: list[RouterWeights],
    cfg: RouterConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    lookups: list[tuple[np.ndarray, np.ndarray]] = []
    point_to_buckets = np.empty((index_train.shape[0], cfg.repetitions), dtype=np.int32)
    adaptive_lookup = cfg.assignment_strategy == "confidence_threshold" and cfg.max_assignments > 1
    max_assignments = int(max(1, min(cfg.max_assignments, cfg.index_partitions)))
    for rep, rep_weights in enumerate(weights):
        if adaptive_lookup:
            ids_parts: list[np.ndarray] = []
            bucket_parts: list[np.ndarray] = []
            top1_assign = np.empty(index_train.shape[0], dtype=np.int32)
            fill = 0
            for start in range(0, index_train.shape[0], cfg.reassign_batch_size):
                end = min(start + cfg.reassign_batch_size, index_train.shape[0])
                logits = _router_forward(index_train[start:end], rep_weights)
                top_batch = _topk_sorted(logits, max_assignments)
                top1 = top_batch[:, 0].astype(np.int32, copy=False)
                top1_assign[fill : fill + top1.shape[0]] = top1
                top1_prob = _top1_probabilities(logits)
                batch_ids = np.arange(start, end, dtype=np.int32)
                ids_parts.append(batch_ids)
                bucket_parts.append(top1)
                low_conf = top1_prob < float(cfg.assignment_threshold)
                if np.any(low_conf):
                    exp_ids = batch_ids[low_conf]
                    exp_top = top_batch[low_conf]
                    for j in range(1, max_assignments):
                        ids_parts.append(exp_ids)
                        bucket_parts.append(exp_top[:, j].astype(np.int32, copy=False))
                fill += top1.shape[0]
            point_to_buckets[:, rep] = top1_assign
            rep_ids = np.concatenate(ids_parts, axis=0).astype(np.int32, copy=False)
            rep_buckets = np.concatenate(bucket_parts, axis=0).astype(np.int32, copy=False)
            lookups.append(_build_inverted_multi(rep_ids, rep_buckets, cfg.index_partitions))
        else:
            counts = np.zeros(cfg.index_partitions, dtype=np.int64)
            assign = np.empty(index_train.shape[0], dtype=np.int32)
            fill = 0
            for start in range(0, index_train.shape[0], cfg.reassign_batch_size):
                end = min(start + cfg.reassign_batch_size, index_train.shape[0])
                logits = _router_forward(index_train[start:end], rep_weights)
                top_batch = _topk_sorted(logits, cfg.top_k_reassign)
                assign_batch = _greedy_reassign_with_counts(top_batch, counts)
                assign[fill : fill + assign_batch.shape[0]] = assign_batch
                fill += top_batch.shape[0]
            point_to_buckets[:, rep] = assign
            lookups.append(_build_inverted(assign, cfg.index_partitions))
    return lookups, point_to_buckets


def _collect_candidates(
    topm: np.ndarray,
    *,
    lookups: list[tuple[np.ndarray, np.ndarray]],
    min_reps: int,
    buffer: int,
) -> np.ndarray:
    gathered: list[np.ndarray] = []
    for rep, row in enumerate(topm):
        offsets, ids = lookups[rep]
        n_buckets = int(offsets.shape[0] - 1)
        for bucket in row.tolist():
            if bucket < 0 or bucket >= n_buckets:
                continue
            start = int(offsets[bucket])
            end = int(offsets[bucket + 1])
            if end > start:
                gathered.append(ids[start:end])
    if not gathered:
        return np.empty(0, dtype=np.int32)
    all_ids = np.concatenate(gathered, axis=0)
    uniq, freq = np.unique(all_ids, return_counts=True)
    keep = freq >= int(min_reps)
    if not np.any(keep):
        return np.empty(0, dtype=np.int32)
    uniq = uniq[keep].astype(np.int32, copy=False)
    freq = freq[keep]
    if uniq.size > buffer:
        top_idx = np.argpartition(-freq, buffer - 1)[:buffer]
        uniq = uniq[top_idx]
        freq = freq[top_idx]
    order = np.lexsort((uniq, -freq))
    return uniq[order].astype(np.int32, copy=False)


class RouterFamily:
    name: str = ""
    init_mode: str = ""

    def __init__(self, cfg: RouterConfig) -> None:
        self.cfg = cfg

    def _initial_assignments(self, train: np.ndarray, data_root: Path) -> InitState:
        if self.cfg.init_mode == "hash":
            return _hash_assignments(train, self.cfg)
        if self.cfg.init_mode == "kmeans":
            return _kmeans_assignments(train, self.cfg, data_root)
        if self.cfg.init_mode == "ivf":
            return _ivf_assignments(train, self.cfg)
        raise ValueError(f"Unsupported init_mode: {self.cfg.init_mode}")

    def fit(self, bundle: LearnedDataset, exp_dir: Path, *, data_root: Path) -> FitState:
        banner("method", f"{self.cfg.method} | dataset={self.cfg.dataset} | init={self.cfg.init_mode}")
        info(
            f"router_train={bundle.router_train.shape[0]}, index_train={bundle.index_train.shape[0]}, "
            f"queries={bundle.queries.shape[0]}, dim={bundle.router_train.shape[1]}"
        )
        init_state = self._initial_assignments(bundle.router_train, data_root)
        train_assignments = np.empty_like(init_state.assignments)
        weights: list[RouterWeights] = []
        t0 = time.perf_counter()
        for rep in range(self.cfg.repetitions):
            learned_assign, rep_weights = _fit_single_rep(
                rep,
                bundle.router_train,
                bundle.router_neighbors,
                init_state.assignments[rep],
                self.cfg,
            )
            train_assignments[rep] = learned_assign
            weights.append(rep_weights)
        train_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        lookups, point_to_buckets = _full_index_lookups(bundle.index_train, weights, self.cfg)
        index_seconds = time.perf_counter() - t0
        ivf_list_ids = None
        pq_codebooks = None
        pq_codes = None
        if init_state.ivf_centroids is not None:
            ivf_list_ids = _ivf_list_ids(bundle.index_train, init_state.ivf_centroids, self.cfg.metric)
            if self.cfg.pq_enabled:
                pq_codebooks, pq_codes = _build_pq_artifacts(
                    bundle.index_train,
                    centroids=init_state.ivf_centroids,
                    list_ids=ivf_list_ids,
                    cfg=self.cfg,
                )
        _write_training_artifacts(
            exp_dir,
            self.cfg,
            weights,
            lookups,
            point_to_buckets,
            train_assignments,
            train_seconds,
            index_seconds,
            ivf_centroids=init_state.ivf_centroids,
            ivf_perms=init_state.ivf_perms,
            ivf_inv_perms=init_state.ivf_inv_perms,
            ivf_list_ids=ivf_list_ids,
            pq_codebooks=pq_codebooks,
            pq_codes=pq_codes,
        )
        write_index_manifest(exp_dir)
        return FitState(
            exp_dir=exp_dir,
            weights=weights,
            lookups=lookups,
            point_to_buckets=point_to_buckets,
            train_seconds=train_seconds,
            index_seconds=index_seconds,
            ivf_centroids=init_state.ivf_centroids,
            ivf_perms=init_state.ivf_perms,
            ivf_inv_perms=init_state.ivf_inv_perms,
            ivf_list_ids=ivf_list_ids,
            pq_codebooks=pq_codebooks,
            pq_codes=pq_codes,
        )

    def evaluate(self, bundle: LearnedDataset, state: FitState, *, probe_depth: int) -> QueryResult:
        metric = self.cfg.metric
        adaptive_probing = self.cfg.probing_strategy == "entropy_adaptive"
        m_cap = int(min(self.cfg.index_partitions, self.cfg.m_max)) if adaptive_probing else int(probe_depth)
        _, buffer = _query_buffer(
            index_full_dataset=True,
            n_points=bundle.index_train.shape[0],
            repetitions=self.cfg.repetitions,
            probe_depth=m_cap,
            partitions=self.cfg.index_partitions,
            min_reps=self.cfg.min_reps,
            k=self.cfg.k,
        )
        train_view = bundle.index_train.astype(np.float32, copy=False)
        query_view = bundle.queries.astype(np.float32, copy=False)
        train_sq_norms = None
        if metric == "cosine":
            train_view = _normalize_if_cosine(bundle.index_train, metric)
            query_view = _normalize_if_cosine(bundle.queries, metric)
        elif metric == "l2":
            train_sq_norms = np.sum(bundle.index_train * bundle.index_train, axis=1).astype(np.float32, copy=False)
        pq_query_view = query_view

        mem_before = _rss_mb()
        t0 = time.perf_counter()
        retrieved = np.full((bundle.queries.shape[0], self.cfg.k), -1, dtype=np.int32)
        candidate_counts = np.zeros(bundle.queries.shape[0], dtype=np.int32)
        for start in range(0, bundle.queries.shape[0], self.cfg.query_batch_size):
            end = min(start + self.cfg.query_batch_size, bundle.queries.shape[0])
            batch = bundle.queries[start:end]
            batch_query_view = query_view[start:end]
            batch_pq_query_view = pq_query_view[start:end]
            per_rep_topm: list[np.ndarray] = []
            entropy_sum = np.zeros(end - start, dtype=np.float32)
            for rep_weights in state.weights:
                logits = _router_forward(batch, rep_weights)
                per_rep_topm.append(_topk_sorted(logits, m_cap))
                if adaptive_probing:
                    ent, _ = _softmax_entropy(logits)
                    entropy_sum += ent
            stacked = np.stack(per_rep_topm, axis=1)  # (batch, R, m)
            probe_counts = np.full(end - start, int(probe_depth), dtype=np.int32)
            if adaptive_probing:
                entropy_avg = entropy_sum / max(self.cfg.repetitions, 1)
                entropy_norm = np.clip(entropy_avg / max(np.log(float(self.cfg.index_partitions)), 1e-12), 0.0, None)
                probe_counts = np.floor(
                    float(self.cfg.m_base) + float(self.cfg.entropy_scale) * (np.exp(entropy_norm) - 1.0)
                ).astype(np.int32)
                probe_counts = np.clip(probe_counts, int(self.cfg.m_base), int(m_cap))
            for local_idx in range(stacked.shape[0]):
                m_i = int(probe_counts[local_idx])
                candidates = _collect_candidates(
                    stacked[local_idx, :, :m_i],
                    lookups=state.lookups,
                    min_reps=self.cfg.min_reps,
                    buffer=buffer,
                )
                candidate_counts[start + local_idx] = int(candidates.shape[0])
                if candidates.size == 0:
                    continue
                rerank_ids = candidates
                if self.cfg.pq_enabled:
                    if state.pq_codebooks is None or state.pq_codes is None or state.ivf_centroids is None or state.ivf_list_ids is None:
                        raise RuntimeError("PQ evaluation requires PQ artifacts and IVF centroids/list ids.")
                    candidate_list_ids = state.ivf_list_ids[candidates]
                    uniq_lists = np.unique(candidate_list_ids)
                    approx_scores = np.empty(candidates.shape[0], dtype=np.float32)
                    d_pad = int(state.pq_codebooks.shape[0] * state.pq_codebooks.shape[2])
                    for list_id in uniq_lists.tolist():
                        mask = candidate_list_ids == int(list_id)
                        residual_q = batch_pq_query_view[local_idx] - state.ivf_centroids[int(list_id)]
                        table = adc_table(residual_q, state.pq_codebooks, d_pad)
                        approx_scores[mask] = -adc_distances(state.pq_codes[candidates[mask]], table)
                    shortlist = int(
                        min(
                            candidates.shape[0],
                            max(
                                self.cfg.k,
                                int(self.cfg.return_candidates_k or max(self.cfg.k * 16, 256)),
                            ),
                        )
                    )
                    if shortlist < candidates.shape[0]:
                        top_short = np.argpartition(-approx_scores, kth=shortlist - 1)[:shortlist]
                        rerank_ids = candidates[top_short].astype(np.int32, copy=False)
                scores = _exact_scores_for_ids(
                    train_view=train_view,
                    train_sq_norms=train_sq_norms,
                    query_vec=batch_query_view[local_idx],
                    ids=rerank_ids,
                    metric=metric,
                )
                top_n = min(self.cfg.k, rerank_ids.shape[0])
                top_idx = np.argpartition(-scores, kth=top_n - 1)[:top_n]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                retrieved[start + local_idx, :top_n] = rerank_ids[top_idx]
        total_s = time.perf_counter() - t0
        mem_after = _rss_mb()

        recall = recall_at_k(retrieved, bundle.eval_neighbors, self.cfg.k)
        qps = float(bundle.queries.shape[0] / total_s) if total_s > 0 else None
        return QueryResult(
            probe_depth=probe_depth,
            recall_at_10=recall,
            avg_computations=float(np.mean(candidate_counts)),
            computation_min=float(np.min(candidate_counts)),
            computation_max=float(np.max(candidate_counts)),
            qps=qps,
            query_mem_delta_mb=max(0.0, float(mem_after) - float(mem_before)),
        )

    def run_competitiveness(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool = False,
    ) -> dict[str, object]:
        bundle = load_learned_dataset(
            self.cfg.dataset,
            data_root,
            max_samples=self.cfg.prepare_max_samples,
            k=self.cfg.prepare_k,
            seed=self.cfg.seed,
            force_prepare=force_prepare,
        )
        state = self.fit(bundle, exp_dir, data_root=data_root)
        query_result = self.evaluate(bundle, state, probe_depth=int(self.cfg.probe_depth))
        manifest = json.loads((state.exp_dir / "index_manifest.json").read_text(encoding="utf-8"))
        isolated_mem = _isolated_query_memory(
            exp_dir=state.exp_dir,
            dataset=self.cfg.dataset,
            data_root=data_root,
            probe_depth=int(self.cfg.probe_depth),
        )
        query_mem_delta_mb = query_result.query_mem_delta_mb
        query_mem_delta_mode = QUERY_MEM_MODE_IN_PROCESS
        if isolated_mem is not None and "query_mem_delta_mb_isolated" in isolated_mem:
            query_mem_delta_mb = float(isolated_mem["query_mem_delta_mb_isolated"])
            query_mem_delta_mode = QUERY_MEM_MODE_ISOLATED
        row: dict[str, object] = {
            "dataset": self.cfg.dataset,
            "method": self.cfg.method,
            "seed": int(self.cfg.seed),
            "probe_depth": int(self.cfg.probe_depth),
            "recall_at_10": query_result.recall_at_10,
            "avg_computations": query_result.avg_computations,
            "computation_min": query_result.computation_min,
            "computation_max": query_result.computation_max,
            "avg_computations_mode": AVG_COMPUTATIONS_MODE_ROUTER,
            "avg_candidates": query_result.avg_computations,
            "avg_candidates_mode": AVG_COMPUTATIONS_MODE_ROUTER,
            "qps": query_result.qps,
            "index_overhead_mb": float(manifest["index_overhead_mb"]),
            "index_overhead_mode": INDEX_OVERHEAD_MODE,
            "serving_footprint_mb": float(manifest["serving_footprint_mb"]),
            "serving_footprint_mode": SERVING_FOOTPRINT_MODE,
            "index_size_mb": float(manifest["index_overhead_mb"]),
            "index_size_mode": INDEX_OVERHEAD_MODE,
            "query_mem_delta_mb": query_mem_delta_mb,
            "query_mem_delta_mode": query_mem_delta_mode,
            "train_s": state.train_seconds,
            "experiment_dir": str(state.exp_dir),
        }
        if isolated_mem is not None:
            row.update(isolated_mem)
        print_table(
            f"{self.cfg.method} | competitiveness",
            ["metric", "value"],
            [
                ["Recall@10", f"{query_result.recall_at_10:.4f}"],
                ["Avg computations", f"{query_result.avg_computations:.1f}"],
                ["QPS", f"{query_result.qps:.1f}" if query_result.qps is not None else "nan"],
                ["Index overhead (MB)", f"{float(manifest['index_overhead_mb']):.3f}"],
                ["Serving footprint (MB)", f"{float(manifest['serving_footprint_mb']):.3f}"],
            ],
        )
        write_json(state.exp_dir / "competitiveness_summary.json", row)
        return row

    def run_initialization_sweep(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        probes: list[int],
        force_prepare: bool = False,
    ) -> list[dict[str, object]]:
        bundle = load_learned_dataset(
            self.cfg.dataset,
            data_root,
            max_samples=self.cfg.prepare_max_samples,
            k=self.cfg.prepare_k,
            seed=self.cfg.seed,
            force_prepare=force_prepare,
        )
        state = self.fit(bundle, exp_dir, data_root=data_root)
        rows: list[dict[str, object]] = []
        sweep_payload: list[dict[str, object]] = []
        for probe_depth in probes:
            query_result = self.evaluate(bundle, state, probe_depth=probe_depth)
            manifest = json.loads((state.exp_dir / "index_manifest.json").read_text(encoding="utf-8"))
            isolated_mem = _isolated_query_memory(
                exp_dir=state.exp_dir,
                dataset=self.cfg.dataset,
                data_root=data_root,
                probe_depth=int(probe_depth),
            )
            query_mem_delta_mb = query_result.query_mem_delta_mb
            query_mem_delta_mode = QUERY_MEM_MODE_IN_PROCESS
            if isolated_mem is not None and "query_mem_delta_mb_isolated" in isolated_mem:
                query_mem_delta_mb = float(isolated_mem["query_mem_delta_mb_isolated"])
                query_mem_delta_mode = QUERY_MEM_MODE_ISOLATED
            row = {
                "dataset": self.cfg.dataset,
                "method": self.cfg.method,
                "seed": int(self.cfg.seed),
                "probe_depth": int(probe_depth),
                "recall_at_10": query_result.recall_at_10,
                "avg_computations": query_result.avg_computations,
                "avg_computations_mode": AVG_COMPUTATIONS_MODE_ROUTER,
                "avg_candidates": query_result.avg_computations,
                "avg_candidates_mode": AVG_COMPUTATIONS_MODE_ROUTER,
                "qps": query_result.qps,
                "index_overhead_mb": float(manifest["index_overhead_mb"]),
                "index_overhead_mode": INDEX_OVERHEAD_MODE,
                "serving_footprint_mb": float(manifest["serving_footprint_mb"]),
                "serving_footprint_mode": SERVING_FOOTPRINT_MODE,
                "index_size_mb": float(manifest["index_overhead_mb"]),
                "index_size_mode": INDEX_OVERHEAD_MODE,
                "query_mem_delta_mb": query_mem_delta_mb,
                "query_mem_delta_mode": query_mem_delta_mode,
                "train_s": state.train_seconds,
                "experiment_dir": str(state.exp_dir),
            }
            if isolated_mem is not None:
                row.update(isolated_mem)
            rows.append(row)
            sweep_payload.append(row)
            print_table(
                f"{self.cfg.method} @ m={probe_depth}",
                ["metric", "value"],
                [
                    ["Recall@10", f"{query_result.recall_at_10:.4f}"],
                    ["Avg computations", f"{query_result.avg_computations:.1f}"],
                    ["QPS", f"{query_result.qps:.1f}" if query_result.qps is not None else "nan"],
                    ["Index overhead (MB)", f"{float(manifest['index_overhead_mb']):.3f}"],
                    ["Serving footprint (MB)", f"{float(manifest['serving_footprint_mb']):.3f}"],
                ],
            )
        write_json(state.exp_dir / "query_sweep.json", {"rows": sweep_payload})
        return rows


def make_router_method(method: str, dataset: str, *, seed: int = 0, m_max: int | None = None) -> RouterFamily:
    key = method.strip().upper()
    if key == "BLISS":
        cfg = RouterConfig(dataset=dataset, method="BLISS", init_mode="hash", seed=seed)
        return RouterFamily(cfg)
    if key in {"BLISS-KMEANS", "BLISS_KMEANS"}:
        cfg = RouterConfig(dataset=dataset, method="BLISS-KMeans", init_mode="kmeans", seed=seed)
        return RouterFamily(cfg)
    if key == "MLP-IVF":
        cfg = RouterConfig(dataset=dataset, method="MLP-IVF", init_mode="ivf", seed=seed)
        return RouterFamily(cfg)
    if key in {"MLP-IVFPQ", "MLP_IVFPQ"}:
        cfg = RouterConfig(
            dataset=dataset,
            method="MLP-IVFPQ",
            init_mode="ivf",
            pq_enabled=True,
            seed=seed,
        )
        return RouterFamily(cfg)
    _m_max = m_max if m_max is not None else 10
    if key == "ADAPTIVF":
        cfg = RouterConfig(
            dataset=dataset,
            method="AdaptIVF",
            init_mode="ivf",
            repetitions=1,
            min_reps=1,
            ivf_permute_reps=False,
            pq_enabled=False,
            assignment_strategy="confidence_threshold",
            assignment_threshold=0.75,
            max_assignments=3,
            probing_strategy="entropy_adaptive",
            m_base=5,
            m_max=_m_max,
            entropy_scale=10.0,
            seed=seed,
        )
        return RouterFamily(cfg)
    if key in {"ADAPTIVF-STATIC", "ADAPTIVF_STATIC"}:
        cfg = RouterConfig(
            dataset=dataset,
            method="AdaptIVF-Static",
            init_mode="ivf",
            repetitions=1,
            min_reps=1,
            ivf_permute_reps=False,
            pq_enabled=False,
            assignment_strategy="confidence_threshold",
            assignment_threshold=0.75,
            max_assignments=3,
            probing_strategy="entropy_adaptive",
            m_base=5,
            m_max=_m_max,
            entropy_scale=10.0,
            reassign_interval=0,
            seed=seed,
        )
        return RouterFamily(cfg)
    if key in {"ADAPTIVF-A4", "ADAPTIVF_A4"}:
        cfg = RouterConfig(
            dataset=dataset,
            method="AdaptIVF-A4",
            init_mode="ivf",
            repetitions=1,
            min_reps=1,
            ivf_permute_reps=False,
            pq_enabled=False,
            assignment_strategy="confidence_threshold",
            assignment_threshold=0.75,
            max_assignments=4,
            probing_strategy="entropy_adaptive",
            m_base=5,
            m_max=_m_max,
            entropy_scale=10.0,
            seed=seed,
        )
        return RouterFamily(cfg)
    if key in {"ADAPTIVF+PQ", "ADAPTIVF_PQ"}:
        cfg = RouterConfig(
            dataset=dataset,
            method="AdaptIVF+PQ",
            init_mode="ivf",
            repetitions=1,
            min_reps=1,
            ivf_permute_reps=False,
            pq_enabled=True,
            assignment_strategy="confidence_threshold",
            assignment_threshold=0.75,
            max_assignments=3,
            probing_strategy="entropy_adaptive",
            m_base=5,
            m_max=_m_max,
            entropy_scale=10.0,
            seed=seed,
        )
        return RouterFamily(cfg)
    if key in {"ADAPTIVF-STATIC+PQ", "ADAPTIVF_STATIC+PQ", "ADAPTIVF_STATIC_PQ"}:
        cfg = RouterConfig(
            dataset=dataset,
            method="AdaptIVF-Static+PQ",
            init_mode="ivf",
            repetitions=1,
            min_reps=1,
            ivf_permute_reps=False,
            pq_enabled=True,
            assignment_strategy="confidence_threshold",
            assignment_threshold=0.75,
            max_assignments=3,
            probing_strategy="entropy_adaptive",
            m_base=5,
            m_max=_m_max,
            entropy_scale=10.0,
            reassign_interval=0,
            seed=seed,
        )
        return RouterFamily(cfg)
    if key in {"ADAPTIVF-A4+PQ", "ADAPTIVF_A4+PQ", "ADAPTIVF_A4_PQ"}:
        cfg = RouterConfig(
            dataset=dataset,
            method="AdaptIVF-A4+PQ",
            init_mode="ivf",
            repetitions=1,
            min_reps=1,
            ivf_permute_reps=False,
            pq_enabled=True,
            assignment_strategy="confidence_threshold",
            assignment_threshold=0.75,
            max_assignments=4,
            probing_strategy="entropy_adaptive",
            m_base=5,
            m_max=_m_max,
            entropy_scale=10.0,
            seed=seed,
        )
        return RouterFamily(cfg)
    # --- m80 variants: explicit m_max=80 for analysis / discussion ---
    if key in {"ADAPTIVF-M80", "ADAPTIVF_M80"}:
        cfg = RouterConfig(
            dataset=dataset,
            method="AdaptIVF-m80",
            init_mode="ivf",
            repetitions=1,
            min_reps=1,
            ivf_permute_reps=False,
            pq_enabled=False,
            assignment_strategy="confidence_threshold",
            assignment_threshold=0.75,
            max_assignments=3,
            probing_strategy="entropy_adaptive",
            m_base=5,
            m_max=80,
            entropy_scale=10.0,
            seed=seed,
        )
        return RouterFamily(cfg)
    if key in {"ADAPTIVF+PQ-M80", "ADAPTIVF_PQ_M80", "ADAPTIVF+PQ_M80"}:
        cfg = RouterConfig(
            dataset=dataset,
            method="AdaptIVF+PQ-m80",
            init_mode="ivf",
            repetitions=1,
            min_reps=1,
            ivf_permute_reps=False,
            pq_enabled=True,
            assignment_strategy="confidence_threshold",
            assignment_threshold=0.75,
            max_assignments=3,
            probing_strategy="entropy_adaptive",
            m_base=5,
            m_max=80,
            entropy_scale=10.0,
            seed=seed,
        )
        return RouterFamily(cfg)
    raise ValueError(f"Unsupported router method: {method}")

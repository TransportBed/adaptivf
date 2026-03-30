from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from artifacts import ensure_dir, write_json
from console import banner, info, print_table
from index_manifest import write_index_manifest
from measurement_contract import (
    INDEX_OVERHEAD_MODE,
    QUERY_MEM_MODE_ISOLATED,
    QUERY_MEM_MODE_IN_PROCESS,
    SERVING_FOOTPRINT_MODE,
    hnsw_candidate_stats,
    ivf_candidate_stats,
    recall_at_k,
)
from presets import DATASETS


def _rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return float(parts[1]) / 1024.0
    return float("nan")


def _isolated_query_memory(
    *,
    exp_dir: Path,
    dataset: str,
    data_root: Path,
) -> dict[str, float] | None:
    env = os.environ.copy()
    src_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
    cmd = [
        sys.executable,
        "-m",
        "run_faiss_query_isolated",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--experiment-dir",
        str(exp_dir),
    ]
    proc = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True)
    payload = json.loads(proc.stdout.strip())
    if not isinstance(payload, dict):
        return None
    return {str(k): float(v) for k, v in payload.items() if v is not None}


def _normalize_if_cosine(arr: np.ndarray, metric: str) -> np.ndarray:
    if metric != "cosine":
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _pad_vectors(arr: np.ndarray, m: int) -> np.ndarray:
    if arr.shape[1] % m == 0:
        return arr
    new_dim = int(math.ceil(arr.shape[1] / float(m)) * m)
    out = np.zeros((arr.shape[0], new_dim), dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    return out


def _exact_scores_for_ids(
    *,
    train_view: np.ndarray,
    train_sq_norms: np.ndarray | None,
    query_vec: np.ndarray,
    ids: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric in {"ip", "cosine"}:
        return train_view[ids] @ query_vec
    if metric == "l2":
        if train_sq_norms is None:
            raise ValueError("train_sq_norms required for l2 rerank")
        q_norm = float(np.dot(query_vec, query_vec))
        return -(train_sq_norms[ids] + q_norm - 2.0 * (train_view[ids] @ query_vec))
    raise ValueError(f"Unsupported metric: {metric}")


@dataclass(frozen=True)
class FaissConfig:
    dataset: str
    k: int = 10
    nprobe: int | None = 10
    pq_m: int = 16
    pq_bits: int = 8
    train_samples: int = 200_000
    index_seed: int = 0
    ivf_niter: int = 25
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    return_candidates_k: int | None = None

    @property
    def partitions(self) -> int:
        return DATASETS[self.dataset].partitions

    @property
    def metric(self) -> str:
        return DATASETS[self.dataset].metric

    @property
    def normalize(self) -> bool:
        return DATASETS[self.dataset].normalize


class FaissBaseline:
    name: str = ""
    index_kind: str = ""
    backend: str = "faiss"

    def __init__(self, cfg: FaissConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        train: np.ndarray,
        queries: np.ndarray,
        neighbors: np.ndarray,
        exp_dir: Path,
        *,
        data_root: Path,
    ) -> dict[str, float | int | str | None]:
        try:
            import faiss
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("faiss-cpu is required for FAISS baselines") from exc

        staging_dir = ensure_dir(exp_dir / "staging")
        config_path = exp_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "method": self.name,
                    "index_kind": self.index_kind,
                    "backend": self.backend,
                    **asdict(self.cfg),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        train_vecs = _normalize_if_cosine(train, self.cfg.metric)
        query_vecs = _normalize_if_cosine(queries, self.cfg.metric)
        if self.index_kind == "ivfpq":
            train_vecs = _pad_vectors(train_vecs, self.cfg.pq_m)
            query_vecs = _pad_vectors(query_vecs, self.cfg.pq_m)

        train_vecs = train_vecs.astype(np.float32, copy=False)
        query_vecs = query_vecs.astype(np.float32, copy=False)
        n, d = train_vecs.shape

        banner("method", f"{self.name} | dataset={self.cfg.dataset} | index={self.index_kind}")
        info(f"train={n}, queries={query_vecs.shape[0]}, dim={d}, metric={self.cfg.metric}")

        t0 = time.perf_counter()
        if self.index_kind == "ivf":
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, self.cfg.partitions, faiss.METRIC_L2)
            index.cp.seed = int(self.cfg.index_seed)
            index.cp.niter = int(self.cfg.ivf_niter)
            if n > self.cfg.train_samples:
                rng = np.random.default_rng(self.cfg.index_seed)
                sample_idx = rng.choice(n, size=self.cfg.train_samples, replace=False)
                sample = train_vecs[sample_idx]
            else:
                sample = train_vecs
            index.train(sample)
            build_train_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            index.add(train_vecs)
            build_add_s = time.perf_counter() - t1
            index.nprobe = int(self.cfg.nprobe or 10)
        elif self.index_kind == "ivfpq":
            index = faiss.IndexIVFPQ(
                faiss.IndexFlatL2(d),
                d,
                self.cfg.partitions,
                self.cfg.pq_m,
                self.cfg.pq_bits,
            )
            index.cp.seed = int(self.cfg.index_seed)
            index.cp.niter = int(self.cfg.ivf_niter)
            index.pq.cp.seed = int(self.cfg.index_seed)
            if n > self.cfg.train_samples:
                rng = np.random.default_rng(self.cfg.index_seed)
                sample_idx = rng.choice(n, size=self.cfg.train_samples, replace=False)
                sample = train_vecs[sample_idx]
            else:
                sample = train_vecs
            index.train(sample)
            build_train_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            index.add(train_vecs)
            build_add_s = time.perf_counter() - t1
            index.nprobe = int(self.cfg.nprobe or 10)
        elif self.index_kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, int(self.cfg.hnsw_m), faiss.METRIC_L2)
            index.hnsw.efConstruction = int(self.cfg.hnsw_ef_construction)
            t1 = time.perf_counter()
            index.add(train_vecs)
            build_add_s = time.perf_counter() - t1
            build_train_s = 0.0
            index.hnsw.efSearch = int(self.cfg.hnsw_ef_search)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported index kind: {self.index_kind}")
        build_total_s = build_train_s + build_add_s

        index_path = staging_dir / "faiss.index"
        faiss.write_index(index, str(index_path))
        meta = {
            "method": self.name,
            "index_kind": self.index_kind,
            "build_train_s": build_train_s,
            "build_add_s": build_add_s,
            "build_total_s": build_total_s,
            "ntotal": int(index.ntotal),
            "dim": int(d),
        }
        write_json(staging_dir / "faiss_meta.json", meta)

        list_sizes = None
        if self.index_kind in {"ivf", "ivfpq"}:
            _, list_ids = index.quantizer.search(train_vecs, 1)
            list_ids = list_ids.reshape(-1).astype(np.int32, copy=False)
            np.save(staging_dir / "ivf_list_ids_index.npy", list_ids)
            list_sizes = np.bincount(list_ids, minlength=self.cfg.partitions)

        return_k = max(self.cfg.k, int(self.cfg.return_candidates_k or self.cfg.k))
        mem_before = _rss_mb()
        prep_t0 = time.perf_counter()
        query_batch = query_vecs.astype(np.float32, copy=False)
        prep_s = time.perf_counter() - prep_t0

        if self.index_kind == "hnsw":
            try:
                faiss.cvar.hnsw_stats.reset()
            except Exception:
                pass

        search_t0 = time.perf_counter()
        _, retrieved = index.search(query_batch, return_k)
        search_s = time.perf_counter() - search_t0
        total_s = prep_s + search_s
        mem_after = _rss_mb()

        hnsw_ndis_total = None
        if self.index_kind == "hnsw":
            try:
                hnsw_ndis_total = float(faiss.cvar.hnsw_stats.ndis)
            except Exception:
                hnsw_ndis_total = None

        metric_key = self.cfg.metric
        if self.index_kind == "ivfpq" and return_k > self.cfg.k:
            train_rerank = train
            query_rerank = queries
            if metric_key == "cosine" and self.cfg.normalize:
                train_rerank = _normalize_if_cosine(train, metric_key)
                query_rerank = _normalize_if_cosine(queries, metric_key)
            train_sq_norms = (
                np.sum(train_rerank * train_rerank, axis=1).astype(np.float32, copy=False)
                if metric_key == "l2"
                else None
            )
            reranked = np.full((retrieved.shape[0], self.cfg.k), -1, dtype=np.int32)
            for idx in range(retrieved.shape[0]):
                ids = retrieved[idx]
                ids = ids[ids >= 0]
                if ids.size == 0:
                    continue
                scores = _exact_scores_for_ids(
                    train_view=train_rerank,
                    train_sq_norms=train_sq_norms,
                    query_vec=query_rerank[idx],
                    ids=ids,
                    metric=metric_key,
                )
                top_idx = np.argpartition(-scores, kth=self.cfg.k - 1)[: self.cfg.k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                reranked[idx] = ids[top_idx]
            eval_ids = reranked
        else:
            eval_ids = retrieved[:, : self.cfg.k]

        recall_at_10 = recall_at_k(eval_ids, neighbors, self.cfg.k)

        avg_computations = float(self.cfg.k)
        computation_min = float(self.cfg.k)
        computation_max = float(self.cfg.k)
        avg_computations_mode = "topk_fallback"
        if self.index_kind == "hnsw" and hnsw_ndis_total is not None and eval_ids.shape[0] > 0:
            cand_stats = hnsw_candidate_stats(hnsw_ndis_total, eval_ids.shape[0])
            avg_computations = cand_stats.mean
            computation_min = cand_stats.minimum
            computation_max = cand_stats.maximum
            avg_computations_mode = cand_stats.mode
        elif list_sizes is not None:
            _, probe_lists = index.quantizer.search(query_batch, int(self.cfg.nprobe or 10))
            cand_stats = ivf_candidate_stats(list_sizes, probe_lists)
            avg_computations = cand_stats.mean
            computation_min = cand_stats.minimum
            computation_max = cand_stats.maximum
            avg_computations_mode = cand_stats.mode

        manifest_path = write_index_manifest(exp_dir)
        index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        query_mem_delta_mb = max(0.0, float(mem_after) - float(mem_before))
        query_mem_delta_mode = QUERY_MEM_MODE_IN_PROCESS
        isolated_mem = _isolated_query_memory(exp_dir=exp_dir, dataset=self.cfg.dataset, data_root=data_root)
        if isolated_mem is not None and "query_mem_delta_mb_isolated" in isolated_mem:
            query_mem_delta_mb = float(isolated_mem["query_mem_delta_mb_isolated"])
            query_mem_delta_mode = QUERY_MEM_MODE_ISOLATED

        metrics = {
            "dataset": self.cfg.dataset,
            "method": self.name,
            "seed": int(self.cfg.index_seed),
            "index_kind": self.index_kind,
            "recall_at_10": recall_at_10,
            "avg_computations": avg_computations,
            "avg_computations_mode": avg_computations_mode,
            "avg_candidates": avg_computations,
            "avg_candidates_mode": avg_computations_mode,
            "computation_min": computation_min,
            "computation_max": computation_max,
            "qps": float(eval_ids.shape[0] / total_s) if total_s > 0 else None,
            "index_overhead_mb": float(index_manifest["index_overhead_mb"]),
            "index_overhead_mode": INDEX_OVERHEAD_MODE,
            "serving_footprint_mb": float(index_manifest["serving_footprint_mb"]),
            "serving_footprint_mode": SERVING_FOOTPRINT_MODE,
            "index_size_mb": float(index_manifest["index_overhead_mb"]),
            "index_size_mode": INDEX_OVERHEAD_MODE,
            "query_mem_delta_mb": query_mem_delta_mb,
            "query_mem_delta_mode": query_mem_delta_mode,
            "build_total_s": build_total_s,
            "faiss_query_prep_s": prep_s,
            "faiss_search_s": search_s,
            "faiss_total_query_s": total_s,
            "queries_evaluated": int(eval_ids.shape[0]),
            "k": int(self.cfg.k),
            "nprobe": int(self.cfg.nprobe) if self.cfg.nprobe is not None else None,
            "hnsw_ef_search": int(self.cfg.hnsw_ef_search) if self.index_kind == "hnsw" else None,
            "hnsw_ndis_total": hnsw_ndis_total,
            "experiment_dir": str(exp_dir),
        }
        if isolated_mem is not None:
            metrics.update(isolated_mem)
        write_json(exp_dir / "metrics.json", metrics)

        print_table(
            "Query summary",
            ["metric", "value"],
            [
                ["Recall@10", f"{metrics['recall_at_10']:.4f}"],
                ["Avg computations", f"{metrics['avg_computations']:.1f}"],
                ["QPS", f"{metrics['qps']:.1f}" if metrics["qps"] is not None else "nan"],
                ["Index overhead (MB)", f"{metrics['index_overhead_mb']:.2f}"],
                ["Serving footprint (MB)", f"{metrics['serving_footprint_mb']:.2f}"],
                ["Query RSS delta (MB)", f"{metrics['query_mem_delta_mb']:.2f}"],
            ],
        )
        info(
            "modes: "
            f"computations={metrics['avg_computations_mode']}, "
            f"index={metrics['index_size_mode']}, "
            f"footprint={metrics['serving_footprint_mode']}, "
            f"query_mem={metrics['query_mem_delta_mode']}"
        )
        info(f"artifacts -> {exp_dir}")
        return metrics


class Ivf(FaissBaseline):
    name = "IVF"
    index_kind = "ivf"


class IvfPQ(FaissBaseline):
    name = "IVFPQ"
    index_kind = "ivfpq"


class Hnsw(FaissBaseline):
    name = "HNSW"
    index_kind = "hnsw"


def make_method(method: str, dataset: str, *, seed: int = 0) -> FaissBaseline:
    method_key = method.strip().upper()
    cfg = FaissConfig(dataset=dataset, index_seed=seed)
    if method_key == "IVF":
        return Ivf(cfg)
    if method_key == "IVFPQ":
        return IvfPQ(cfg)
    if method_key == "HNSW":
        return Hnsw(cfg)
    raise ValueError(f"Unsupported FAISS method: {method}")

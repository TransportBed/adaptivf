from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore[assignment]

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.compat.v1.set_random_seed(seed)
    except Exception:
        pass


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def maybe_set_threads(num_threads: int) -> None:
    if num_threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    if faiss is not None:
        try:
            faiss.omp_set_num_threads(int(num_threads))
        except Exception:
            pass
    if torch is not None:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass


def load_or_compute_flat_knn(
    *,
    data: np.ndarray,
    query: np.ndarray,
    k: int,
    cache_path: Path,
    metric: str,
) -> np.ndarray:
    if cache_path.exists():
        try:
            cached = np.load(cache_path).astype(np.int32, copy=False)
            if cached.ndim == 2 and cached.shape[0] == query.shape[0] and cached.shape[1] >= int(k):
                return cached[:, : int(k)].astype(np.int32, copy=False)
        except Exception:
            pass
    if faiss is None:
        raise RuntimeError("faiss-cpu is required for flat KNN computation.")

    metric_norm = str(metric).lower()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if metric_norm in {"inner_product", "ip", "cosine"}:
        index = faiss.IndexFlatIP(data.shape[1])
    else:
        index = faiss.IndexFlatL2(data.shape[1])
    index.add(data.astype(np.float32, copy=False))
    _, knn = index.search(query.astype(np.float32, copy=False), int(k))
    knn = knn.astype(np.int32, copy=False)
    np.save(cache_path, knn)
    return knn


def distance_to_centroids(vecs: np.ndarray, centroids: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    n = vecs.shape[0]
    c_norm = np.sum(centroids * centroids, axis=1, keepdims=True).T
    out = np.empty((n, centroids.shape[0]), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = vecs[start:end]
        x_norm = np.sum(x * x, axis=1, keepdims=True)
        d2 = x_norm + c_norm - 2.0 * (x @ centroids.T)
        np.maximum(d2, 0.0, out=d2)
        out[start:end] = np.sqrt(d2, dtype=np.float32)
    return out


def scaled_centroid_distances_with_scaler(
    train_vecs: np.ndarray,
    query_vecs: np.ndarray,
    infer_vecs: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dist = distance_to_centroids(train_vecs, centroids)
    query_dist = distance_to_centroids(query_vecs, centroids)
    infer_dist = distance_to_centroids(infer_vecs, centroids)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_dist).astype(np.float32, copy=False)
    query_scaled = scaler.transform(query_dist).astype(np.float32, copy=False)
    infer_scaled = scaler.transform(infer_dist).astype(np.float32, copy=False)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    return train_scaled, query_scaled, infer_scaled, mean, scale

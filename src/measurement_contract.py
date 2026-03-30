from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SERVING_FOOTPRINT_MODE = "serving_footprint_manifest"
INDEX_OVERHEAD_MODE = "index_overhead_manifest"
QUERY_MEM_MODE_IN_PROCESS = "in_process_rss_delta"
QUERY_MEM_MODE_ISOLATED = "isolated_subprocess_rss_delta"
AVG_COMPUTATIONS_MODE_IVF = "avg_unique_candidate_vectors_per_query"
AVG_COMPUTATIONS_MODE_HNSW = "avg_distance_computations_per_query"
AVG_COMPUTATIONS_MODE_ROUTER = "avg_unique_candidate_vectors_per_query"


@dataclass(frozen=True)
class CandidateStats:
    mean: float
    minimum: float
    maximum: float
    mode: str


def recall_at_k(retrieved: np.ndarray, neighbors: np.ndarray, k: int) -> float:
    if retrieved.ndim != 2 or neighbors.ndim != 2:
        raise ValueError("retrieved and neighbors must be rank-2 arrays")
    if retrieved.shape[0] != neighbors.shape[0]:
        raise ValueError("retrieved and neighbors must have the same number of queries")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    recalls = np.zeros(retrieved.shape[0], dtype=np.float64)
    for idx in range(retrieved.shape[0]):
        hits = len(np.intersect1d(retrieved[idx, :k], neighbors[idx, :k]))
        recalls[idx] = hits / float(k)
    return float(np.mean(recalls))


def ivf_candidate_stats(list_sizes: np.ndarray, probe_lists: np.ndarray) -> CandidateStats:
    counts = np.sum(list_sizes[probe_lists], axis=1).astype(np.float64, copy=False)
    return CandidateStats(
        mean=float(np.mean(counts)),
        minimum=float(np.min(counts)),
        maximum=float(np.max(counts)),
        mode=AVG_COMPUTATIONS_MODE_IVF,
    )


def hnsw_candidate_stats(ndis_total: float, total_queries: int) -> CandidateStats:
    if total_queries <= 0:
        raise ValueError("total_queries must be positive")
    mean = float(ndis_total) / float(total_queries)
    return CandidateStats(
        mean=mean,
        minimum=mean,
        maximum=mean,
        mode=AVG_COMPUTATIONS_MODE_HNSW,
    )

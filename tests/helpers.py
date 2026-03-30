from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets import LearnedDataset
from methods.router_family import (
    FitState,
    RouterConfig,
    RouterWeights,
    _build_inverted,
    _build_pq_artifacts,
    _write_training_artifacts,
)


def tiny_l2_bundle() -> LearnedDataset:
    index_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [10.0, 10.0],
            [10.0, 10.2],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.05, 0.0],
            [10.0, 10.05],
        ],
        dtype=np.float32,
    )
    eval_neighbors = np.array([[0], [2]], dtype=np.int32)
    router_neighbors = np.array(
        [
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
        ],
        dtype=np.int32,
    )
    return LearnedDataset(
        dataset="sift",
        router_train=index_train.copy(),
        router_neighbors=router_neighbors,
        index_train=index_train,
        queries=queries,
        eval_neighbors=eval_neighbors,
        sample_ids=None,
    )


def routing_weights() -> RouterWeights:
    return RouterWeights(
        W1=np.eye(2, dtype=np.float32),
        b1=np.zeros(2, dtype=np.float32),
        W2=np.array([[3.0, -3.0], [-3.0, 3.0]], dtype=np.float32),
        b2=np.zeros(2, dtype=np.float32),
    )


def tiny_fit_state(exp_dir: Path, *, pq_enabled: bool) -> FitState:
    cfg = RouterConfig(
        dataset="sift",
        method="AdaptIVF+PQ" if pq_enabled else "AdaptIVF",
        init_mode="ivf",
        repetitions=1,
        partitions=2,
        k=1,
        hidden=2,
        query_batch_size=8,
        min_reps=1,
        pq_enabled=pq_enabled,
        pq_m=2,
        pq_bits=1,
        return_candidates_k=2,
        seed=0,
    )
    bundle = tiny_l2_bundle()
    assign = np.array([0, 0, 1, 1], dtype=np.int32)
    lookups = [_build_inverted(assign, cfg.index_partitions)]
    point_to_buckets = assign[:, None]
    pq_codebooks = None
    pq_codes = None
    ivf_centroids = None
    ivf_list_ids = None
    if pq_enabled:
        ivf_centroids = np.array([[0.0, 0.1], [10.0, 10.1]], dtype=np.float32)
        ivf_list_ids = assign.copy()
        pq_codebooks, pq_codes = _build_pq_artifacts(
            bundle.index_train,
            centroids=ivf_centroids,
            list_ids=ivf_list_ids,
            cfg=cfg,
        )
    _write_training_artifacts(
        exp_dir=exp_dir,
        cfg=cfg,
        weights=[routing_weights()],
        lookups=lookups,
        point_to_buckets=point_to_buckets,
        train_assignments=assign[None, :],
        train_seconds=0.1,
        index_seconds=0.2,
        ivf_centroids=ivf_centroids,
        ivf_list_ids=ivf_list_ids,
        pq_codebooks=pq_codebooks,
        pq_codes=pq_codes,
    )
    return FitState(
        exp_dir=exp_dir,
        weights=[routing_weights()],
        lookups=lookups,
        point_to_buckets=point_to_buckets,
        train_seconds=0.1,
        index_seconds=0.2,
        ivf_centroids=ivf_centroids,
        ivf_perms=None,
        ivf_inv_perms=None,
        ivf_list_ids=ivf_list_ids,
        pq_codebooks=pq_codebooks,
        pq_codes=pq_codes,
    )

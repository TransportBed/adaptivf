from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptivf import AdaptIVFConfig, make_adaptivf
from datasets import LearnedDataset
from index_manifest import write_index_manifest
from measurement_contract import (
    AVG_COMPUTATIONS_MODE_HNSW,
    AVG_COMPUTATIONS_MODE_IVF,
    hnsw_candidate_stats,
    ivf_candidate_stats,
)
import methods.lira_runtime as lira_runtime
import plots
import studies
import tables
from methods.router_family import (
    FitState,
    RouterConfig,
    RouterFamily,
    RouterWeights,
    _build_inverted,
    _collect_candidates,
    _exact_scores_for_ids,
    _full_index_lookups,
    _query_buffer,
    make_router_method,
)
from tests.helpers import routing_weights


def test_make_adaptivf_applies_user_controls() -> None:
    index = make_adaptivf(
        "glove10k",
        config=AdaptIVFConfig(
            seed=7,
            confidence_threshold=0.61,
            max_assignments=4,
            m_base=3,
            m_max=33,
            entropy_scale=4.5,
            reassign_interval=9,
            pq_enabled=True,
            pq_m=8,
            pq_bits=4,
        ),
    )
    assert isinstance(index, RouterFamily)
    assert index.cfg.method == "AdaptIVF+PQ"
    assert index.cfg.assignment_threshold == pytest.approx(0.61)
    assert index.cfg.max_assignments == 4
    assert index.cfg.m_base == 3
    assert index.cfg.m_max == 33
    assert index.cfg.entropy_scale == pytest.approx(4.5)
    assert index.cfg.reassign_interval == 9
    assert index.cfg.pq_enabled is True
    assert index.cfg.pq_m == 8
    assert index.cfg.pq_bits == 4
    assert index.cfg.seed == 7


def test_studies_package_exports_ablation_module() -> None:
    assert "ablation" in studies.__all__


@pytest.mark.parametrize(
    ("method_name", "expected"),
    [
        ("AdaptIVF", {"method": "AdaptIVF", "pq_enabled": False, "reassign_interval": 5, "max_assignments": 3}),
        ("AdaptIVF-Static", {"method": "AdaptIVF-Static", "pq_enabled": False, "reassign_interval": 0, "max_assignments": 3}),
        ("AdaptIVF-A4", {"method": "AdaptIVF-A4", "pq_enabled": False, "reassign_interval": 5, "max_assignments": 4}),
        ("AdaptIVF+PQ", {"method": "AdaptIVF+PQ", "pq_enabled": True, "reassign_interval": 5, "max_assignments": 3}),
    ],
)
def test_make_router_method_variants(method_name: str, expected: dict[str, object]) -> None:
    method = make_router_method(method_name, "glove10k", seed=11)
    assert method.cfg.method == expected["method"]
    assert method.cfg.repetitions == 1
    assert method.cfg.min_reps == 1
    assert method.cfg.ivf_permute_reps is False
    assert method.cfg.assignment_strategy == "confidence_threshold"
    assert method.cfg.probing_strategy == "entropy_adaptive"
    assert method.cfg.assignment_threshold == pytest.approx(0.75)
    assert method.cfg.max_assignments == expected["max_assignments"]
    assert method.cfg.pq_enabled is expected["pq_enabled"]
    assert method.cfg.reassign_interval == expected["reassign_interval"]
    assert method.cfg.seed == 11


def test_full_index_lookups_duplicates_only_low_confidence_points(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = RouterConfig(
        dataset="glove10k",
        method="AdaptIVF",
        init_mode="ivf",
        repetitions=1,
        partitions=4,
        assignment_strategy="confidence_threshold",
        assignment_threshold=0.75,
        max_assignments=3,
        reassign_batch_size=16,
        seed=0,
    )
    index_train = np.zeros((3, 2), dtype=np.float32)
    dummy_weights = [
        RouterWeights(
            W1=np.zeros((2, 1), dtype=np.float32),
            b1=np.zeros(1, dtype=np.float32),
            W2=np.zeros((1, 4), dtype=np.float32),
            b2=np.zeros(4, dtype=np.float32),
        )
    ]

    def fake_forward(_x: np.ndarray, _weights: RouterWeights) -> np.ndarray:
        return np.array(
            [
                [10.0, -10.0, -10.0, -10.0],
                [0.1, 0.0, -0.1, -10.0],
                [1.8, 1.7, 1.9, -10.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr("methods.router_family._router_forward", fake_forward)
    lookups, point_to_buckets = _full_index_lookups(index_train, dummy_weights, cfg)

    offsets, ids = lookups[0]
    bucket_members = {}
    for bucket in range(cfg.index_partitions):
        bucket_members[bucket] = ids[offsets[bucket] : offsets[bucket + 1]].tolist()

    assert point_to_buckets[:, 0].tolist() == [0, 0, 2]
    assert bucket_members[0] == [0, 1, 2]
    assert sorted(bucket_members[1]) == [1, 2]
    assert sorted(bucket_members[2]) == [1, 2]
    assert bucket_members[3] == []
    assert int(offsets[-1]) == 7


def test_evaluate_uses_entropy_adaptive_probe_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = RouterConfig(
        dataset="sift",
        method="AdaptIVF",
        init_mode="ivf",
        repetitions=1,
        partitions=4,
        hidden=1,
        k=1,
        min_reps=1,
        query_batch_size=8,
        probing_strategy="entropy_adaptive",
        m_base=1,
        m_max=4,
        entropy_scale=6.0,
        seed=0,
    )
    router = RouterFamily(cfg)
    bundle = LearnedDataset(
        dataset="sift",
        router_train=np.zeros((4, 2), dtype=np.float32),
        router_neighbors=np.zeros((4, 1), dtype=np.int32),
        index_train=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        queries=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        eval_neighbors=np.array([[0], [0]], dtype=np.int32),
        sample_ids=None,
    )
    state = FitState(
        exp_dir=tmp_path,
        weights=[
            RouterWeights(
                W1=np.zeros((2, 1), dtype=np.float32),
                b1=np.zeros(1, dtype=np.float32),
                W2=np.zeros((1, 4), dtype=np.float32),
                b2=np.zeros(4, dtype=np.float32),
            )
        ],
        lookups=[_build_inverted(np.array([0, 1, 2, 3], dtype=np.int32), 4)],
        point_to_buckets=np.array([[0], [1], [2], [3]], dtype=np.int32),
        train_seconds=0.0,
        index_seconds=0.0,
    )
    probe_widths: list[int] = []

    def fake_forward(batch: np.ndarray, _weights: RouterWeights) -> np.ndarray:
        assert batch.shape[0] == 2
        return np.array(
            [
                [12.0, -12.0, -12.0, -12.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    def fake_collect(topm: np.ndarray, **_kwargs: object) -> np.ndarray:
        probe_widths.append(int(topm.shape[1]))
        return np.arange(topm.shape[1], dtype=np.int32)

    def fake_scores(**kwargs: object) -> np.ndarray:
        ids = np.asarray(kwargs["ids"])
        return -ids.astype(np.float32)

    monkeypatch.setattr("methods.router_family._router_forward", fake_forward)
    monkeypatch.setattr("methods.router_family._collect_candidates", fake_collect)
    monkeypatch.setattr("methods.router_family._exact_scores_for_ids", fake_scores)

    result = router.evaluate(bundle, state, probe_depth=2)
    assert probe_widths == [1, 4]
    assert result.avg_computations == pytest.approx(2.5)
    assert result.recall_at_10 == pytest.approx(1.0)


def test_query_buffer_matches_full_and_nonfull_dataset_modes() -> None:
    raw_full, buffer_full = _query_buffer(
        index_full_dataset=True,
        n_points=10_000,
        repetitions=4,
        probe_depth=10,
        partitions=100,
        min_reps=2,
        k=10,
    )
    raw_subset, buffer_subset = _query_buffer(
        index_full_dataset=False,
        n_points=10_000,
        repetitions=4,
        probe_depth=10,
        partitions=100,
        min_reps=2,
        k=10,
    )

    assert raw_full == raw_subset == 4000
    assert buffer_full == 3072
    assert buffer_subset == 8000


def test_collect_candidates_ignores_out_of_range_bucket_ids() -> None:
    lookups = [_build_inverted(np.array([0, 1, 0], dtype=np.int32), 2)]
    candidates = _collect_candidates(
        np.array([[0, 5, -1]], dtype=np.int32),
        lookups=lookups,
        min_reps=1,
        buffer=16,
    )
    assert candidates.tolist() == [0, 2]


def test_exact_scores_for_cosine_use_normalized_inner_product() -> None:
    train_view = np.array(
        [
            [0.6, 0.8],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    query_vec = np.array([0.6, 0.8], dtype=np.float32)
    scores = _exact_scores_for_ids(
        train_view=train_view,
        train_sq_norms=None,
        query_vec=query_vec,
        ids=np.array([0, 1], dtype=np.int32),
        metric="cosine",
    )
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.6)


def test_lira_replay_reconstructs_selected_intermediate_part(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_data_to_buckets = np.array([[0, -1], [1, -1]], dtype=np.int32)
    initial_cluster_ids = [[0], [1], [], []]
    data_scores = np.zeros((2, 4), dtype=np.float32)
    data_predicts = np.array([[True, True, False, False], [True, False, False, False]])
    calls: list[tuple[int, int]] = []

    def fake_apply(
        _data_scores: np.ndarray,
        _data_predicts: np.ndarray,
        _sorted_ids: np.ndarray,
        data_to_buckets: np.ndarray,
        cluster_ids: list[list[int]],
        *,
        n_mul: int,
        begin: int,
        end: int,
    ) -> None:
        calls.append((begin, end))
        bucket = 2 + len(calls) - 1
        data_to_buckets[0, 1] = bucket
        cluster_ids[bucket].append(0)

    monkeypatch.setattr(lira_runtime, "_apply_redundancy_by_model", fake_apply)

    cfg = lira_runtime.LiraConfig(dataset="glove10k", n_bkt=4, repa_step=3)
    replayed_buckets, replayed_clusters = lira_runtime._replay_repartition_state(
        initial_data_to_buckets,
        initial_cluster_ids,
        data_scores,
        data_predicts,
        cfg,
        selected_part=2,
    )

    assert calls == [(0, 1), (1, 2)]
    assert replayed_buckets[0, 1] == 3
    assert replayed_clusters[3] == [0]


def test_nvidia_smi_parser_accepts_rows_with_units(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = "0, 143 MiB\n1, 12 MiB\n"

    monkeypatch.setattr(lira_runtime.subprocess, "check_output", lambda *args, **kwargs: raw)

    idx, detail = lira_runtime._pick_idle_cuda_index_via_nvidia_smi()
    assert idx == 1
    assert "12 MB" in detail


def test_index_manifest_separates_shared_vectors_from_index_overhead(tmp_path: Path) -> None:
    exp_dir = tmp_path / "manifest_case"
    (exp_dir / "models").mkdir(parents=True)
    (exp_dir / "staging" / "lookups").mkdir(parents=True)
    np.save(exp_dir / "staging" / "train.npy", np.zeros((4, 2), dtype=np.float32))
    np.save(exp_dir / "staging" / "index.npy", np.zeros((4, 2), dtype=np.float32))
    np.save(exp_dir / "staging" / "point_to_buckets.npy", np.zeros((4, 1), dtype=np.int32))
    np.save(exp_dir / "staging" / "lookups" / "rep0_offsets.npy", np.array([0, 2, 4], dtype=np.int64))
    np.save(exp_dir / "staging" / "lookups" / "rep0_ids.npy", np.array([0, 1, 2, 3], dtype=np.int32))
    np.savez(
        exp_dir / "models" / "router_weights.npz",
        W1=np.zeros((1, 2, 1), dtype=np.float32),
        b1=np.zeros((1, 1), dtype=np.float32),
        W2=np.zeros((1, 1, 2), dtype=np.float32),
        b2=np.zeros((1, 2), dtype=np.float32),
    )

    manifest_path = write_index_manifest(exp_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["shared_vector_payload_mb"] > 0.0
    assert manifest["serving_footprint_mb"] > manifest["index_overhead_mb"]
    assert manifest["index_overhead_mb"] > 0.0
    assert any(row["component"] == "serving_staging" for row in manifest["index_overhead_components"])


def test_index_manifest_classifies_serving_and_nonserving_components(tmp_path: Path) -> None:
    exp_dir = tmp_path / "manifest_components"
    (exp_dir / "models").mkdir(parents=True)
    (exp_dir / "staging" / "lookups").mkdir(parents=True)
    (exp_dir / "staging" / "lira_inner_indexes").mkdir(parents=True)
    (exp_dir / "data").mkdir(parents=True)

    np.save(exp_dir / "staging" / "train.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(exp_dir / "staging" / "index.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(exp_dir / "staging" / "point_to_buckets.npy", np.zeros((2, 1), dtype=np.int32))
    np.save(exp_dir / "staging" / "lookups" / "rep0_offsets.npy", np.array([0, 1, 2], dtype=np.int64))
    np.save(exp_dir / "staging" / "lookups" / "rep0_ids.npy", np.array([0, 1], dtype=np.int32))
    (exp_dir / "staging" / "lira_inner_indexes" / "bucket0.bin").write_bytes(b"inner-index")
    (exp_dir / "data" / "lira_threshold_part0.csv").write_text("threshold,nprobe\n0.1,4\n", encoding="utf-8")
    np.savez(
        exp_dir / "models" / "router_weights.npz",
        W1=np.zeros((1, 2, 1), dtype=np.float32),
        b1=np.zeros((1, 1), dtype=np.float32),
        W2=np.zeros((1, 1, 2), dtype=np.float32),
        b2=np.zeros((1, 2), dtype=np.float32),
    )

    manifest = json.loads(write_index_manifest(exp_dir).read_text(encoding="utf-8"))

    components = {row["component"] for row in manifest["components"]}
    serving_components = {row["component"] for row in manifest["serving_components"]}
    overhead_components = {row["component"] for row in manifest["index_overhead_components"]}

    assert "lira_data" in components
    assert "lira_data" not in serving_components
    assert "lira_data" not in overhead_components
    assert "lira_inner_indexes" in serving_components
    assert "models" in serving_components
    assert "lookups" in serving_components
    assert manifest["shared_vector_payload_mb"] > 0.0
    assert manifest["serving_footprint_mb"] > manifest["index_overhead_mb"]


def test_export_aggregates_preserve_operating_point_fields() -> None:
    df = pd.DataFrame(
        [
            {
                "dataset": "glove10k",
                "method": "LIRA",
                "seed": 0,
                "probe_depth": 10,
                "recall_at_10": 0.90,
                "avg_computations": 100.0,
                "qps": 50.0,
                "index_size_mb": 7.0,
                "index_overhead_mb": 7.0,
                "serving_footprint_mb": 19.0,
                "query_mem_delta_mb": 4.0,
                "train_s": 12.0,
                "threshold": 0.25,
                "nprobe": 9.5,
                "selected_part": 1,
            },
            {
                "dataset": "glove10k",
                "method": "LIRA",
                "seed": 1,
                "probe_depth": 10,
                "recall_at_10": 0.92,
                "avg_computations": 110.0,
                "qps": 55.0,
                "index_size_mb": 8.0,
                "index_overhead_mb": 8.0,
                "serving_footprint_mb": 20.0,
                "query_mem_delta_mb": 5.0,
                "train_s": 13.0,
                "threshold": 0.35,
                "nprobe": 10.5,
                "selected_part": 2,
            },
        ]
    )

    table_df = tables._aggregate(df, ["dataset", "method"])
    plot_df = plots._aggregate(df, ["dataset", "method"])

    assert table_df.loc[0, "probe_depth"] == pytest.approx(10.0)
    assert table_df.loc[0, "threshold"] == pytest.approx(0.30)
    assert table_df.loc[0, "nprobe"] == pytest.approx(10.0)
    assert table_df.loc[0, "selected_part"] == pytest.approx(1.5)
    assert plot_df.loc[0, "threshold"] == pytest.approx(0.30)
    assert plot_df.loc[0, "nprobe"] == pytest.approx(10.0)
    assert plot_df.loc[0, "selected_part"] == pytest.approx(1.5)


def test_candidate_stats_modes_are_explicit() -> None:
    list_sizes = np.array([3, 5, 7], dtype=np.int64)
    probe_lists = np.array([[0, 1], [1, 2]], dtype=np.int64)
    ivf_stats = ivf_candidate_stats(list_sizes, probe_lists)
    assert ivf_stats.mean == pytest.approx(10.0)
    assert ivf_stats.minimum == pytest.approx(8.0)
    assert ivf_stats.maximum == pytest.approx(12.0)
    assert ivf_stats.mode == AVG_COMPUTATIONS_MODE_IVF

    hnsw_stats = hnsw_candidate_stats(42.0, 6)
    assert hnsw_stats.mean == pytest.approx(7.0)
    assert hnsw_stats.minimum == pytest.approx(7.0)
    assert hnsw_stats.maximum == pytest.approx(7.0)
    assert hnsw_stats.mode == AVG_COMPUTATIONS_MODE_HNSW

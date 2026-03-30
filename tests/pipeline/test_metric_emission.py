from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from measurement_contract import (
    AVG_COMPUTATIONS_MODE_HNSW,
    AVG_COMPUTATIONS_MODE_IVF,
    INDEX_OVERHEAD_MODE,
    QUERY_MEM_MODE_ISOLATED,
    SERVING_FOOTPRINT_MODE,
)
from methods.faiss_baselines import FaissConfig, Hnsw, Ivf
from methods.lira import Lira
from presets import DatasetSpec
from tests.helpers import tiny_l2_bundle


def test_faiss_ivf_metrics_emit_shared_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import methods.faiss_baselines as fb

    monkeypatch.setitem(
        fb.DATASETS,
        "sift",
        DatasetSpec(
            key="sift",
            ann_bench_name="sift-128-euclidean",
            metric="l2",
            dim=2,
            indexed_size=4,
            query_count=2,
            eval_queries=2,
            partitions=2,
            normalize=False,
        ),
    )
    monkeypatch.setattr(fb, "_isolated_query_memory", lambda **_kwargs: {"query_mem_delta_mb_isolated": 2.5})

    bundle = tiny_l2_bundle()
    method = Ivf(FaissConfig(dataset="sift", k=1, nprobe=1, train_samples=4, ivf_niter=2))
    metrics = method.run(bundle.index_train, bundle.queries, bundle.eval_neighbors, tmp_path / "ivf_case", data_root=tmp_path)

    assert metrics["avg_computations_mode"] == AVG_COMPUTATIONS_MODE_IVF
    assert metrics["index_size_mode"] == INDEX_OVERHEAD_MODE
    assert metrics["serving_footprint_mode"] == SERVING_FOOTPRINT_MODE
    assert metrics["query_mem_delta_mode"] == QUERY_MEM_MODE_ISOLATED
    assert metrics["query_mem_delta_mb"] == pytest.approx(2.5)
    assert metrics["index_overhead_mb"] >= 0.0
    assert metrics["serving_footprint_mb"] >= metrics["index_overhead_mb"]


def test_faiss_hnsw_metrics_emit_distance_computation_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import methods.faiss_baselines as fb

    monkeypatch.setitem(
        fb.DATASETS,
        "sift",
        DatasetSpec(
            key="sift",
            ann_bench_name="sift-128-euclidean",
            metric="l2",
            dim=2,
            indexed_size=4,
            query_count=2,
            eval_queries=2,
            partitions=2,
            normalize=False,
        ),
    )
    monkeypatch.setattr(fb, "_isolated_query_memory", lambda **_kwargs: {"query_mem_delta_mb_isolated": 1.75})

    bundle = tiny_l2_bundle()
    method = Hnsw(FaissConfig(dataset="sift", k=1, hnsw_m=8, hnsw_ef_search=16))
    metrics = method.run(bundle.index_train, bundle.queries, bundle.eval_neighbors, tmp_path / "hnsw_case", data_root=tmp_path)

    assert metrics["avg_computations_mode"] == AVG_COMPUTATIONS_MODE_HNSW
    assert metrics["index_size_mode"] == INDEX_OVERHEAD_MODE
    assert metrics["serving_footprint_mode"] == SERVING_FOOTPRINT_MODE
    assert metrics["query_mem_delta_mode"] == QUERY_MEM_MODE_ISOLATED
    assert metrics["query_mem_delta_mb"] == pytest.approx(1.75)


def test_lira_metrics_emit_shared_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import methods.lira as lira_mod

    def fake_run_lira_smallscale(_cfg, exp_dir: Path, *, data_root: Path, force_prepare: bool):
        assert data_root == tmp_path
        assert force_prepare is False
        return SimpleNamespace(
            selected_rows=[
                {"threshold": 0.40, "nprobe": 12.0, "recall10_at_10": 0.91, "avg_computations": 15.0},
                {"threshold": 0.25, "nprobe": 10.0, "recall10_at_10": 0.90, "avg_computations": 12.0},
            ],
            metadata={"timings_s": {"total": 123.0}},
            selected_part=3,
        )

    def fake_online_eval(_cfg, exp_dir: Path, *, data_root: Path, threshold: float):
        assert data_root == tmp_path
        assert threshold == pytest.approx(0.25)
        return {
            "recall10_at_10": 0.93,
            "avg_computations": 11.0,
            "avg_computations_mode": "avg_unique_candidate_vectors_per_query",
            "qps": 55.0,
            "query_mem_delta_mb": 4.0,
            "nprobe": 10.0,
        }

    def fake_manifest(exp_dir: Path) -> Path:
        path = exp_dir / "index_manifest.json"
        exp_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "index_overhead_mb": 7.5,
                    "serving_footprint_mb": 19.0,
                }
            ),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(lira_mod, "run_lira_smallscale", fake_run_lira_smallscale)
    monkeypatch.setattr(lira_mod, "run_lira_selected_online_eval", fake_online_eval)
    monkeypatch.setattr(lira_mod, "write_index_manifest", fake_manifest)

    row = Lira("glove10k", seed=5).run_competitiveness(data_root=tmp_path, exp_dir=tmp_path / "lira_case", force_prepare=False)

    assert row["method"] == "LIRA"
    assert row["recall_at_10"] == pytest.approx(0.93)
    assert row["avg_computations"] == pytest.approx(11.0)
    assert row["qps"] == pytest.approx(55.0)
    assert row["index_overhead_mb"] == pytest.approx(7.5)
    assert row["serving_footprint_mb"] == pytest.approx(19.0)
    assert row["index_size_mode"] == INDEX_OVERHEAD_MODE
    assert row["serving_footprint_mode"] == SERVING_FOOTPRINT_MODE
    assert row["query_mem_delta_mode"] == QUERY_MEM_MODE_ISOLATED
    assert row["query_mem_delta_mb"] == pytest.approx(4.0)
    assert row["threshold"] == pytest.approx(0.25)
    assert row["selected_part"] == 3
    summary_path = tmp_path / "lira_case" / "competitiveness_summary.json"
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["threshold"] == pytest.approx(0.25)
    assert summary_payload["nprobe"] == pytest.approx(10.0)
    assert summary_payload["selected_part"] == 3

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivf import AdaptIVFConfig, make_adaptivf
from measurement_contract import (
    AVG_COMPUTATIONS_MODE_ROUTER,
    INDEX_OVERHEAD_MODE,
    QUERY_MEM_MODE_ISOLATED,
)
from methods.router_family import QueryResult, RouterConfig, RouterFamily
from tests.helpers import tiny_fit_state, tiny_l2_bundle


def test_adaptivf_evaluate_pipeline_returns_perfect_recall(tmp_path: Path) -> None:
    bundle = tiny_l2_bundle()
    cfg = RouterConfig(
        dataset="sift",
        method="AdaptIVF",
        init_mode="ivf",
        repetitions=1,
        partitions=2,
        hidden=2,
        k=1,
        min_reps=1,
        query_batch_size=8,
        probe_depth=1,
        probing_strategy="fixed",
        pq_enabled=False,
        seed=0,
    )
    state = tiny_fit_state(tmp_path / "adaptivf", pq_enabled=False)
    result = RouterFamily(cfg).evaluate(bundle, state, probe_depth=1)
    assert result.recall_at_10 == pytest.approx(1.0)
    assert result.avg_computations == pytest.approx(1.0)
    assert result.qps is not None and result.qps > 0


def test_adaptivf_pq_evaluate_pipeline_returns_perfect_recall(tmp_path: Path) -> None:
    bundle = tiny_l2_bundle()
    cfg = RouterConfig(
        dataset="sift",
        method="AdaptIVF+PQ",
        init_mode="ivf",
        repetitions=1,
        partitions=2,
        hidden=2,
        k=1,
        min_reps=1,
        query_batch_size=8,
        probe_depth=1,
        probing_strategy="fixed",
        pq_enabled=True,
        pq_m=2,
        pq_bits=1,
        return_candidates_k=2,
        seed=0,
    )
    state = tiny_fit_state(tmp_path / "adaptivf_pq", pq_enabled=True)
    result = RouterFamily(cfg).evaluate(bundle, state, probe_depth=1)
    assert result.recall_at_10 == pytest.approx(1.0)
    assert result.avg_computations == pytest.approx(1.0)
    assert result.qps is not None and result.qps > 0


@pytest.mark.parametrize("pq_enabled", [False, True], ids=["adaptivf", "adaptivf_pq"])
def test_run_competitiveness_pipeline_emits_paper_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pq_enabled: bool,
) -> None:
    bundle = tiny_l2_bundle()
    config = AdaptIVFConfig(pq_enabled=pq_enabled)
    router = make_adaptivf("sift", config=config)
    exp_dir = tmp_path / ("adaptivf_pq" if pq_enabled else "adaptivf")
    state = tiny_fit_state(exp_dir, pq_enabled=pq_enabled)

    def fake_load(*_args: object, **_kwargs: object):
        return bundle

    def fake_fit(self: RouterFamily, loaded, run_dir: Path, *, data_root: Path):
        assert loaded is bundle
        assert run_dir == exp_dir
        return state

    def fake_isolated_query_memory(**_kwargs: object) -> dict[str, float]:
        return {"query_mem_delta_mb_isolated": 1.5}

    def fake_evaluate(
        self: RouterFamily,
        loaded,
        fit_state,
        *,
        probe_depth: int,
    ) -> QueryResult:
        assert loaded is bundle
        assert fit_state is state
        assert probe_depth == int(self.cfg.probe_depth)
        return QueryResult(
            probe_depth=probe_depth,
            recall_at_10=0.9,
            avg_computations=12.0,
            computation_min=5.0,
            computation_max=20.0,
            qps=34.0,
            query_mem_delta_mb=0.25,
        )

    monkeypatch.setattr("methods.router_family.load_learned_dataset", fake_load)
    monkeypatch.setattr(RouterFamily, "fit", fake_fit)
    monkeypatch.setattr(RouterFamily, "evaluate", fake_evaluate)
    monkeypatch.setattr("methods.router_family._isolated_query_memory", fake_isolated_query_memory)

    row = router.run_competitiveness(data_root=tmp_path, exp_dir=exp_dir, force_prepare=False)

    assert row["method"] == ("AdaptIVF+PQ" if pq_enabled else "AdaptIVF")
    assert row["recall_at_10"] == pytest.approx(0.9)
    assert row["avg_computations"] == pytest.approx(12.0)
    assert row["qps"] == pytest.approx(34.0)
    assert row["avg_computations_mode"] == AVG_COMPUTATIONS_MODE_ROUTER
    assert row["index_size_mode"] == INDEX_OVERHEAD_MODE
    assert row["index_size_mb"] == pytest.approx(row["index_overhead_mb"])
    assert row["query_mem_delta_mode"] == QUERY_MEM_MODE_ISOLATED
    assert row["query_mem_delta_mb"] == pytest.approx(1.5)
    assert Path(str(row["experiment_dir"])) == exp_dir

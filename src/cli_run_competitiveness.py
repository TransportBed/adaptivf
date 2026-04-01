from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts import ensure_dir, new_experiment_dir, write_csv, write_json
from console import banner, info
from datasets import parse_datasets
from methods.adaptivf import (
    AdaptIVF,
    AdaptIVFPQ,
    make_adaptivf_ablation,
)
from methods.bliss import Bliss
from methods.router_family import make_router_method
from methods.faiss_baselines import FaissConfig, Hnsw, Ivf, IvfPQ
from methods.lira import Lira
from methods.mlp_ivf import MlpIvf, MlpIvfPQ


_METHOD_ORDER = {
    "HNSW": 0,
    "IVF": 1,
    "MLP-IVF": 2,
    "BLISS": 3,
    "LIRA": 4,
    "AdaptIVF": 5,
    "IVFPQ": 6,
    "MLP-IVFPQ": 7,
    "AdaptIVF+PQ": 8,
    "AdaptIVF-Static": 9,
    "AdaptIVF-A4": 10,
    "AdaptIVF-Static+PQ": 11,
    "AdaptIVF-A4+PQ": 12,
    "AdaptIVF-m80": 13,
    "AdaptIVF+PQ-m80": 14,
}


def _make_method(method: str, dataset: str, *, seed: int, m_max: int | None = None):
    key = method.strip().upper()
    if key == "HNSW":
        return Hnsw(FaissConfig(dataset=dataset, index_seed=seed))
    if key == "IVF":
        return Ivf(FaissConfig(dataset=dataset, index_seed=seed))
    if key == "IVFPQ":
        return IvfPQ(FaissConfig(dataset=dataset, index_seed=seed))
    if key == "BLISS":
        return Bliss(dataset, seed=seed)
    if key == "MLP-IVF":
        return MlpIvf(dataset, seed=seed)
    if key in {"MLP-IVFPQ", "MLP_IVFPQ"}:
        return MlpIvfPQ(dataset, seed=seed)
    if key == "LIRA":
        return Lira(dataset, seed=seed)
    if key == "ADAPTIVF":
        return AdaptIVF(dataset, seed=seed, m_max=m_max)
    if key in {"ADAPTIVF-STATIC", "ADAPTIVF_STATIC"}:
        return make_adaptivf_ablation("AdaptIVF-Static", dataset, seed=seed, m_max=m_max)
    if key in {"ADAPTIVF-A4", "ADAPTIVF_A4"}:
        return make_adaptivf_ablation("AdaptIVF-A4", dataset, seed=seed, m_max=m_max)
    if key in {"ADAPTIVF+PQ", "ADAPTIVF_PQ"}:
        return AdaptIVFPQ(dataset, seed=seed, m_max=m_max)
    if key in {"ADAPTIVF-STATIC+PQ", "ADAPTIVF_STATIC+PQ", "ADAPTIVF_STATIC_PQ"}:
        return make_adaptivf_ablation("AdaptIVF-Static+PQ", dataset, seed=seed, m_max=m_max)
    if key in {"ADAPTIVF-A4+PQ", "ADAPTIVF_A4+PQ", "ADAPTIVF_A4_PQ"}:
        return make_adaptivf_ablation("AdaptIVF-A4+PQ", dataset, seed=seed, m_max=m_max)
    if key in {"ADAPTIVF-M80", "ADAPTIVF_M80"}:
        return make_router_method("AdaptIVF-m80", dataset, seed=seed)
    if key in {"ADAPTIVF+PQ-M80", "ADAPTIVF_PQ_M80", "ADAPTIVF+PQ_M80"}:
        return make_router_method("AdaptIVF+PQ-m80", dataset, seed=seed)
    raise ValueError(f"Unsupported competitiveness method: {method}")


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _norm_text(value: object) -> str:
    return str(value or "").strip()


def _norm_int(value: object) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def _merge_rows(existing: list[dict[str, object]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in existing + new_rows:
        key = (
            _norm_text(row.get("dataset")),
            _norm_text(row.get("method")),
            _norm_int(row.get("seed")),
        )
        normalized = dict(row)
        normalized["dataset"] = key[0]
        normalized["method"] = key[1]
        normalized["seed"] = key[2]
        merged[key] = normalized
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("dataset", "")),
            _METHOD_ORDER.get(str(row.get("method", "")), 999),
            int(row.get("seed", 0) or 0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone paper competitiveness methods.")
    parser.add_argument("--datasets", default="glove10k")
    parser.add_argument("--methods", default="HNSW,IVF,MLP-IVF,BLISS,LIRA,AdaptIVF,IVFPQ,MLP-IVFPQ,AdaptIVF+PQ")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--exports-root", default="paper_exports")
    parser.add_argument("--export-subdir", default="competitiveness")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--m-max", type=int, default=None, help="Override m_max for AdaptIVF family (default: 10)")
    args = parser.parse_args()

    datasets = parse_datasets([args.datasets])
    methods = [token.strip() for token in args.methods.split(",") if token.strip()]
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    data_root = Path(args.data_root).expanduser().resolve()
    exp_root = Path(args.experiments_root).expanduser().resolve()
    exports_root = Path(args.exports_root).expanduser().resolve()
    export_dir = ensure_dir(exports_root / args.export_subdir)

    all_rows: list[dict[str, object]] = []
    for dataset in datasets:
        banner("dataset", f"{dataset} | competitiveness")
        rows: list[dict[str, object]] = []
        train = queries = neighbors = None
        for seed in seeds:
            for method_name in methods:
                method = _make_method(method_name, dataset, seed=seed, m_max=args.m_max)
                exp_dir = new_experiment_dir(exp_root, dataset, getattr(method, "name", method_name), seed=seed)
                info(f"running {getattr(method, 'name', method_name)} seed={seed} -> {exp_dir}")
                if hasattr(method, "run"):
                    if train is None or queries is None or neighbors is None:
                        from datasets import load_search_dataset

                        train, queries, neighbors = load_search_dataset(dataset, data_root)
                    row = method.run(train, queries, neighbors, exp_dir, data_root=data_root)  # type: ignore[attr-defined]
                else:
                    row = method.run_competitiveness(  # type: ignore[attr-defined]
                        data_root=data_root,
                        exp_dir=exp_dir,
                        force_prepare=args.force_prepare,
                    )
                rows.append(row)
                all_rows.append(row)
        dataset_json = export_dir / f"{dataset}_summary.json"
        merged_rows = _merge_rows(_load_rows(dataset_json), rows)
        write_csv(export_dir / f"{dataset}_summary.csv", merged_rows)
        write_json(dataset_json, merged_rows)
        info(f"dataset summary -> {export_dir / f'{dataset}_summary.csv'}")
        print(export_dir / f"{dataset}_summary.csv")

    all_json = export_dir / "all_datasets_summary.json"
    merged_all = _merge_rows(_load_rows(all_json), all_rows)
    write_csv(export_dir / "all_datasets_summary.csv", merged_all)
    write_json(all_json, merged_all)
    info(f"all-datasets summary -> {export_dir / 'all_datasets_summary.csv'}")
    print(export_dir / "all_datasets_summary.csv")


if __name__ == "__main__":
    main()

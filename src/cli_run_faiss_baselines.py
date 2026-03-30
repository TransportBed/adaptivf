from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts import ensure_dir, new_experiment_dir, write_csv, write_json
from console import banner, info
from datasets import load_search_dataset, parse_datasets
from methods.faiss_baselines import make_method


_METHOD_ORDER = {"IVF": 0, "IVFPQ": 1, "HNSW": 2}


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return []


def _merge_rows(existing: list[dict[str, object]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in existing + new_rows:
        merged[(row.get("dataset"), row.get("method"), row.get("seed"))] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("dataset", "")),
            _METHOD_ORDER.get(str(row.get("method", "")), 999),
            int(row.get("seed", 0) or 0),
            str(row.get("method", "")),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal FAISS baselines for the AdaptIVF paper repo.")
    parser.add_argument("--datasets", default="glove10k")
    parser.add_argument("--methods", default="IVF,IVFPQ,HNSW")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--exports-root", default="paper_exports")
    parser.add_argument("--seeds", default="0")
    args = parser.parse_args()

    datasets = parse_datasets([args.datasets])
    methods = [token.strip() for token in args.methods.split(",") if token.strip()]
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    data_root = Path(args.data_root).expanduser().resolve()
    exp_root = Path(args.experiments_root).expanduser().resolve()
    exports_root = Path(args.exports_root).expanduser().resolve()
    export_dir = ensure_dir(exports_root / "faiss")

    all_rows: list[dict[str, object]] = []
    for dataset in datasets:
        train, queries, neighbors = load_search_dataset(dataset, data_root)
        banner("dataset", f"{dataset} | train={train.shape[0]} queries={queries.shape[0]} dim={train.shape[1]}")
        rows: list[dict[str, object]] = []
        for seed in seeds:
            for method_name in methods:
                method = make_method(method_name, dataset, seed=seed)
                exp_dir = new_experiment_dir(exp_root, dataset, method.name, seed=seed)
                info(f"running {method.name} seed={seed} -> {exp_dir}")
                metrics = method.run(train, queries, neighbors, exp_dir, data_root=data_root)
                row = {
                    "dataset": dataset,
                    "method": method.name,
                    "seed": metrics["seed"],
                    "recall_at_10": metrics["recall_at_10"],
                    "avg_computations": metrics["avg_computations"],
                    "avg_computations_mode": metrics["avg_computations_mode"],
                    "avg_candidates": metrics["avg_computations"],
                    "avg_candidates_mode": metrics["avg_computations_mode"],
                    "qps": metrics["qps"],
                    "index_overhead_mb": metrics["index_overhead_mb"],
                    "index_overhead_mode": metrics["index_overhead_mode"],
                    "serving_footprint_mb": metrics["serving_footprint_mb"],
                    "serving_footprint_mode": metrics["serving_footprint_mode"],
                    "index_size_mb": metrics["index_size_mb"],
                    "index_size_mode": metrics["index_size_mode"],
                    "query_mem_delta_mb": metrics["query_mem_delta_mb"],
                    "query_mem_delta_mode": metrics["query_mem_delta_mode"],
                    "experiment_dir": metrics["experiment_dir"],
                }
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

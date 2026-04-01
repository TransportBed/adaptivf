from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts import ensure_dir, new_experiment_dir, write_csv, write_json
from console import banner, info
from datasets import parse_datasets
from methods.bliss import Bliss, BlissKMeans
from methods.mlp_ivf import MlpIvf


_METHOD_ORDER = {"BLISS": 0, "BLISS-KMeans": 1, "MLP-IVF": 2}


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return []


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
    merged: dict[tuple[object, object, object, object], dict[str, object]] = {}
    for row in existing + new_rows:
        key = (
            _norm_text(row.get("dataset")),
            _norm_text(row.get("method")),
            _norm_int(row.get("probe_depth")),
            _norm_int(row.get("seed")),
        )
        normalized = dict(row)
        normalized["dataset"] = key[0]
        normalized["method"] = key[1]
        normalized["probe_depth"] = key[2]
        normalized["seed"] = key[3]
        merged[key] = normalized
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("dataset", "")),
            _METHOD_ORDER.get(str(row.get("method", "")), 999),
            int(row.get("probe_depth", 0)),
            int(row.get("seed", 0) or 0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone BLISS / BLISS-KMeans / MLP-IVF sweeps.")
    parser.add_argument("--datasets", default="glove10k")
    parser.add_argument("--methods", default="BLISS,BLISS-KMeans,MLP-IVF")
    parser.add_argument("--probes", default="5,10,20,40")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--exports-root", default="paper_exports")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    datasets = parse_datasets([args.datasets])
    methods = [token.strip() for token in args.methods.split(",") if token.strip()]
    probes = [int(token.strip()) for token in args.probes.split(",") if token.strip()]
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    data_root = Path(args.data_root).expanduser().resolve()
    exp_root = Path(args.experiments_root).expanduser().resolve()
    exports_root = Path(args.exports_root).expanduser().resolve()
    export_dir = ensure_dir(exports_root / "initialization")

    all_rows: list[dict[str, object]] = []
    for dataset in datasets:
        banner("dataset", f"{dataset} | initialization sweep")
        rows: list[dict[str, object]] = []
        for seed in seeds:
            for method_name in methods:
                key = method_name.strip().upper()
                if key == "BLISS":
                    method = Bliss(dataset, seed=seed)
                elif key in {"BLISS-KMEANS", "BLISS_KMEANS"}:
                    method = BlissKMeans(dataset, seed=seed)
                elif key == "MLP-IVF":
                    method = MlpIvf(dataset, seed=seed)
                else:
                    raise ValueError(f"Unsupported router method: {method_name}")
                exp_dir = new_experiment_dir(exp_root, dataset, method.name, seed=seed)
                info(f"running {method.name} seed={seed} -> {exp_dir}")
                method_rows = method.run_initialization_sweep(
                    data_root=data_root,
                    exp_dir=exp_dir,
                    probes=probes,
                    force_prepare=args.force_prepare,
                )
                rows.extend(method_rows)
                all_rows.extend(method_rows)
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

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from artifacts import write_csv, write_json
from presets import DATASETS, PAPER_DATASETS


BASELINE_ROWS = [
    {"method": "HNSW", "role_or_key_mechanism": "Graph baseline"},
    {"method": "IVF", "role_or_key_mechanism": "Static coarse quantization with fixed probing"},
    {"method": "IVFPQ", "role_or_key_mechanism": "IVF with PQ compression and ADC scan"},
    {"method": "BLISS", "role_or_key_mechanism": "Hash initialization and repeated partitions"},
    {"method": "BLISS-KMeans", "role_or_key_mechanism": "BLISS with k-means initialization"},
    {"method": "MLP-IVF", "role_or_key_mechanism": "BLISS-style MLP with IVF initialization"},
    {"method": "MLP-IVFPQ", "role_or_key_mechanism": "MLP-IVF with PQ compression"},
    {"method": "LIRA", "role_or_key_mechanism": "Learned partition selection with local indexes"},
    {"method": "AdaptIVF", "role_or_key_mechanism": "Single-backbone uncertainty-aware routing"},
    {"method": "AdaptIVF+PQ", "role_or_key_mechanism": "Compressed variant of AdaptIVF"},
]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [
        col
        for col in (
            "probe_depth",
            "recall_at_10",
            "avg_computations",
            "qps",
            "index_size_mb",
            "index_overhead_mb",
            "serving_footprint_mb",
            "query_mem_delta_mb",
            "rss_baseline_mb",
            "rss_serving_mb",
            "rss_peak_mb",
            "method_overhead_mb",
            "train_s",
            "threshold",
            "nprobe",
            "selected_part",
        )
        if col in df.columns and col not in group_cols
    ]
    agg = df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    agg["n_runs"] = df.groupby(group_cols, dropna=False).size().reset_index(name="n_runs")["n_runs"]
    return agg


def _write_table(path: Path, df: pd.DataFrame) -> None:
    rows = df.to_dict(orient="records") if not df.empty else []
    write_csv(path, rows)


def _out_relative_path(path: Path, *, out_root: Path) -> str:
    return os.path.relpath(path.resolve(), out_root.resolve()).replace(os.sep, "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit paper tables from completed AdaptIVF summaries.")
    parser.add_argument("--out-root", default="paper_exports")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    table_dir = out_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    for key in PAPER_DATASETS:
        spec = DATASETS[key]
        dataset_rows.append(
            {
                "dataset": spec.key,
                "N": spec.indexed_size,
                "Q": spec.query_count,
                "d": spec.dim,
                "metric": spec.metric,
                "B": spec.partitions,
            }
        )
    datasets_df = pd.DataFrame(dataset_rows)
    _write_table(table_dir / "main_datasets_table.csv", datasets_df)

    baselines_df = pd.DataFrame(BASELINE_ROWS)
    _write_table(table_dir / "main_baselines_table.csv", baselines_df)

    init_df = _load_csv(out_root / "initialization" / "all_datasets_summary.csv")
    init_table = _aggregate(init_df, ["dataset", "method", "probe_depth"]) if not init_df.empty else pd.DataFrame()
    _write_table(table_dir / "appendix_initialization_summary_table.csv", init_table)

    comp_df = _load_csv(out_root / "competitiveness" / "all_datasets_summary.csv")
    comp_table = _aggregate(comp_df, ["dataset", "method"]) if not comp_df.empty else pd.DataFrame()
    _write_table(table_dir / "appendix_competitiveness_summary_table.csv", comp_table)

    ablation_df = _load_csv(out_root / "ablations" / "all_datasets_summary.csv")
    ablation_table = _aggregate(ablation_df, ["dataset", "method"]) if not ablation_df.empty else pd.DataFrame()
    _write_table(table_dir / "main_ablation_summary_table.csv", ablation_table)

    payload = {
        "tables": {
            "main_datasets_table": _out_relative_path(table_dir / "main_datasets_table.csv", out_root=out_root),
            "main_baselines_table": _out_relative_path(table_dir / "main_baselines_table.csv", out_root=out_root),
            "appendix_initialization_summary_table": _out_relative_path(table_dir / "appendix_initialization_summary_table.csv", out_root=out_root),
            "appendix_competitiveness_summary_table": _out_relative_path(table_dir / "appendix_competitiveness_summary_table.csv", out_root=out_root),
            "main_ablation_summary_table": _out_relative_path(table_dir / "main_ablation_summary_table.csv", out_root=out_root),
        }
    }
    write_json(out_root / "tables_manifest.json", payload)
    print(out_root / "tables_manifest.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from artifacts import write_json
from presets import ABLATION_METHODS, DATASETS, PAPER_DATASETS


MAIN_METHOD_ORDER = [
    "HNSW",
    "IVF",
    "MLP-IVF",
    "BLISS",
    "LIRA",
    "AdaptIVF",
    "IVFPQ",
    "MLP-IVFPQ",
    "AdaptIVF+PQ",
]

ABLATION_FAMILIES = [
    ["AdaptIVF-Static", "AdaptIVF", "AdaptIVF-A4"],
    ["AdaptIVF-Static+PQ", "AdaptIVF+PQ", "AdaptIVF-A4+PQ"],
]

METHOD_COLORS = {
    "HNSW": "#1b9e77",
    "IVF": "#d95f02",
    "MLP-IVF": "#7570b3",
    "BLISS": "#e7298a",
    "LIRA": "#66a61e",
    "AdaptIVF": "#1f78b4",
    "IVFPQ": "#a6761d",
    "MLP-IVFPQ": "#6a3d9a",
    "AdaptIVF+PQ": "#b15928",
    "AdaptIVF-Static": "#6baed6",
    "AdaptIVF-A4": "#08306b",
    "AdaptIVF-Static+PQ": "#fdae6b",
    "AdaptIVF-A4+PQ": "#7f2704",
}

METHOD_MARKERS = {
    "HNSW": "o",
    "IVF": "s",
    "MLP-IVF": "^",
    "BLISS": "D",
    "LIRA": "P",
    "AdaptIVF": "*",
    "IVFPQ": "o",
    "MLP-IVFPQ": "^",
    "AdaptIVF+PQ": "*",
    "AdaptIVF-Static": "s",
    "AdaptIVF-A4": "D",
    "AdaptIVF-Static+PQ": "s",
    "AdaptIVF-A4+PQ": "D",
}


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
            "computation_min",
            "computation_max",
            "qps",
            "index_size_mb",
            "index_overhead_mb",
            "serving_footprint_mb",
            "query_mem_delta_mb",
            "build_total_s",
            "train_s",
            "threshold",
            "nprobe",
            "selected_part",
        )
        if col in df.columns and col not in group_cols
    ]
    out = df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    out["n_runs"] = df.groupby(group_cols, dropna=False).size().reset_index(name="n_runs")["n_runs"]
    return out


def _facet_axes(datasets: list[str]) -> tuple[plt.Figure, list[plt.Axes]]:
    n = len(datasets)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), squeeze=False)
    flat = list(axes.reshape(-1))
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def _present_datasets(df: pd.DataFrame) -> list[str]:
    present = [str(ds) for ds in sorted(set(df["dataset"].astype(str)))]
    ordered = [ds for ds in PAPER_DATASETS if ds in present]
    extras = [ds for ds in present if ds not in ordered]
    return ordered + extras


def _save_plot(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_csv(df: pd.DataFrame, stem: Path) -> None:
    stem.with_suffix(".csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)


def _method_style(method: str) -> tuple[str, str]:
    return METHOD_COLORS.get(method, "#333333"), METHOD_MARKERS.get(method, "o")


def _scatter_facets(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    stem: Path,
    title: str,
    method_order: list[str],
) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    datasets = _present_datasets(df)
    if not datasets:
        return
    fig, axes = _facet_axes(datasets)
    legend_handles = {}
    for ax, dataset in zip(axes, datasets):
        ds_df = df[df["dataset"] == dataset].copy()
        for method in method_order:
            sub = ds_df[ds_df["method"] == method]
            if sub.empty:
                continue
            color, marker = _method_style(method)
            handle = ax.scatter(
                sub[x],
                sub[y],
                color=color,
                marker=marker,
                s=70,
                alpha=0.95,
                label=method,
            )
            legend_handles.setdefault(method, handle)
        ax.set_title(dataset)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.legend(
        [legend_handles[m] for m in method_order if m in legend_handles],
        [m for m in method_order if m in legend_handles],
        loc="lower center",
        ncol=min(5, max(1, len(legend_handles))),
        bbox_to_anchor=(0.5, -0.02),
    )
    _save_csv(df, stem)
    _save_plot(fig, stem)


def _line_facets(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    stem: Path,
    title: str,
    method_order: list[str],
) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    datasets = _present_datasets(df)
    if not datasets:
        return
    fig, axes = _facet_axes(datasets)
    legend_handles = {}
    for ax, dataset in zip(axes, datasets):
        ds_df = df[df["dataset"] == dataset].copy()
        for method in method_order:
            sub = ds_df[ds_df["method"] == method].sort_values(x)
            if sub.empty:
                continue
            color, marker = _method_style(method)
            (line,) = ax.plot(
                sub[x],
                sub[y],
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=6,
                label=method,
            )
            legend_handles.setdefault(method, line)
        ax.set_title(dataset)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.legend(
        [legend_handles[m] for m in method_order if m in legend_handles],
        [m for m in method_order if m in legend_handles],
        loc="lower center",
        ncol=min(5, max(1, len(legend_handles))),
        bbox_to_anchor=(0.5, -0.02),
    )
    _save_csv(df, stem)
    _save_plot(fig, stem)


def _bucket_profile(exp_dir: Path, method: str) -> np.ndarray | None:
    if method == "LIRA":
        selected = exp_dir / "data" / "lira_bucket_sizes_selected.npy"
        if selected.exists():
            arr = np.load(selected).astype(np.float64, copy=False)
            return np.sort(arr)[::-1]
        return None
    lookups_dir = exp_dir / "staging" / "lookups"
    offset_paths = sorted(lookups_dir.glob("rep*_offsets.npy"))
    if not offset_paths:
        return None
    rep_counts = []
    for path in offset_paths:
        offsets = np.load(path).astype(np.int64, copy=False)
        rep_counts.append(np.diff(offsets).astype(np.float64, copy=False))
    max_len = max(arr.shape[0] for arr in rep_counts)
    stack = np.zeros((len(rep_counts), max_len), dtype=np.float64)
    for idx, arr in enumerate(rep_counts):
        stack[idx, : arr.shape[0]] = np.sort(arr)[::-1]
    return np.mean(stack, axis=0)


def _load_balance_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["dataset", "method"], dropna=False)
    for (dataset, method), sub in grouped:
        profiles = []
        for exp_dir in sub["experiment_dir"].dropna().astype(str):
            profile = _bucket_profile(Path(exp_dir), str(method))
            if profile is not None:
                profiles.append(profile)
        if not profiles:
            continue
        max_len = max(p.shape[0] for p in profiles)
        stack = np.zeros((len(profiles), max_len), dtype=np.float64)
        for idx, arr in enumerate(profiles):
            stack[idx, : arr.shape[0]] = arr
        mean_profile = np.mean(stack, axis=0)
        for idx, bucket_size in enumerate(mean_profile, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "bucket_rank": idx,
                    "bucket_rank_pct": 0.0 if max_len <= 1 else float(idx - 1) / float(max_len - 1),
                    "bucket_size": float(bucket_size),
                }
            )
    return pd.DataFrame(rows)


def _load_balance_plot(df: pd.DataFrame, stem: Path) -> None:
    if df.empty:
        return
    datasets = _present_datasets(df)
    if not datasets:
        return
    fig, axes = _facet_axes(datasets)
    legend_handles = {}
    for ax, dataset in zip(axes, datasets):
        ds_df = df[df["dataset"] == dataset]
        for method in sorted(set(ds_df["method"]), key=lambda m: MAIN_METHOD_ORDER.index(m) if m in MAIN_METHOD_ORDER else 999):
            sub = ds_df[ds_df["method"] == method].sort_values("bucket_rank")
            if sub.empty:
                continue
            color, marker = _method_style(str(method))
            (line,) = ax.plot(
                sub["bucket_rank"],
                sub["bucket_size"],
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=3,
                label=str(method),
            )
            legend_handles.setdefault(str(method), line)
        ax.set_title(dataset)
        ax.set_xlabel("bucket_rank")
        ax.set_ylabel("bucket_size")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Load balance by bucket rank")
    fig.legend(
        [legend_handles[m] for m in legend_handles],
        [m for m in legend_handles],
        loc="lower center",
        ncol=min(4, max(1, len(legend_handles))),
        bbox_to_anchor=(0.5, -0.02),
    )
    _save_csv(df, stem)
    _save_plot(fig, stem)


# ---------------------------------------------------------------------------
# Appendix: grouped-bar and range-bar helpers
# ---------------------------------------------------------------------------

def _grouped_bar_facets(
    df: pd.DataFrame,
    *,
    y: str,
    stem: Path,
    title: str,
    method_order: list[str],
    ylabel: str | None = None,
    log_y: bool = False,
) -> None:
    """Grouped bar chart: one group per method, faceted by dataset."""
    if df.empty or y not in df.columns:
        return
    datasets = _present_datasets(df)
    if not datasets:
        return
    fig, axes = _facet_axes(datasets)
    for ax, dataset in zip(axes, datasets):
        ds = df[df["dataset"] == dataset]
        methods = [m for m in method_order if m in ds["method"].values]
        vals = [float(ds[ds["method"] == m][y].iloc[0]) if not ds[ds["method"] == m].empty else 0.0 for m in methods]
        colors = [METHOD_COLORS.get(m, "#333333") for m in methods]
        x_pos = np.arange(len(methods))
        ax.bar(x_pos, vals, color=colors, edgecolor="white", width=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel or y)
        ax.set_title(dataset)
        ax.grid(True, axis="y", alpha=0.25)
        if log_y:
            ax.set_yscale("log")
    fig.suptitle(title)
    _save_csv(df, stem)
    _save_plot(fig, stem)


def _range_bar_facets(
    df: pd.DataFrame,
    *,
    y_mid: str,
    y_lo: str,
    y_hi: str,
    stem: Path,
    title: str,
    method_order: list[str],
    ylabel: str | None = None,
) -> None:
    """Bar chart with error bars showing per-query computation range."""
    if df.empty or y_mid not in df.columns or y_lo not in df.columns or y_hi not in df.columns:
        return
    datasets = _present_datasets(df)
    if not datasets:
        return
    fig, axes = _facet_axes(datasets)
    for ax, dataset in zip(axes, datasets):
        ds = df[df["dataset"] == dataset].dropna(subset=[y_mid, y_lo, y_hi])
        methods = [m for m in method_order if m in ds["method"].values]
        if not methods:
            ax.set_title(dataset)
            continue
        mids, los, his = [], [], []
        for m in methods:
            row = ds[ds["method"] == m].iloc[0]
            mid = float(row[y_mid])
            lo = float(row[y_lo])
            hi = float(row[y_hi])
            mids.append(mid)
            los.append(max(0, mid - lo))
            his.append(max(0, hi - mid))
        colors = [METHOD_COLORS.get(m, "#333333") for m in methods]
        x_pos = np.arange(len(methods))
        ax.bar(x_pos, mids, color=colors, edgecolor="white", width=0.7)
        ax.errorbar(x_pos, mids, yerr=[los, his], fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel or y_mid)
        ax.set_title(dataset)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title)
    _save_csv(df, stem)
    _save_plot(fig, stem)


def _ablation_plot(df: pd.DataFrame, *, x: str, stem: Path, title: str) -> None:
    if df.empty or x not in df.columns or "recall_at_10" not in df.columns:
        return
    datasets = _present_datasets(df)
    if not datasets:
        return
    fig, axes = _facet_axes(datasets)
    for ax, dataset in zip(axes, datasets):
        ds_df = df[df["dataset"] == dataset]
        for family in ABLATION_FAMILIES:
            fam_rows = ds_df[ds_df["method"].isin(family)].copy()
            if fam_rows.empty:
                continue
            fam_rows["method"] = pd.Categorical(fam_rows["method"], categories=family, ordered=True)
            fam_rows = fam_rows.sort_values("method")
            color = METHOD_COLORS.get(family[1], "#1f78b4")
            ax.plot(fam_rows[x], fam_rows["recall_at_10"], color=color, linewidth=1.8, alpha=0.9)
            for _, row in fam_rows.iterrows():
                _, marker = _method_style(str(row["method"]))
                ax.scatter(row[x], row["recall_at_10"], color=METHOD_COLORS.get(str(row["method"]), color), marker=marker, s=75)
        ax.set_title(dataset)
        ax.set_xlabel(x)
        ax.set_ylabel("recall_at_10")
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    _save_csv(df, stem)
    _save_plot(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper plots from completed AdaptIVF summaries.")
    parser.add_argument("--out-root", default="paper_exports")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    init_raw = _load_csv(out_root / "initialization" / "all_datasets_summary.csv")
    comp_raw = _load_csv(out_root / "competitiveness" / "all_datasets_summary.csv")
    ablation_raw = _load_csv(out_root / "ablations" / "all_datasets_summary.csv")
    init_df = _aggregate(init_raw, ["dataset", "method", "probe_depth"])
    comp_df = _aggregate(comp_raw, ["dataset", "method"])
    ablation_df = _aggregate(ablation_raw, ["dataset", "method"])

    plot_names: list[str] = []

    if not init_df.empty:
        stem = out_root / "init_recall_vs_m_facets"
        _line_facets(init_df, x="probe_depth", y="recall_at_10", stem=stem, title="Initialization: Recall@10 vs probe depth", method_order=["BLISS", "BLISS-KMeans", "MLP-IVF"])
        plot_names.append(stem.stem)
        stem = out_root / "init_recall_vs_avg_computations_facets"
        _line_facets(init_df, x="avg_computations", y="recall_at_10", stem=stem, title="Initialization: Recall@10 vs avg computations", method_order=["BLISS", "BLISS-KMeans", "MLP-IVF"])
        plot_names.append(stem.stem)

    if not comp_df.empty:
        stem = out_root / "main_recall_vs_avg_computations_facets"
        _scatter_facets(comp_df, x="avg_computations", y="recall_at_10", stem=stem, title="Main: Recall@10 vs avg computations", method_order=MAIN_METHOD_ORDER)
        plot_names.append(stem.stem)
        stem = out_root / "main_recall_vs_qps_facets"
        _scatter_facets(comp_df, x="qps", y="recall_at_10", stem=stem, title="Main: Recall@10 vs QPS", method_order=MAIN_METHOD_ORDER)
        plot_names.append(stem.stem)
        stem = out_root / "main_recall_vs_index_overhead_facets"
        _scatter_facets(comp_df, x="index_size_mb", y="recall_at_10", stem=stem, title="Main: Recall@10 vs index overhead", method_order=MAIN_METHOD_ORDER)
        plot_names.append(stem.stem)
        stem = out_root / "main_query_mem_vs_index_overhead_facets"
        _scatter_facets(comp_df, x="index_size_mb", y="query_mem_delta_mb", stem=stem, title="Main: query memory vs index overhead", method_order=MAIN_METHOD_ORDER)
        plot_names.append(stem.stem)
        load_balance_df = _load_balance_rows(comp_raw)
        stem = out_root / "main_load_balance_bucket_rank_facets"
        _load_balance_plot(load_balance_df, stem)
        if not load_balance_df.empty:
            plot_names.append(stem.stem)

    if not ablation_df.empty:
        stem = out_root / "ablation_recall_vs_avg_computations_facets"
        _ablation_plot(ablation_df, x="avg_computations", stem=stem, title="AdaptIVF ablations: Recall@10 vs avg computations")
        plot_names.append(stem.stem)
        stem = out_root / "ablation_recall_vs_index_overhead_facets"
        _ablation_plot(ablation_df, x="index_size_mb", stem=stem, title="AdaptIVF ablations: Recall@10 vs index overhead")
        plot_names.append(stem.stem)

    # ---- Appendix plots ----

    # A1: Training time grouped bars (learned methods only)
    learned_methods = [m for m in MAIN_METHOD_ORDER if m not in ("HNSW", "IVF", "IVFPQ")]
    if not comp_df.empty and "train_s" in comp_df.columns:
        train_df = comp_df.dropna(subset=["train_s"])
        if not train_df.empty:
            stem = out_root / "appendix_training_time_bars"
            _grouped_bar_facets(train_df, y="train_s", stem=stem, title="Appendix: Training time (s) by method", method_order=learned_methods, ylabel="train_s (seconds)")
            plot_names.append(stem.stem)

    # A2: Index overhead grouped bars (all methods)
    if not comp_df.empty and "serving_footprint_mb" in comp_df.columns:
        stem = out_root / "appendix_serving_footprint_bars"
        _grouped_bar_facets(comp_df, y="serving_footprint_mb", stem=stem, title="Appendix: Index overhead excl. base vectors (MB)", method_order=MAIN_METHOD_ORDER, ylabel="serving_footprint_mb", log_y=True)
        plot_names.append(stem.stem)

    # A3: Computation range bars (methods that report min/max)
    if not comp_df.empty and "computation_min" in comp_df.columns and "computation_max" in comp_df.columns:
        range_df = comp_df.dropna(subset=["avg_computations", "computation_min", "computation_max"])
        if not range_df.empty:
            stem = out_root / "appendix_computation_range_bars"
            _range_bar_facets(range_df, y_mid="avg_computations", y_lo="computation_min", y_hi="computation_max", stem=stem, title="Appendix: Per-query computation range (min / avg / max)", method_order=MAIN_METHOD_ORDER, ylabel="distance computations")
            plot_names.append(stem.stem)

    # A4: QPS grouped bars (all methods)
    if not comp_df.empty and "qps" in comp_df.columns:
        stem = out_root / "appendix_qps_bars"
        _grouped_bar_facets(comp_df, y="qps", stem=stem, title="Appendix: Queries per second by method", method_order=MAIN_METHOD_ORDER, ylabel="QPS", log_y=True)
        plot_names.append(stem.stem)

    # A5: Ablation QPS + training time grouped bars
    ablation_order = list(ABLATION_METHODS)
    if not ablation_df.empty and "qps" in ablation_df.columns:
        stem = out_root / "appendix_ablation_qps_bars"
        _grouped_bar_facets(ablation_df, y="qps", stem=stem, title="Appendix: Ablation QPS comparison", method_order=ablation_order, ylabel="QPS")
        plot_names.append(stem.stem)
    if not ablation_df.empty and "train_s" in ablation_df.columns:
        abl_train = ablation_df.dropna(subset=["train_s"])
        if not abl_train.empty:
            stem = out_root / "appendix_ablation_training_time_bars"
            _grouped_bar_facets(abl_train, y="train_s", stem=stem, title="Appendix: Ablation training time comparison", method_order=ablation_order, ylabel="train_s (seconds)")
            plot_names.append(stem.stem)

    payload = {"plots": plot_names}
    write_json(out_root / "plots_manifest.json", payload)
    print(out_root / "plots_manifest.json")


if __name__ == "__main__":
    main()

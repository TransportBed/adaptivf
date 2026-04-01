from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from artifacts import write_json
from presets import ABLATION_METHODS, DATASETS, PAPER_DATASETS


MAIN_METHOD_ORDER = [
    "HNSW",
    "IVF",
    "IVFPQ",
    "BLISS",
    "MLP-IVF",
    "MLP-IVFPQ",
    "LIRA",
    "AdaptIVF",
    "AdaptIVF-m80",
    "AdaptIVF+PQ",
    "AdaptIVF+PQ-m80",
]

MAIN_PLOT_METHOD_ORDER = [
    "HNSW",
    "IVF",
    "IVFPQ",
    "BLISS",
    "LIRA",
    "AdaptIVF",
    "AdaptIVF-m80",
    "AdaptIVF+PQ",
    "AdaptIVF+PQ-m80",
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
    "AdaptIVF-m80": "#08519c",
    "IVFPQ": "#a6761d",
    "MLP-IVFPQ": "#6a3d9a",
    "AdaptIVF+PQ": "#b15928",
    "AdaptIVF+PQ-m80": "#8c2d04",
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
    "AdaptIVF-m80": "X",
    "IVFPQ": "o",
    "MLP-IVFPQ": "^",
    "AdaptIVF+PQ": "*",
    "AdaptIVF+PQ-m80": "X",
    "AdaptIVF-Static": "s",
    "AdaptIVF-A4": "D",
    "AdaptIVF-Static+PQ": "s",
    "AdaptIVF-A4+PQ": "D",
}

FACET_WIDTH_PER_PANEL = 4.9
FACET_FIG_HEIGHT = 3.95
TITLE_FONT_SIZE = 17
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 15
LEGEND_FONT_SIZE = 20
MAIN_LEGEND_FONT_SIZE = 16
LOAD_BALANCE_LEGEND_FONT_SIZE = 16
SUPXLABEL_Y = 0.050
SUPYLABEL_X = 0.022
SCATTER_MARKER_SIZE = 84
LINE_MARKER_SIZE = 7
LOAD_BALANCE_MARKER_SIZE = 4
ABLATION_MARKER_SIZE = 88

METHOD_DISPLAY_NAMES = {
    "AdaptIVF": "AdaptIVF-10",
    "AdaptIVF-m80": "AdaptIVF-80",
    "AdaptIVF+PQ": "AdaptIVF+PQ-10",
    "AdaptIVF+PQ-m80": "AdaptIVF+PQ-80",
}

DATASET_DISPLAY_NAMES = {
    "glove": "GloVe",
    "sift": "SIFT",
    "gist": "GIST",
    "deep1m": "Deep1M",
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
            "rss_baseline_mb",
            "rss_serving_mb",
            "rss_peak_mb",
            "method_overhead_mb",
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
    ncols = n
    nrows = 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FACET_WIDTH_PER_PANEL * ncols, FACET_FIG_HEIGHT),
        squeeze=False,
    )
    flat = list(axes.reshape(-1))
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def _facet_axes_grid(datasets: list[str], *, ncols: int) -> tuple[plt.Figure, list[plt.Axes]]:
    n = len(datasets)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FACET_WIDTH_PER_PANEL * ncols, FACET_FIG_HEIGHT * nrows * 0.95),
        squeeze=False,
    )
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
    # Trim whitespace but preserve room for the shared labels and upper legend
    fig.tight_layout(rect=[0.02, 0.085, 1, 0.87])
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_csv(df: pd.DataFrame, stem: Path) -> None:
    stem.with_suffix(".csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)


def _method_style(method: str) -> tuple[str, str]:
    return METHOD_COLORS.get(method, "#333333"), METHOD_MARKERS.get(method, "o")


def _method_label(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method)


def _dataset_label(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset, dataset)


def _apply_numeric_tick_policy(ax: plt.Axes, *, x: bool = True, y: bool = False) -> None:
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))


def _draw_shared_legend(
    fig: plt.Figure,
    handles: list[object],
    labels: list[str],
    *,
    single_row: bool = False,
    fontsize: int | None = None,
    ncols_override: int | None = None,
) -> None:
    if not handles or not labels:
        return
    if single_row:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            ncols=len(labels),
            bbox_to_anchor=(0.5, 0.90),
            frameon=False,
            fontsize=fontsize or MAIN_LEGEND_FONT_SIZE,
            columnspacing=1.1,
            handletextpad=0.35,
            borderaxespad=0.0,
        )
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncols_override or max(1, min(4, len(labels))),
        ncols=ncols_override or max(1, min(4, len(labels))),
        bbox_to_anchor=(0.5, 0.89),
        frameon=False,
        fontsize=fontsize or LEGEND_FONT_SIZE,
        columnspacing=0.9,
        handletextpad=0.5,
        handlelength=1.8,
        borderaxespad=0.0,
    )


def _load_serving_ram(out_root: Path) -> pd.DataFrame:
    csv_path = out_root / "serving_ram.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    json_path = out_root / "serving_ram.json"
    if json_path.exists():
        return pd.DataFrame(pd.read_json(json_path))
    return pd.DataFrame()


def _restrict_methods(df: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    if df.empty or "method" not in df.columns:
        return df
    allowed = set(method_order)
    return df[df["method"].isin(allowed)].copy()


def _scatter_facets(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    stem: Path,
    title: str,
    method_order: list[str],
    xlabel: str | None = None,
    ylabel: str | None = None,
    highlight_pairs: list[tuple[str, str]] | None = None,
    legend_single_row: bool = True,
    legend_fontsize: int | None = None,
    legend_ncols_override: int | None = None,
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
        if highlight_pairs:
            for left_method, right_method in highlight_pairs:
                left = ds_df[ds_df["method"] == left_method]
                right = ds_df[ds_df["method"] == right_method]
                if left.empty or right.empty:
                    continue
                left_row = left.iloc[0]
                right_row = right.iloc[0]
                color, _ = _method_style(left_method)
                ax.plot(
                    [left_row[x], right_row[x]],
                    [left_row[y], right_row[y]],
                    linestyle=":",
                    linewidth=2.0,
                    color=color,
                    alpha=0.35,
                    zorder=1,
                )
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
                s=SCATTER_MARKER_SIZE,
                alpha=0.95,
                label=_method_label(method),
                zorder=2,
            )
            legend_handles.setdefault(method, handle)
        ax.set_title(_dataset_label(dataset), fontsize=TITLE_FONT_SIZE)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        _apply_numeric_tick_policy(ax, x=True, y=False)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.supxlabel(xlabel or x, fontsize=AXIS_LABEL_FONT_SIZE, y=SUPXLABEL_Y)
    fig.supylabel(ylabel or y, fontsize=AXIS_LABEL_FONT_SIZE, x=SUPYLABEL_X)
    ordered_methods = [m for m in method_order if m in legend_handles]
    _draw_shared_legend(
        fig,
        [legend_handles[m] for m in ordered_methods],
        [_method_label(m) for m in ordered_methods],
        single_row=legend_single_row,
        fontsize=legend_fontsize,
        ncols_override=legend_ncols_override,
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
    xlabel: str | None = None,
    ylabel: str | None = None,
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
                markersize=LINE_MARKER_SIZE,
                label=_method_label(method),
            )
            legend_handles.setdefault(method, line)
        ax.set_title(_dataset_label(dataset), fontsize=TITLE_FONT_SIZE)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        _apply_numeric_tick_policy(ax, x=True, y=False)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.supxlabel(xlabel or x, fontsize=AXIS_LABEL_FONT_SIZE, y=SUPXLABEL_Y)
    fig.supylabel(ylabel or y, fontsize=AXIS_LABEL_FONT_SIZE, x=SUPYLABEL_X)
    ordered_methods = [m for m in method_order if m in legend_handles]
    _draw_shared_legend(
        fig,
        [legend_handles[m] for m in ordered_methods],
        [_method_label(m) for m in ordered_methods],
        single_row=True,
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
    datasets = [ds for ds in PAPER_DATASETS if ds in set(df["dataset"].astype(str))] + [
        ds for ds in PAPER_DATASETS if ds not in set(df["dataset"].astype(str))
    ]
    if not datasets:
        return
    ncols = 2
    fig, axes = _facet_axes_grid(datasets, ncols=ncols)
    legend_handles = {}
    for ax, dataset in zip(axes, datasets):
        ds_df = df[df["dataset"] == dataset]
        has_data = not ds_df.empty
        for method in sorted(set(ds_df["method"]), key=lambda m: MAIN_METHOD_ORDER.index(m) if m in MAIN_METHOD_ORDER else 999):
            sub = ds_df[ds_df["method"] == method].sort_values("bucket_rank")
            if sub.empty:
                continue
            color, marker = _method_style(str(method))
            (line,) = ax.plot(
                sub["bucket_rank"],
                sub["bucket_size"],
                color=color,
                linewidth=2.0,
                label=_method_label(str(method)),
            )
            legend_handles.setdefault(str(method), line)
        ax.set_title(_dataset_label(dataset), fontsize=TITLE_FONT_SIZE)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        if has_data:
            _apply_numeric_tick_policy(ax, x=True, y=False)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                "pending",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=TICK_LABEL_FONT_SIZE,
                color="#888888",
            )
    for idx, ax in enumerate(axes):
        if idx % ncols != 0:
            ax.tick_params(axis="y", labelleft=False)
    fig.supxlabel("bucket_rank", fontsize=AXIS_LABEL_FONT_SIZE, y=SUPXLABEL_Y)
    fig.supylabel("bucket_size", fontsize=AXIS_LABEL_FONT_SIZE, x=SUPYLABEL_X)
    ordered_methods = [m for m in MAIN_PLOT_METHOD_ORDER if m in legend_handles]
    _draw_shared_legend(
        fig,
        [legend_handles[m] for m in ordered_methods],
        [_method_label(m) for m in ordered_methods],
        single_row=False,
        fontsize=LOAD_BALANCE_LEGEND_FONT_SIZE,
        ncols_override=3,
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
    ncols: int | None = None,
    fixed_method_axis: bool = False,
) -> None:
    """Grouped bar chart: one group per method, faceted by dataset."""
    if df.empty or y not in df.columns:
        return
    datasets = _present_datasets(df)
    if not datasets:
        return
    fig, axes = _facet_axes_grid(datasets, ncols=ncols or len(datasets))
    for ax, dataset in zip(axes, datasets):
        ds = df[df["dataset"] == dataset]
        methods = list(method_order) if fixed_method_axis else [m for m in method_order if m in ds["method"].values]
        vals = [float(ds[ds["method"] == m][y].iloc[0]) if not ds[ds["method"] == m].empty else np.nan for m in methods]
        colors = [METHOD_COLORS.get(m, "#333333") for m in methods]
        x_pos = np.arange(len(methods))
        ax.bar(x_pos, vals, color=colors, edgecolor="white", width=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_method_label(m) for m in methods], rotation=45, ha="right", fontsize=TICK_LABEL_FONT_SIZE)
        ax.set_title(_dataset_label(dataset), fontsize=TITLE_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
        if log_y:
            ax.set_yscale("log")
    if ncols and ncols > 1:
        for idx, ax in enumerate(axes):
            if idx % ncols != 0:
                ax.tick_params(axis="y", labelleft=False)
            if idx < len(axes) - ncols:
                ax.tick_params(axis="x", labelbottom=False)
    else:
        for ax in axes[1:]:
            ax.tick_params(axis="y", labelleft=False)
    fig.supylabel(ylabel or y, fontsize=AXIS_LABEL_FONT_SIZE, x=SUPYLABEL_X)
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
            ax.set_title(_dataset_label(dataset))
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
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=TICK_LABEL_FONT_SIZE)
        ax.set_title(_dataset_label(dataset), fontsize=TITLE_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.supylabel(ylabel or y_mid, fontsize=AXIS_LABEL_FONT_SIZE, x=SUPYLABEL_X)
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
                ax.scatter(
                    row[x],
                    row["recall_at_10"],
                    color=METHOD_COLORS.get(str(row["method"]), color),
                    marker=marker,
                    s=ABLATION_MARKER_SIZE,
                )
        ax.set_title(_dataset_label(dataset), fontsize=TITLE_FONT_SIZE)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        _apply_numeric_tick_policy(ax, x=True, y=False)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.supxlabel(x, fontsize=AXIS_LABEL_FONT_SIZE, y=SUPXLABEL_Y)
    fig.supylabel("recall_at_10", fontsize=AXIS_LABEL_FONT_SIZE, x=SUPYLABEL_X)
    _save_csv(df, stem)
    _save_plot(fig, stem)


def _ternary_point(recall_w: float, compute_w: float, storage_w: float) -> tuple[float, float]:
    h = np.sqrt(3.0) / 2.0
    x = storage_w + 0.5 * recall_w
    y = h * recall_w
    return float(x), float(y)


def _tradeoff_triangle_facets(
    df: pd.DataFrame,
    *,
    stem: Path,
    method_order: list[str],
    highlight_pairs: list[tuple[str, str]] | None = None,
) -> None:
    needed = {"dataset", "method", "recall_at_10", "avg_computations", "index_overhead_mb"}
    if df.empty or not needed.issubset(df.columns):
        return
    tri_df = df.dropna(subset=["recall_at_10", "avg_computations", "index_overhead_mb"]).copy()
    tri_df = tri_df[(tri_df["avg_computations"] > 0) & (tri_df["index_overhead_mb"] > 0)]
    if tri_df.empty:
        return
    rows: list[dict[str, float | str]] = []
    datasets = _present_datasets(tri_df)
    fig, axes = _facet_axes(datasets)
    h = np.sqrt(3.0) / 2.0
    legend_handles = {}
    for ax, dataset in zip(axes, datasets):
        ds = tri_df[tri_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        max_recall = float(ds["recall_at_10"].max())
        min_compute = float(ds["avg_computations"].min())
        min_storage = float(ds["index_overhead_mb"].min())
        if max_recall <= 0 or min_compute <= 0 or min_storage <= 0:
            continue
        ds["recall_score"] = ds["recall_at_10"] / max_recall
        ds["compute_efficiency"] = min_compute / ds["avg_computations"]
        ds["storage_efficiency"] = min_storage / ds["index_overhead_mb"]
        ds["score_sum"] = ds["recall_score"] + ds["compute_efficiency"] + ds["storage_efficiency"]
        ds["recall_weight"] = ds["recall_score"] / ds["score_sum"]
        ds["compute_weight"] = ds["compute_efficiency"] / ds["score_sum"]
        ds["storage_weight"] = ds["storage_efficiency"] / ds["score_sum"]
        coords = ds.apply(
            lambda row: _ternary_point(
                float(row["recall_weight"]),
                float(row["compute_weight"]),
                float(row["storage_weight"]),
            ),
            axis=1,
        )
        ds["plot_x"] = [xy[0] for xy in coords]
        ds["plot_y"] = [xy[1] for xy in coords]
        rows.extend(ds.to_dict("records"))

        boundary = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h], [0.0, 0.0]])
        ax.plot(boundary[:, 0], boundary[:, 1], color="#666666", linewidth=1.4, zorder=0)
        for t in (0.2, 0.4, 0.6, 0.8):
            grid_color = "#bbbbbb"
            alpha = 0.35
            p1 = _ternary_point(t, 1 - t, 0)
            p2 = _ternary_point(t, 0, 1 - t)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=0.8, alpha=alpha, zorder=0)
            p1 = _ternary_point(1 - t, t, 0)
            p2 = _ternary_point(0, t, 1 - t)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=0.8, alpha=alpha, zorder=0)
            p1 = _ternary_point(1 - t, 0, t)
            p2 = _ternary_point(0, 1 - t, t)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=0.8, alpha=alpha, zorder=0)

        if highlight_pairs:
            for left_method, right_method in highlight_pairs:
                left = ds[ds["method"] == left_method]
                right = ds[ds["method"] == right_method]
                if left.empty or right.empty:
                    continue
                left_row = left.iloc[0]
                right_row = right.iloc[0]
                color, _ = _method_style(left_method)
                ax.plot(
                    [left_row["plot_x"], right_row["plot_x"]],
                    [left_row["plot_y"], right_row["plot_y"]],
                    linestyle=":",
                    linewidth=2.0,
                    color=color,
                    alpha=0.35,
                    zorder=1,
                )

        for method in method_order:
            sub = ds[ds["method"] == method]
            if sub.empty:
                continue
            color, marker = _method_style(method)
            handle = ax.scatter(
                sub["plot_x"],
                sub["plot_y"],
                color=color,
                marker=marker,
                s=SCATTER_MARKER_SIZE,
                alpha=0.95,
                zorder=2,
                label=_method_label(method),
            )
            legend_handles.setdefault(method, handle)

        ax.text(0.5, h + 0.06, "Recall", ha="center", va="center", fontsize=AXIS_LABEL_FONT_SIZE - 1)
        ax.text(-0.03, -0.06, "Compute Efficiency", ha="left", va="top", fontsize=AXIS_LABEL_FONT_SIZE - 3)
        ax.text(1.03, -0.06, "Storage Efficiency", ha="right", va="top", fontsize=AXIS_LABEL_FONT_SIZE - 3)
        ax.set_title(_dataset_label(dataset), fontsize=TITLE_FONT_SIZE)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.10, h + 0.10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ordered_methods = [m for m in method_order if m in legend_handles]
    _draw_shared_legend(
        fig,
        [legend_handles[m] for m in ordered_methods],
        [_method_label(m) for m in ordered_methods],
        single_row=True,
    )
    _save_csv(pd.DataFrame(rows), stem)
    _save_plot(fig, stem)


def _tradeoff_triangle_aggregate(
    df: pd.DataFrame,
    *,
    stem: Path,
    method_order: list[str],
    highlight_pairs: list[tuple[str, str]] | None = None,
) -> None:
    needed = {"dataset", "method", "recall_at_10", "avg_computations", "index_overhead_mb"}
    if df.empty or not needed.issubset(df.columns):
        return
    tri_df = df.dropna(subset=["recall_at_10", "avg_computations", "index_overhead_mb"]).copy()
    tri_df = tri_df[(tri_df["avg_computations"] > 0) & (tri_df["index_overhead_mb"] > 0)]
    if tri_df.empty:
        return

    norm_rows: list[dict[str, float | str]] = []
    for dataset in _present_datasets(tri_df):
        ds = tri_df[tri_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        max_recall = float(ds["recall_at_10"].max())
        min_compute = float(ds["avg_computations"].min())
        min_storage = float(ds["index_overhead_mb"].min())
        if max_recall <= 0 or min_compute <= 0 or min_storage <= 0:
            continue
        ds["recall_score"] = ds["recall_at_10"] / max_recall
        ds["compute_efficiency"] = min_compute / ds["avg_computations"]
        ds["storage_efficiency"] = min_storage / ds["index_overhead_mb"]
        ds["score_sum"] = ds["recall_score"] + ds["compute_efficiency"] + ds["storage_efficiency"]
        ds["recall_weight"] = ds["recall_score"] / ds["score_sum"]
        ds["compute_weight"] = ds["compute_efficiency"] / ds["score_sum"]
        ds["storage_weight"] = ds["storage_efficiency"] / ds["score_sum"]
        norm_rows.extend(ds.to_dict("records"))

    norm_df = pd.DataFrame(norm_rows)
    if norm_df.empty:
        return
    agg = (
        norm_df.groupby("method", dropna=False)[["recall_weight", "compute_weight", "storage_weight"]]
        .mean()
        .reset_index()
    )
    coords = agg.apply(
        lambda row: _ternary_point(
            float(row["recall_weight"]),
            float(row["compute_weight"]),
            float(row["storage_weight"]),
        ),
        axis=1,
    )
    agg["plot_x"] = [xy[0] for xy in coords]
    agg["plot_y"] = [xy[1] for xy in coords]

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    h = np.sqrt(3.0) / 2.0
    boundary = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h], [0.0, 0.0]])
    ax.plot(boundary[:, 0], boundary[:, 1], color="#666666", linewidth=1.6, zorder=0)
    for t in (0.2, 0.4, 0.6, 0.8):
        grid_color = "#bbbbbb"
        alpha = 0.35
        p1 = _ternary_point(t, 1 - t, 0)
        p2 = _ternary_point(t, 0, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=0.8, alpha=alpha, zorder=0)
        p1 = _ternary_point(1 - t, t, 0)
        p2 = _ternary_point(0, t, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=0.8, alpha=alpha, zorder=0)
        p1 = _ternary_point(1 - t, 0, t)
        p2 = _ternary_point(0, 1 - t, t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=0.8, alpha=alpha, zorder=0)

    if highlight_pairs:
        for left_method, right_method in highlight_pairs:
            left = agg[agg["method"] == left_method]
            right = agg[agg["method"] == right_method]
            if left.empty or right.empty:
                continue
            left_row = left.iloc[0]
            right_row = right.iloc[0]
            color, _ = _method_style(left_method)
            ax.plot(
                [left_row["plot_x"], right_row["plot_x"]],
                [left_row["plot_y"], right_row["plot_y"]],
                linestyle=":",
                linewidth=2.0,
                color=color,
                alpha=0.35,
                zorder=1,
            )

    legend_handles = {}
    for method in method_order:
        sub = agg[agg["method"] == method]
        if sub.empty:
            continue
        color, marker = _method_style(method)
        handle = ax.scatter(
            sub["plot_x"],
            sub["plot_y"],
            color=color,
            marker=marker,
            s=SCATTER_MARKER_SIZE + 14,
            alpha=0.95,
            zorder=2,
            label=_method_label(method),
        )
        legend_handles[method] = handle

    ax.text(0.5, h + 0.07, "Recall", ha="center", va="center", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.text(-0.04, -0.07, "Compute Efficiency", ha="left", va="top", fontsize=AXIS_LABEL_FONT_SIZE - 1)
    ax.text(1.04, -0.07, "Storage Efficiency", ha="right", va="top", fontsize=AXIS_LABEL_FONT_SIZE - 1)
    ax.set_xlim(-0.06, 1.26)
    ax.set_ylim(-0.10, h + 0.10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ordered_methods = [m for m in method_order if m in legend_handles]
    ax.legend(
        [legend_handles[m] for m in ordered_methods],
        [_method_label(m) for m in ordered_methods],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=MAIN_LEGEND_FONT_SIZE,
        ncol=1,
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.6,
    )
    _save_csv(agg, stem)
    _save_plot(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper plots from completed AdaptIVF summaries.")
    parser.add_argument("--out-root", default="paper_exports")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    init_raw = _load_csv(out_root / "initialization" / "all_datasets_summary.csv")
    comp_raw = _load_csv(out_root / "competitiveness" / "all_datasets_summary.csv")
    ablation_raw = _load_csv(out_root / "ablations" / "all_datasets_summary.csv")
    serving_ram_raw = _load_serving_ram(out_root)
    if not comp_raw.empty and not serving_ram_raw.empty and "experiment" in serving_ram_raw.columns:
        serving_ram_raw = serving_ram_raw.rename(columns={"experiment": "experiment_dir"})
        merge_cols = ["experiment_dir", "rss_baseline_mb", "rss_serving_mb", "rss_peak_mb", "method_overhead_mb"]
        serving_ram_raw = serving_ram_raw[[col for col in merge_cols if col in serving_ram_raw.columns]].drop_duplicates(subset=["experiment_dir"])
        comp_raw = comp_raw.merge(serving_ram_raw, on="experiment_dir", how="left")
    init_df = _aggregate(init_raw, ["dataset", "method", "probe_depth"])
    comp_df = _aggregate(comp_raw, ["dataset", "method"])
    ablation_df = _aggregate(ablation_raw, ["dataset", "method"])

    plot_names: list[str] = []

    if not init_df.empty:
        stem = out_root / "main_init_recall_vs_m_facets"
        _line_facets(init_df, x="probe_depth", y="recall_at_10", stem=stem, title="Initialization: Recall@10 vs probe depth", method_order=["BLISS", "BLISS-KMeans", "MLP-IVF"])
        plot_names.append(stem.stem)
        stem = out_root / "appendix_init_recall_vs_avg_computations_facets"
        _line_facets(init_df, x="avg_computations", y="recall_at_10", stem=stem, title="Initialization: Recall@10 vs avg computations", method_order=["BLISS", "BLISS-KMeans", "MLP-IVF"])
        plot_names.append(stem.stem)

    if not comp_df.empty:
        comp_main_df = _restrict_methods(comp_df, MAIN_PLOT_METHOD_ORDER)
        stem = out_root / "main_recall_vs_avg_computations_facets"
        _scatter_facets(
            comp_main_df,
            x="avg_computations",
            y="recall_at_10",
            stem=stem,
            title="Main: Recall@10 vs avg computations",
            method_order=MAIN_PLOT_METHOD_ORDER,
            highlight_pairs=[
                ("AdaptIVF", "AdaptIVF-m80"),
                ("AdaptIVF+PQ", "AdaptIVF+PQ-m80"),
            ],
        )
        plot_names.append(stem.stem)
        stem = out_root / "appendix_recall_vs_qps_facets"
        _scatter_facets(comp_main_df, x="qps", y="recall_at_10", stem=stem, title="Main: Recall@10 vs QPS", method_order=MAIN_PLOT_METHOD_ORDER)
        plot_names.append(stem.stem)
        stem = out_root / "main_qps_vs_index_overhead_facets"
        _scatter_facets(comp_main_df, x="index_overhead_mb", y="qps", stem=stem, title="Main: QPS vs index overhead", method_order=MAIN_PLOT_METHOD_ORDER)
        plot_names.append(stem.stem)
        stem = out_root / "appendix_recall_compute_storage_triangle_facets"
        _tradeoff_triangle_facets(
            comp_main_df,
            stem=stem,
            method_order=MAIN_PLOT_METHOD_ORDER,
            highlight_pairs=[
                ("AdaptIVF", "AdaptIVF-m80"),
                ("AdaptIVF+PQ", "AdaptIVF+PQ-m80"),
            ],
        )
        plot_names.append(stem.stem)
        stem = out_root / "main_recall_compute_storage_triangle_aggregate"
        _tradeoff_triangle_aggregate(
            comp_main_df,
            stem=stem,
            method_order=MAIN_PLOT_METHOD_ORDER,
            highlight_pairs=[
                ("AdaptIVF", "AdaptIVF-m80"),
                ("AdaptIVF+PQ", "AdaptIVF+PQ-m80"),
            ],
        )
        plot_names.append(stem.stem)
        stem = out_root / "appendix_recall_vs_index_overhead_facets"
        _scatter_facets(comp_main_df, x="index_overhead_mb", y="recall_at_10", stem=stem, title="Main: Recall@10 vs index overhead", method_order=MAIN_PLOT_METHOD_ORDER)
        plot_names.append(stem.stem)
        load_balance_df = _load_balance_rows(_restrict_methods(comp_raw, MAIN_PLOT_METHOD_ORDER))
        stem = out_root / "main_load_balance_bucket_rank_facets"
        _load_balance_plot(load_balance_df, stem)
        if not load_balance_df.empty:
            plot_names.append(stem.stem)

    if not ablation_df.empty:
        stem = out_root / "appendix_ablation_recall_vs_avg_computations_facets"
        _ablation_plot(ablation_df, x="avg_computations", stem=stem, title="AdaptIVF ablations: Recall@10 vs avg computations")
        plot_names.append(stem.stem)
        stem = out_root / "appendix_ablation_recall_vs_index_overhead_facets"
        _ablation_plot(ablation_df, x="index_overhead_mb", stem=stem, title="AdaptIVF ablations: Recall@10 vs index overhead")
        plot_names.append(stem.stem)

    # ---- Appendix plots ----

    # Main: Training time grouped bars (paper learned methods only)
    learned_methods = [m for m in MAIN_PLOT_METHOD_ORDER if m not in ("HNSW", "IVF", "IVFPQ")]
    if not comp_df.empty and "train_s" in comp_df.columns:
        train_df = comp_df.dropna(subset=["train_s"])
        if not train_df.empty:
            stem = out_root / "main_training_time_bars"
            _grouped_bar_facets(
                train_df,
                y="train_s",
                stem=stem,
                title="Main: Training time (s) by method",
                method_order=learned_methods,
                ylabel="train_s (seconds)",
                ncols=2,
                fixed_method_axis=True,
            )
            plot_names.append(stem.stem)

    # A2: Index overhead grouped bars (all methods)
    if not comp_df.empty and "serving_footprint_mb" in comp_df.columns:
        stem = out_root / "appendix_serving_footprint_bars"
        _grouped_bar_facets(comp_df, y="serving_footprint_mb", stem=stem, title="Appendix: Index overhead excl. base vectors (MB)", method_order=MAIN_METHOD_ORDER, ylabel="serving_footprint_mb", log_y=True)
        plot_names.append(stem.stem)

    # A2b: query-RAM plots (all methods)
    if not comp_df.empty and "query_mem_delta_mb" in comp_df.columns:
        query_ram_df = comp_df.dropna(subset=["query_mem_delta_mb"])
        if not query_ram_df.empty:
            stem = out_root / "appendix_query_ram_vs_index_overhead_facets"
            _scatter_facets(
                query_ram_df,
                x="index_overhead_mb",
                y="query_mem_delta_mb",
                stem=stem,
                title="Appendix: Query RAM delta vs index overhead",
                method_order=MAIN_METHOD_ORDER,
                xlabel="Index Overhead (MB)",
                ylabel="Query RAM Delta (MB)",
                legend_single_row=False,
                legend_ncols_override=4,
            )
            plot_names.append(stem.stem)
            stem = out_root / "appendix_query_ram_bars"
            _grouped_bar_facets(
                query_ram_df,
                y="query_mem_delta_mb",
                stem=stem,
                title="Appendix: Query RAM delta by method",
                method_order=MAIN_METHOD_ORDER,
                ylabel="query RAM delta (MB)",
                log_y=True,
                ncols=2,
                fixed_method_axis=True,
            )
            plot_names.append(stem.stem)

    # A2c: serving-RAM plots (all methods with isolated RSS measurements)
    if not comp_df.empty and "rss_serving_mb" in comp_df.columns:
        serving_ram_df = comp_df.dropna(subset=["rss_serving_mb"])
        if not serving_ram_df.empty:
            stem = out_root / "appendix_serving_ram_vs_index_overhead_facets"
            _scatter_facets(
                serving_ram_df,
                x="index_overhead_mb",
                y="rss_serving_mb",
                stem=stem,
                title="Appendix: Serving RAM vs index overhead",
                method_order=MAIN_METHOD_ORDER,
                xlabel="Index Overhead (MB)",
                ylabel="Serving RAM (MB)",
                legend_single_row=False,
                legend_ncols_override=4,
            )
            plot_names.append(stem.stem)
            stem = out_root / "appendix_serving_ram_bars"
            _grouped_bar_facets(
                serving_ram_df,
                y="rss_serving_mb",
                stem=stem,
                title="Appendix: Serving RAM by method",
                method_order=MAIN_METHOD_ORDER,
                ylabel="serving RAM (MB)",
                log_y=True,
                ncols=2,
                fixed_method_axis=True,
            )
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

# Measurement Contract

All methods emit one shared metric surface. The goal is to keep fairness in the
artifacts themselves rather than in ad hoc plotting code.

## Retrieval Quality

- `recall_at_10` is computed against exact ground-truth neighbors for the same
  evaluation query set.
- Query subsampling, when used, is dataset-scoped and shared across methods.

## Workload

- `avg_computations` is the canonical workload metric.
- For `HNSW`, it means average distance computations per query.
- For IVF-style and router-based methods, it means average unique candidate
  vectors scored per query.
- The emitted `avg_computations_mode` field states which accounting mode was
  used.
- `avg_candidates` may still appear as a compatibility alias in intermediate
  outputs, but the canonical exports and paper plots should use
  `avg_computations`.

## Storage

- `index_overhead_mb` is the primary compactness metric.
- It comes from `index_manifest.json`.
- It counts method/index machinery only:
  - model weights
  - lookup tables
  - centroids
  - offsets
  - PQ codes and codebooks
  - graph/index structures
- It explicitly excludes shared staged base vectors such as `train.npy` and
  `index.npy`.
- `index_size_mb` is retained as a compatibility alias pointing to the same
  quantity.

## Serving Footprint

- `serving_footprint_mb` is a secondary systems metric.
- It also comes from `index_manifest.json`.
- It counts the serving-time payload needed by the realized query path.
- When a method needs staged base vectors for exact reranking, those bytes count
  toward the serving footprint.

## Query RAM

- `query_mem_delta_mb` is the canonical per-query RAM metric.
- It should prefer an isolated subprocess measurement.
- The subprocess loads only the serving artifacts needed for querying, executes
  the query path, and reports the RSS delta.
- The emitted `query_mem_delta_mode` field must state whether the measurement
  was:
  - `isolated_subprocess_rss_delta`, or
  - `in_process_rss_delta`

This is the metric used by the export bundle’s query-RAM appendix plots.

## Serving RAM

- `serving_ram.json` and `serving_ram.csv` are generated during export
  finalization.
- Each row measures a fresh subprocess that loads only the serving artifacts and
  reports:
  - `rss_baseline_mb`
  - `rss_serving_mb`
  - `rss_peak_mb`
  - `method_overhead_mb`
- `rss_serving_mb` is the isolated post-load serving RSS.
- `method_overhead_mb` is `rss_serving_mb - rss_baseline_mb`.

These measurements are distinct from `query_mem_delta_mb`: serving RAM captures
the resident serving process after loading the artifacts, whereas query RAM
captures the additional RSS incurred by executing the query path.

## Training Time

- `train_s` is the recorded training or learned-index construction time for
  learned methods.
- For classical FAISS baselines, comparable wall-clock build values may appear
  as `build_total_s` where available.
- Training-time bar charts in the paper bundle are therefore primarily intended
  for learned-method comparisons, not as a universal build-time benchmark.

## Canonical Export Surface

The canonical export bundle under `paper_exports/` now includes:

- per-study summary CSV/JSON files
- `tables/*.csv`
- top-level plot CSV/PDF/PNG triplets
- `serving_ram.json` and `serving_ram.csv`
- `plots_manifest.json`, `tables_manifest.json`, and `study_manifest.json`

## Paper-Facing Naming

The finished export bundle now uses a strict paper-facing naming rule:

- `main_*` artifacts are the plots and tables used directly by the manuscript
- `appendix_*` artifacts are supporting figures kept for supplementary material
- study-local raw summaries remain under `initialization/`, `competitiveness/`,
  and `ablations/`

This keeps the tracked bundle legible for reviewers while preserving the full
numeric surface for reproduction.

To keep the tracked export bundle publishable, the canonical summaries and
manifests store artifact provenance as repo-relative paths instead of
machine-local absolute paths.

The finished bundle is described for end users in [`paper_exports/README.md`](../paper_exports/README.md).

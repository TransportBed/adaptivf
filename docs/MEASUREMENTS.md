# Measurement Contract

All methods should emit the same system metrics under one shared contract.

## Recall

- `recall_at_10` is computed against exact ground-truth neighbors for the same query set.
- Query subsampling, if any, is dataset-scoped and shared across methods.

## Average Computations

- `avg_computations` is the canonical workload metric.
- For HNSW it means `avg_distance_computations_per_query`: FAISS `hnsw_stats.ndis / n_queries`.
- For IVF-style and router-based methods it means `avg_unique_candidate_vectors_per_query`.
- The emitted `avg_computations_mode` field must always state which accounting mode was used.
- A compatibility alias `avg_candidates` may still be emitted in intermediate CSV/JSON artifacts, but export code should use `avg_computations`.

## Index Overhead

- `index_overhead_mb` is the primary size metric.
- It must come from `index_manifest.json`.
- It counts method/index machinery only:
  - model weights
  - lookup tables
  - centroids
  - offsets
  - PQ codes/codebooks
  - graph/index structures
- It explicitly excludes shared staged base vectors such as `train.npy` and `index.npy`.
- `index_size_mb` is retained as a compatibility alias that points to `index_overhead_mb`.

## Serving Footprint

- `serving_footprint_mb` is still exported as a secondary systems metric.
- It comes from `index_manifest.json`.
- The manifest counts serving-time artifacts only, not logs, plots, or tuning/report files.
- When a method needs staged base vectors (`train.npy` or `index.npy`) to serve exact reranking, those bytes count toward the serving footprint.
- Artifact counting is component-based:
  - `serving_staging`
  - `lookups`
  - `models`
  - `lira_inner_indexes`
  - `lira_data`

## Query Memory

- `query_mem_delta_mb` should prefer an isolated subprocess measurement.
- The subprocess should load only the query data needed for serving, then load the serving artifacts, execute the query path, and report RSS delta.
- The emitted `query_mem_delta_mode` field must state whether the measurement is:
  - `isolated_subprocess_rss_delta`, or
  - `in_process_rss_delta`

The goal is to make fairness explicit in the artifact schema, not implicit in plotting code.

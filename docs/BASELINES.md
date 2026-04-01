# Baseline Provenance

This repository keeps only the baseline surface required for the finished
AdaptIVF benchmark suite and paper exports.

## Competitiveness Baselines

The finalized competitiveness study evaluates four baseline families.

### Classical ANN Baselines

- `HNSW`: high-recall graph baseline and serving-speed anchor
- `IVF`: static coarse quantization baseline
- `IVFPQ`: compressed FAISS baseline and QPS anchor

These methods provide the non-learned reference points for recall, throughput,
and storage.

### Learned Partitioning / Learned Routing Baselines

- `BLISS`: learned partitioning with repeated hash-seeded partitions
- `BLISS-KMeans`: geometry-aware BLISS variant used in the initialization study
- `MLP-IVF`: BLISS-style learned routing with IVF initialization
- `MLP-IVFPQ`: compressed variant of `MLP-IVF`

These methods isolate the effect of initialization and repeated learned
partitioning relative to the single-backbone AdaptIVF family.

### Adaptive Learned Baseline

- `LIRA`: learned partition selection with local inner indexes and thresholded
  probing

In the finished run, `LIRA` remains the compute-efficient learned baseline, but
also the structurally heaviest one.

### AdaptIVF Family

- `AdaptIVF`
- `AdaptIVF+PQ`
- `AdaptIVF-m80`
- `AdaptIVF+PQ-m80`
- ablation-only internal baselines:
  - `AdaptIVF-Static`
  - `AdaptIVF-A4`
  - `AdaptIVF-Static+PQ`
  - `AdaptIVF-A4+PQ`

Only `AdaptIVF` and `AdaptIVF+PQ` are public API methods. The `m80`, `Static`,
and `A4` variants are benchmark-only analysis settings.

## Implementation Strategy

Prefer published packages for generic primitives:

- `faiss-cpu` for IVF, IVFPQ, HNSW, centroid search, and PQ scanning
- `numpy` and `h5py` for dataset IO and dense-vector handling
- `scikit-learn`, `matplotlib`, and `pandas` for auxiliary statistics and
  export generation

Use thin local ports only where the published method does not map cleanly to a
single stable package interface.

## Local Ports

### BLISS

Reference implementation during porting: [`../BLISS_1`](../BLISS_1)

Retained:

- seeded / KMeans initialization used by the benchmark suite
- repeated router training
- iterative reassignment
- final materialization and query aggregation

Dropped:

- XML / repository-specific benchmark machinery
- unrelated configuration layers

### LIRA

Reference implementation during porting: [`../LIRA-ANN-search`](../LIRA-ANN-search)

Retained:

- trained probing model
- adaptive multi-assignment
- thresholded query probing
- partition-local inner-index wrapper

Dropped:

- unrelated large-scale scripts
- original repository-specific dataset layout

## Interpretation After The Finished Run

The final run clarifies the role of each baseline:

- `HNSW` and `IVFPQ` are the throughput anchors.
- `BLISS` remains strong on `GIST` and `Deep1M`, especially when the coarse
  geometry is weak.
- `MLP-IVF` is the strongest geometry-aware learned-routing baseline on
  `SIFT`.
- `LIRA` is the strongest learned compute-efficiency baseline, but only with a
  large training and storage cost.
- the strongest AdaptIVF results currently come from the `m80` analysis
  variants, which expose the upside of the same compact single-backbone design
  under a less restrictive probe cap.

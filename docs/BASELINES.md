# Baseline Provenance

This repository keeps only the baseline surface needed for the AdaptIVF benchmark suite.

## Published Packages

Prefer published packages for stable, generic functionality:

- `faiss-cpu`: IVF, IVFPQ, HNSW, centroid search, PQ scan
- `numpy`, `h5py`: dataset IO and dense vector handling
- `scikit-learn`: small auxiliary KMeans or statistics where FAISS is not required
- `matplotlib`, `pandas`: tables and figures

## Local Ports

Two baselines still need narrow local ports because their published methods do not map cleanly to a single package API.

### BLISS

Reference implementation during porting: [`../BLISS_1`](../BLISS_1)

Keep only:

- seeded / KMeans initialization used by the benchmark suite
- repeated router training
- final materialization
- query aggregation path

Drop:

- unrelated XML / broader benchmark machinery
- repo-specific config indirection

### LIRA

Reference implementation during porting: [`../LIRA-ANN-search`](../LIRA-ANN-search)

Keep only:

- small-scale LIRA path used in the comparison suite
- trained probing model
- adaptive multi-assignment
- thresholded query probing
- partition-local index wrapper

Drop:

- unrelated large-scale scripts unless they are needed later
- repo-specific dataset layout

## AdaptIVF

Reference implementation during porting: current AdaptIVF development branch of [`../lindex`](../lindex)

Keep only:

- IVF backbone setup
- router training
- uncertainty-aware placement
- entropy-adaptive query probing
- no-PQ and +PQ serving paths

## Result

The new repository should use:

- published packages for generic ANN primitives
- thin local ports only for benchmark-critical baseline behavior
- no runtime dependence on sibling research repositories
- no generic research-framework abstraction above those components

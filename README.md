# AdaptIVF

Reference implementation and evaluation suite for AdaptIVF.

## Overview

This repository serves two purposes:

1. run the benchmark pipeline, tables, and plots from a compact command surface;
2. provide an installable Python interface for `AdaptIVF` and `AdaptIVF+PQ`.

The benchmark surface in this repository covers the datasets `glove`, `sift`, `gist`, and `deep1m`, with generated `glove10k` data used only for smoke testing. The implemented methods are `BLISS`, `BLISS-KMeans`, `MLP-IVF`, `IVF`, `IVFPQ`, `MLP-IVFPQ`, `HNSW`, `LIRA`, `AdaptIVF`, and `AdaptIVF+PQ`, plus the competitiveness-analysis variants `AdaptIVF-m80` and `AdaptIVF+PQ-m80`. The evaluation pipeline is organized into three study groups: initialization, main competitiveness, and AdaptIVF ablations. Generated exports are written to `paper_exports/`; the canonical artifact layout, rebuild steps, and exported query-RAM / serving-RAM figures are documented in [`paper_exports/README.md`](paper_exports/README.md).

## Command Surface

All experiment scripts live under [`scripts/`](scripts/). See [`scripts/README.md`](scripts/README.md) for detailed usage and flags.

| Command | What it does |
|---|---|
| `bash scripts/download_datasets.sh` | Download benchmark datasets (`glove`, `sift`, `gist`, `deep1m`) |
| `bash scripts/download_datasets.sh glove10k` | Generate the `glove10k` smoke dataset from downloaded `glove` data |
| `bash scripts/run_paper.sh` | Run the full evaluation pipeline sequentially and regenerate `paper_exports/` |
| `bash scripts/run_paper.sh --no-tmux` | Run the same pipeline in the foreground |
| `bash scripts/run_paper.sh --competitiveness-only` | Run only the main comparison study |
| `bash scripts/run_paper.sh --ablations-only` | Run only the AdaptIVF ablation study |
| `bash scripts/run_paper.sh --plan` | Rebuild tables and plots from finished runs without rerunning experiments |
| `bash scripts/smoke_glove10k.sh` | Run the lightweight smoke path on `glove10k` |
| `bash scripts/test.sh -q` | Run the AdaptIVF test suite |

## Method Diagram

```text
vectors + queries
|
+-- BLISS
|   coarse init (random assignment using murmur3 hash)
|     -> repeated router MLPs
|     -> iterative reassignment
|     -> repeated global lookups
|     -> fixed probe budget
|
+-- LIRA
|   k-means partitions
|     -> probe MLP on vector + centroid-distance features
|     -> model-driven redundancy / repartitioning
|     -> local inner indexes (FLAT / HNSW)
|     -> threshold-selected partitions
|
+-- AdaptIVF / AdaptIVF+PQ
    semantic IVF coarse partitions
      -> neighbor-bucket router MLP
      -> confidence-based multi-assignment
      -> entropy-adaptive probe budget m(q)
      -> single IVF backbone retrieval
      -> optional PQ compression
```

| Method | Initialization | Learned signal | Index-time effect | Query-time effect | Serving structure |
|---|---|---|---|---|---|
| `BLISS` / `MLP-IVF` | hash / k-means / IVF | repeated router MLPs | iterative reassignment | fixed probe budget | repeated global lookups |
| `LIRA` | k-means partitions | probe MLP over vector and centroid-distance features | model-driven redundancy and repartitioning | threshold-selected partitions | partition-local inner indexes |
| `AdaptIVF` / `AdaptIVF+PQ` | semantic IVF partitions | neighbor-bucket router MLP | confidence-based selective duplication | entropy-adaptive `m(q)` | single IVF backbone, optional PQ |

At a high level:

- `BLISS` learns partitioning through repeated routers and iterative reassignment, then serves through repeated global lookups under a fixed probe budget.
- `LIRA` learns a probing model, uses it to repartition redundantly, and serves through threshold-selected local indexes inside each partition.
- `AdaptIVF` keeps a single semantic IVF backbone, uses confidence for selective duplication at index time, and uses entropy for per-query adaptive probing at search time. `AdaptIVF+PQ` keeps the same routing logic but swaps the final scoring path to IVF-PQ style compressed retrieval.

## Layout

```text
adaptivf/
  pyproject.toml                  # package metadata & dependencies
  docs/
    BASELINES.md                  # baseline provenance notes
    FINDINGS.md                   # experimental findings (living doc)
    MEASUREMENTS.md               # shared measurement contract
    NEXT_DIRECTIONS.md            # planned follow-up work
  scripts/                        # all runnable scripts (see scripts/README.md)
    run_paper.sh                  # full paper pipeline
    download_datasets.sh          # dataset download & generation
    smoke_glove10k.sh             # lightweight smoke test
    measure_serving_ram.py        # cold-load RAM measurement
    test.sh                       # test runner
  src/
    adaptivf.py                   # public API (AdaptIVFConfig, make_adaptivf)
    datasets.py                   # dataset registry & loading
    methods/                      # all method implementations
      adaptivf.py                 # AdaptIVF / AdaptIVF+PQ adapters
      router_family.py            # core router factory & RouterConfig
      faiss_baselines.py          # IVF, IVFPQ, HNSW
      bliss.py                    # BLISS / MLP-IVF
      lira.py                     # LIRA
    studies/                      # experiment orchestration
      competitiveness.py          # main comparison study
      ablation.py                 # AdaptIVF ablation study
      initialization.py           # routing efficiency study
    cli_run_competitiveness.py    # CLI: competitiveness experiments
    cli_run_faiss_baselines.py    # CLI: FAISS baseline experiments
    cli_run_plan.py               # CLI: tables & plots from results
    cli_run_router_methods.py     # CLI: learned-routing experiments
    collect.py                    # result collection & merging
    plots.py / tables.py          # figure & table generation
  tests/                          # pytest suite
  paper_exports/                  # canonical generated export bundle
    README.md                     # artifact index & reproducibility notes
```

## Dependency Strategy

Use published packages wherever possible:

- `faiss-cpu` for IVF / IVFPQ / HNSW and shared ANN primitives
- `numpy`, `h5py`, `scikit-learn`, `matplotlib`, `pandas`, `PyYAML`
- `tensorflow` only where BLISS / MLP-IVF / AdaptIVF parity requires it
- `torch` only where LIRA parity requires it

All benchmark methods run locally inside this repository. Provenance notes for the ports are recorded in [docs/BASELINES.md](docs/BASELINES.md). The shared metric rules are in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

## Quick Start

List datasets and metadata:

```bash
PYTHONPATH=src python -m datasets --list
```

Download the benchmark datasets:

```bash
bash scripts/download_datasets.sh
```

Generate the smoke dataset:

```bash
bash scripts/download_datasets.sh glove10k
```

Default run surface:

```bash
bash scripts/run_paper.sh
```

This is the intended default command: all benchmark datasets, all benchmark methods, and the AdaptIVF ablation study, sequential in tmux, with cleanup of prior generated artifacts. The run regenerates `paper_exports/` at the end, including normalized study summaries, isolated query-RAM and serving-RAM measurements, plots, tables, and the export-bundle README. Re-running the pipeline is expected to recreate the same export layout and file set, but not necessarily byte-identical plots or identical runtime-sensitive values such as QPS and RAM.

For repeated seeds:

```bash
bash scripts/run_paper.sh --seeds 0,1,2
```

For foreground execution without tmux:

```bash
bash scripts/run_paper.sh --no-tmux
```

Run the initialization study only:

```bash
bash scripts/run_paper.sh --initialization-only
```

Run the main competitiveness study only:

```bash
bash scripts/run_paper.sh --competitiveness-only
```

Run only the AdaptIVF ablations:

```bash
bash scripts/run_paper.sh --ablations-only
```

Only `AdaptIVF` and `AdaptIVF+PQ` are first-class AdaptIVF methods in this repository. The `Static` and `A4` variants are kept only as ablation-only internal baselines and are not part of the public method surface or packaged API.

For the finished paper artifact bundle and the canonical post-run export layout, see [`paper_exports/README.md`](paper_exports/README.md).

## Measurement Contract

The primary size and system metrics use:

- `index_overhead_mb`: method/index overhead only
  - model weights
  - lookup tables
  - centroids
  - offsets
  - PQ codes/codebooks
  - graph/index structures
  - excludes shared base vectors
- `serving_footprint_mb`: full serving payload, including shared base vectors if required by the serving path
- `avg_computations`: shared workload metric
  - HNSW: average distance computations per query
  - IVF/router methods: average unique candidate vectors scored per query

`index_overhead_mb` is the primary compactness metric. `serving_footprint_mb` is exported as a secondary systems metric.

## Python API

The package can also be installed directly:

```bash
pip install -e .
```

Then import the methods, for example:

```python
from adaptivf import AdaptIVFConfig, make_adaptivf

index = make_adaptivf(
    "glove",
    config=AdaptIVFConfig(
        confidence_threshold=0.75,
        max_assignments=3,
        m_base=5,
        m_max=10,
        entropy_scale=10.0,
    ),
)
```

The packaged API is intentionally thin and user-facing. For lower-level control, use the config objects in [`src/methods/router_family.py`](src/methods/router_family.py) and [`src/methods/lira_runtime.py`](src/methods/lira_runtime.py).

## Repository Principles

- no generic resolver
- no run-id string dispatch
- no monolithic YAML matrix
- no plot regeneration tied to a full rerun
- all artifact generation should be dataset-scoped and incremental
- the default size metric should measure index overhead, not shared vector payload

Baseline provenance is documented in [docs/BASELINES.md](docs/BASELINES.md). The shared measurement contract is in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md). Final run findings and follow-up priorities are summarized in [docs/FINDINGS.md](docs/FINDINGS.md) and [docs/NEXT_DIRECTIONS.md](docs/NEXT_DIRECTIONS.md).

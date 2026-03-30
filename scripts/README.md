# Scripts

All runnable scripts for dataset preparation, experiments, measurement, and testing.

## Dataset Download

```bash
bash scripts/download_datasets.sh                # download glove, sift, gist, deep1m
bash scripts/download_datasets.sh glove10k        # generate glove10k smoke dataset from glove
bash scripts/download_datasets.sh glove sift      # download specific datasets only
```

Datasets are written to `../data/` relative to the repository root (configurable via `DATA_ROOT`).

## Full Paper Pipeline

```bash
bash scripts/run_paper.sh
```

Runs the complete evaluation pipeline in tmux:

1. **Initialization study** — routing efficiency across probe depths (BLISS, BLISS-KMeans, MLP-IVF)
2. **Competitiveness study** — all 9 methods × all datasets (HNSW, IVF, IVFPQ, MLP-IVF, BLISS, MLP-IVFPQ, LIRA, AdaptIVF, AdaptIVF+PQ, plus m_max=80 analysis variants)
3. **Ablation study** — AdaptIVF variants (Static, A4, with/without PQ)
4. **Plan** — regenerate tables and plots in `paper_exports/`

### Flags

| Flag | Effect |
|---|---|
| `--no-tmux` | Run in the foreground instead of a tmux session |
| `--no-clean` | Skip cleanup of prior `paper_exports/` |
| `--seeds 0,1,2` | Repeat all experiments for multiple seeds |
| `--datasets glove,sift` | Restrict to specific datasets |
| `--initialization-only` | Run only the initialization study |
| `--competitiveness-only` | Run only the main comparison study |
| `--ablations-only` | Run only the AdaptIVF ablation study |
| `--plan` | Rebuild tables and plots from finished runs without rerunning experiments |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `../data` | Directory containing dataset HDF5 files |
| `EXPERIMENTS_ROOT` | `../experiments/adaptivf` | Directory for experiment artifacts |
| `OUT_ROOT` | `paper_exports` | Output directory for tables, plots, and summaries |
| `PYTHON_BIN` | `python3` | Python interpreter to use |

## Smoke Test

```bash
bash scripts/smoke_glove10k.sh
```

Runs a fast end-to-end check on the tiny `glove10k` dataset. Useful for verifying the pipeline works before committing to a full run.

## Serving RAM Measurement

```bash
PYTHONPATH=src python scripts/measure_serving_ram.py \
  --experiments-root ../experiments/adaptivf \
  --data-root ../data
```

Measures cold-load serving RSS for every completed experiment. Spawns an isolated subprocess per experiment that loads only the artifacts needed for query serving, reads `/proc/self/status` for VmRSS, and exits. No queries are executed and no artifacts are written to experiment directories. Output is JSON lines to stdout.

Optional `--output path.json` writes a combined JSON array to disk.

## Patch Rerun (Ephemeral)

```bash
bash scripts/rerun_adaptivf_patch.sh
```

Reruns only AdaptIVF and AdaptIVF+PQ competitiveness methods with the current `m_max=10` default. Results are written to `paper_exports/patch_m10/` to avoid overwriting existing m_max=80 results. Safe to delete after use.

## Test Suite

```bash
bash scripts/test.sh          # full test run
bash scripts/test.sh -q       # quiet mode
bash scripts/test.sh -x       # stop on first failure
```

Runs the pytest suite under `tests/` with `PYTHONPATH=src`.

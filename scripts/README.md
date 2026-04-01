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

Runs the complete evaluation pipeline in tmux. By default it now starts from a
clean canonical export tree and regenerates `paper_exports/` in place. This is
expected to recreate the same export layout and filenames, but not necessarily
byte-identical figures or identical runtime-sensitive measurements such as QPS
and RAM.

1. **Initialization study** — routing efficiency across probe depths (BLISS, BLISS-KMeans, MLP-IVF)
2. **Competitiveness study** — the 9 core comparison methods across all datasets, plus the `m_max=80` AdaptIVF analysis variants
3. **Ablation study** — AdaptIVF variants (Static, A4, with/without PQ)
4. **Finalize exports** — rebuild normalized summaries, measure serving RAM for canonical competitiveness experiments, regenerate tables/plots including the RAM figures, and write the export-bundle README in `paper_exports/`

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
| `OUT_ROOT` | `paper_exports` | Canonical output directory for tables, plots, and summaries |
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

The full paper pipeline now invokes this measurement during its export-finalize
phase and writes canonical filtered outputs to `paper_exports/serving_ram.json`
and `paper_exports/serving_ram.csv`. The plotting step consumes these
measurements to emit the serving-RAM appendix plots alongside the query-RAM
plots derived from `query_mem_delta_mb`.

## Export-Only Rebuild

If the experiments have already completed and only the canonical export bundle
needs to be refreshed:

```bash
PYTHONPATH=src python -m collect --out-root paper_exports --experiments-root ../experiments/adaptivf --data-root ../data
PYTHONPATH=src python -m tables --out-root paper_exports
PYTHONPATH=src python -m plots --out-root paper_exports
PYTHONPATH=src python -m export_bundle --out-root paper_exports
```

## Targeted AdaptIVF Rerun

```bash
bash scripts/run_paper.sh --competitiveness-only --datasets glove,sift,gist,deep1m --m-max 10
bash scripts/run_paper.sh --ablations-only --datasets glove,sift,gist,deep1m --m-max 10
```

Reruns the AdaptIVF family with an explicit `m_max` override through the main
paper pipeline instead of a separate patch helper. Pair this with a custom
`OUT_ROOT` if you want to keep the results separate from the canonical
`paper_exports/` tree.

## Test Suite

```bash
bash scripts/test.sh          # full test run
bash scripts/test.sh -q       # quiet mode
bash scripts/test.sh -x       # stop on first failure
```

Runs the pytest suite under `tests/` with `PYTHONPATH=src`.

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHARED_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-${SHARED_ROOT}/data}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${SHARED_ROOT}/experiments/adaptivf}"

bash "${REPO_ROOT}/scripts/download_datasets.sh" glove glove10k
PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_faiss_baselines \
  --datasets glove10k \
  --data-root "${DATA_ROOT}" \
  --experiments-root "${EXPERIMENTS_ROOT}" \
  --exports-root "${REPO_ROOT}/paper_exports"
PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_router_methods \
  --datasets glove10k \
  --methods BLISS,BLISS-KMeans,MLP-IVF \
  --probes 5,10,20,40 \
  --data-root "${DATA_ROOT}" \
  --experiments-root "${EXPERIMENTS_ROOT}" \
  --exports-root "${REPO_ROOT}/paper_exports"
PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_competitiveness \
  --datasets glove10k \
  --data-root "${DATA_ROOT}" \
  --experiments-root "${EXPERIMENTS_ROOT}" \
  --exports-root "${REPO_ROOT}/paper_exports"
PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_competitiveness \
  --datasets glove10k \
  --methods AdaptIVF-Static,AdaptIVF,AdaptIVF-A4,AdaptIVF-Static+PQ,AdaptIVF+PQ,AdaptIVF-A4+PQ \
  --export-subdir ablations \
  --data-root "${DATA_ROOT}" \
  --experiments-root "${EXPERIMENTS_ROOT}" \
  --exports-root "${REPO_ROOT}/paper_exports"
bash "${REPO_ROOT}/scripts/run_paper.sh" --datasets glove10k --plan --no-clean

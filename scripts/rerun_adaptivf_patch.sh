#!/usr/bin/env bash
# ============================================================================
# EPHEMERAL PATCH SCRIPT — safe to delete after use
# ============================================================================
# Reruns ONLY AdaptIVF competitiveness methods (AdaptIVF, AdaptIVF+PQ) for
# datasets that were computed with the old m_max=80 default.
#
# The main codebase now defaults to m_max=10 (the published algorithm value).
# This script exists solely because we cannot afford a full rerun; it patches
# just the AdaptIVF rows.
#
# IMPORTANT: Results are written to paper_exports/patch_m10/ to avoid
# overwriting the existing m_max=80 results in paper_exports/competitiveness/.
# Both sets of results are preserved for analysis.
#
# Usage (after the current run_paper.sh pipeline finishes):
#   bash scripts/rerun_adaptivf_patch.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHARED_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/paper_exports/patch_m10}"
DATA_ROOT="${DATA_ROOT:-${SHARED_ROOT}/data}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${SHARED_ROOT}/experiments/adaptivf}"
SEEDS="${SEEDS:-0}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"

export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS="${THREADS}"
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMBA_NUM_THREADS="${THREADS}"
export TF_NUM_INTRAOP_THREADS="${THREADS}"
export TF_NUM_INTEROP_THREADS=2

# Only the AdaptIVF competitiveness methods need patching.
PATCH_METHODS="AdaptIVF,AdaptIVF+PQ"
DATASETS="glove,sift,gist,deep1m"

LOG_ROOT="${REPO_ROOT}/logs/rerun_adaptivf_patch_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_ROOT}"

echo "[patch] Rerunning AdaptIVF competitiveness with m_max=10 (source default)"
echo "[patch] datasets: ${DATASETS}"
echo "[patch] methods:  ${PATCH_METHODS}"
echo "[patch] logs:     ${LOG_ROOT}"

OLD_IFS="${IFS}"
IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
IFS="${OLD_IFS}"

RESULTS_CSV="${LOG_ROOT}/results.csv"
echo "dataset,status,log_path" > "${RESULTS_CSV}"

for dataset in "${DATASET_LIST[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [[ -z "${dataset}" ]] && continue
  LOG_PATH="${LOG_ROOT}/${dataset}_adaptivf_patch.log"
  echo "[patch] dataset=${dataset} threads=${THREADS}" | tee -a "${LOG_ROOT}/session.log"
  if PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_competitiveness \
    --datasets "${dataset}" \
    --methods "${PATCH_METHODS}" \
    --seeds "${SEEDS}" \
    --data-root "${DATA_ROOT}" \
    --experiments-root "${EXPERIMENTS_ROOT}" \
    --exports-root "${OUT_ROOT}" > >(tee "${LOG_PATH}") 2>&1; then
    echo "${dataset},ok,${LOG_PATH}" >> "${RESULTS_CSV}"
  else
    echo "${dataset},failed,${LOG_PATH}" >> "${RESULTS_CSV}"
    echo "[patch] FAILED: dataset=${dataset}" | tee -a "${LOG_ROOT}/session.log" >&2
    exit 1
  fi
done

echo "[patch] All AdaptIVF m_max=10 results written to ${OUT_ROOT}"
echo "[patch] Original m_max=80 results in paper_exports/competitiveness/ are UNTOUCHED"
echo "[patch] Results merged into: ${OUT_ROOT}/competitiveness/"
echo "[patch] This script can now be deleted."

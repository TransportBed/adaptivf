#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHARED_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/paper_exports}"
DATA_ROOT="${DATA_ROOT:-${SHARED_ROOT}/data}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${SHARED_ROOT}/experiments/adaptivf}"
DATASETS="glove,sift,gist,deep1m"
METHODS="IVF,IVFPQ,HNSW"
COMP_METHODS="HNSW,IVF,MLP-IVF,BLISS,LIRA,AdaptIVF,IVFPQ,MLP-IVFPQ,AdaptIVF+PQ,AdaptIVF-m80,AdaptIVF+PQ-m80"
ABLATION_METHODS="AdaptIVF-Static,AdaptIVF,AdaptIVF-A4,AdaptIVF-Static+PQ,AdaptIVF+PQ,AdaptIVF-A4+PQ"
CLEAN=1
PLAN_ONLY=0
FAISS_ONLY=0
INITIALIZATION_ONLY=0
COMPETITIVENESS_ONLY=0
ABLATIONS_ONLY=0
NO_TMUX=0
INSIDE_TMUX=0
SESSION="${SESSION:-adaptivf_paper}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"
SEEDS="${SEEDS:-0}"
INIT_METHODS="BLISS,BLISS-KMeans,MLP-IVF"
INIT_PROBES="5,10,20,40"
M_MAX="${M_MAX:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS}}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS}}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS}}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-${THREADS}}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-${THREADS}}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --no-clean)
      CLEAN=0
      shift
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --experiments-root)
      EXPERIMENTS_ROOT="$2"
      shift 2
      ;;
    --plan)
      PLAN_ONLY=1
      shift
      ;;
    --faiss-only)
      FAISS_ONLY=1
      shift
      ;;
    --initialization-only)
      INITIALIZATION_ONLY=1
      shift
      ;;
    --competitiveness-only)
      COMPETITIVENESS_ONLY=1
      shift
      ;;
    --ablations-only)
      ABLATIONS_ONLY=1
      shift
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      export OMP_NUM_THREADS="${THREADS}"
      export MKL_NUM_THREADS="${THREADS}"
      export OPENBLAS_NUM_THREADS="${THREADS}"
      export NUMBA_NUM_THREADS="${THREADS}"
      export TF_NUM_INTRAOP_THREADS="${THREADS}"
      shift 2
      ;;
    --no-tmux)
      NO_TMUX=1
      shift
      ;;
    --m-max)
      M_MAX="$2"
      shift 2
      ;;
    --_inside-tmux)
      INSIDE_TMUX=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${OUT_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

M_MAX_FLAG=""
if [[ -n "${M_MAX}" ]]; then
  M_MAX_FLAG="--m-max ${M_MAX}"
fi

write_manifests() {
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_plan --datasets "${DATASETS}" --out-root "${OUT_ROOT}"
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m studies.initialization --datasets "${DATASETS}" --out-root "${OUT_ROOT}"
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m studies.competitiveness --datasets "${DATASETS}" --out-root "${OUT_ROOT}"
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m studies.ablation --datasets "${DATASETS}" --out-root "${OUT_ROOT}"
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m collect --out-root "${OUT_ROOT}"
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m tables --out-root "${OUT_ROOT}"
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m plots --out-root "${OUT_ROOT}"
}

clean_generated_artifacts() {
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_clean_artifacts \
    --out-root "${OUT_ROOT}" \
    --experiments-root "${EXPERIMENTS_ROOT}" \
    --logs-root "${REPO_ROOT}/logs"
}

spawn_tmux() {
  local inner_args="$1"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[run_paper] tmux session already exists: ${SESSION}" >&2
    exit 1
  fi
  tmux new-session -d -s "${SESSION}" \
    "cd '${REPO_ROOT}' && PYTHON_BIN='${PYTHON_BIN}' DATA_ROOT='${DATA_ROOT}' EXPERIMENTS_ROOT='${EXPERIMENTS_ROOT}' OUT_ROOT='${OUT_ROOT}' THREADS='${THREADS}' SEEDS='${SEEDS}' M_MAX='${M_MAX}' bash scripts/run_paper.sh ${inner_args} --datasets '${DATASETS}' --session '${SESSION}' --seeds '${SEEDS}' --no-tmux --_inside-tmux"
  echo "[run_paper] started tmux session: ${SESSION}"
  echo "[run_paper] attach with: tmux attach -t ${SESSION}"
  exit 0
}

if [[ "${FAISS_ONLY}" -eq 1 ]]; then
  if [[ "${CLEAN}" -eq 1 && "${INSIDE_TMUX}" -eq 0 ]]; then
    clean_generated_artifacts
  fi
  if [[ "${NO_TMUX}" -eq 0 && "${INSIDE_TMUX}" -eq 0 ]]; then
    spawn_tmux "--faiss-only --methods '${METHODS}' --no-clean"
  fi

  LOG_ROOT="${REPO_ROOT}/logs/${SESSION}_$(date -u +%Y%m%d-%H%M%S)"
  mkdir -p "${LOG_ROOT}"
  RESULTS_CSV="${LOG_ROOT}/results.csv"
  echo "dataset,method,status,log_path" > "${RESULTS_CSV}"

  OLD_IFS="${IFS}"
  IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
  IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
  IFS="${OLD_IFS}"

  for dataset in "${DATASET_LIST[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    [[ -z "${dataset}" ]] && continue
    for method in "${METHOD_LIST[@]}"; do
      method="$(echo "${method}" | xargs)"
      [[ -z "${method}" ]] && continue
      mkdir -p "${LOG_ROOT}"
      LOG_PATH="${LOG_ROOT}/${dataset}_${method}.log"
      echo "[run_paper] dataset=${dataset} method=${method} threads=${THREADS}" | tee -a "${LOG_ROOT}/session.log"
      if PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_faiss_baselines \
        --datasets "${dataset}" \
        --methods "${method}" \
        --seeds "${SEEDS}" \
        --data-root "${DATA_ROOT}" \
        --experiments-root "${EXPERIMENTS_ROOT}" \
        --exports-root "${OUT_ROOT}" > >(tee "${LOG_PATH}") 2>&1; then
        mkdir -p "$(dirname "${RESULTS_CSV}")"
        echo "${dataset},${method},ok,${LOG_PATH}" >> "${RESULTS_CSV}"
      else
        mkdir -p "$(dirname "${RESULTS_CSV}")"
        echo "${dataset},${method},failed,${LOG_PATH}" >> "${RESULTS_CSV}"
        echo "[run_paper] failed: dataset=${dataset} method=${method}" | tee -a "${LOG_ROOT}/session.log" >&2
        exit 1
      fi
    done
  done
  echo "[run_paper] sequential FAISS run complete"
  echo "[run_paper] logs: ${LOG_ROOT}"
  exit 0
fi

if [[ "${INITIALIZATION_ONLY}" -eq 1 ]]; then
  if [[ "${CLEAN}" -eq 1 && "${INSIDE_TMUX}" -eq 0 ]]; then
    clean_generated_artifacts
  fi
  if [[ "${NO_TMUX}" -eq 0 && "${INSIDE_TMUX}" -eq 0 ]]; then
    spawn_tmux "--initialization-only --no-clean"
  fi

  LOG_ROOT="${REPO_ROOT}/logs/${SESSION}_$(date -u +%Y%m%d-%H%M%S)"
  mkdir -p "${LOG_ROOT}"
  RESULTS_CSV="${LOG_ROOT}/results.csv"
  echo "dataset,status,log_path" > "${RESULTS_CSV}"

  OLD_IFS="${IFS}"
  IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
  IFS="${OLD_IFS}"

  for dataset in "${DATASET_LIST[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    [[ -z "${dataset}" ]] && continue
    mkdir -p "${LOG_ROOT}"
    LOG_PATH="${LOG_ROOT}/${dataset}_initialization.log"
    echo "[run_paper] initialization dataset=${dataset} threads=${THREADS}" | tee -a "${LOG_ROOT}/session.log"
    if PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_router_methods \
      --datasets "${dataset}" \
      --methods "${INIT_METHODS}" \
      --probes "${INIT_PROBES}" \
      --seeds "${SEEDS}" \
      --data-root "${DATA_ROOT}" \
      --experiments-root "${EXPERIMENTS_ROOT}" \
      --exports-root "${OUT_ROOT}" > >(tee "${LOG_PATH}") 2>&1; then
      echo "${dataset},ok,${LOG_PATH}" >> "${RESULTS_CSV}"
    else
      echo "${dataset},failed,${LOG_PATH}" >> "${RESULTS_CSV}"
      echo "[run_paper] failed: initialization dataset=${dataset}" | tee -a "${LOG_ROOT}/session.log" >&2
      exit 1
    fi
  done
  echo "[run_paper] sequential initialization run complete"
  echo "[run_paper] logs: ${LOG_ROOT}"
  exit 0
fi

if [[ "${COMPETITIVENESS_ONLY}" -eq 1 ]]; then
  if [[ "${CLEAN}" -eq 1 && "${INSIDE_TMUX}" -eq 0 ]]; then
    clean_generated_artifacts
  fi
  if [[ "${NO_TMUX}" -eq 0 && "${INSIDE_TMUX}" -eq 0 ]]; then
    spawn_tmux "--competitiveness-only --no-clean"
  fi

  LOG_ROOT="${REPO_ROOT}/logs/${SESSION}_$(date -u +%Y%m%d-%H%M%S)"
  mkdir -p "${LOG_ROOT}"
  RESULTS_CSV="${LOG_ROOT}/results.csv"
  echo "dataset,status,log_path" > "${RESULTS_CSV}"

  OLD_IFS="${IFS}"
  IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
  IFS="${OLD_IFS}"

  for dataset in "${DATASET_LIST[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    [[ -z "${dataset}" ]] && continue
    LOG_PATH="${LOG_ROOT}/${dataset}_competitiveness.log"
    echo "[run_paper] competitiveness dataset=${dataset} threads=${THREADS}" | tee -a "${LOG_ROOT}/session.log"
    if PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_competitiveness \
      --datasets "${dataset}" \
      --methods "${COMP_METHODS}" \
      --seeds "${SEEDS}" \
      --data-root "${DATA_ROOT}" \
      --experiments-root "${EXPERIMENTS_ROOT}" \
      --exports-root "${OUT_ROOT}" ${M_MAX_FLAG} > >(tee "${LOG_PATH}") 2>&1; then
      echo "${dataset},ok,${LOG_PATH}" >> "${RESULTS_CSV}"
    else
      echo "${dataset},failed,${LOG_PATH}" >> "${RESULTS_CSV}"
      echo "[run_paper] failed: competitiveness dataset=${dataset}" | tee -a "${LOG_ROOT}/session.log" >&2
      exit 1
    fi
  done
  echo "[run_paper] sequential competitiveness run complete"
  echo "[run_paper] logs: ${LOG_ROOT}"
  exit 0
fi

if [[ "${ABLATIONS_ONLY}" -eq 1 ]]; then
  if [[ "${CLEAN}" -eq 1 && "${INSIDE_TMUX}" -eq 0 ]]; then
    clean_generated_artifacts
  fi
  if [[ "${NO_TMUX}" -eq 0 && "${INSIDE_TMUX}" -eq 0 ]]; then
    spawn_tmux "--ablations-only --no-clean"
  fi

  LOG_ROOT="${REPO_ROOT}/logs/${SESSION}_$(date -u +%Y%m%d-%H%M%S)"
  mkdir -p "${LOG_ROOT}"
  RESULTS_CSV="${LOG_ROOT}/results.csv"
  echo "dataset,status,log_path" > "${RESULTS_CSV}"

  OLD_IFS="${IFS}"
  IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
  IFS="${OLD_IFS}"

  for dataset in "${DATASET_LIST[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    [[ -z "${dataset}" ]] && continue
    LOG_PATH="${LOG_ROOT}/${dataset}_ablations.log"
    echo "[run_paper] ablations dataset=${dataset} threads=${THREADS} seeds=${SEEDS}" | tee -a "${LOG_ROOT}/session.log"
    if PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m cli_run_competitiveness \
      --datasets "${dataset}" \
      --methods "${ABLATION_METHODS}" \
      --seeds "${SEEDS}" \
      --data-root "${DATA_ROOT}" \
      --experiments-root "${EXPERIMENTS_ROOT}" \
      --exports-root "${OUT_ROOT}" \
      --export-subdir "ablations" ${M_MAX_FLAG} > >(tee "${LOG_PATH}") 2>&1; then
      echo "${dataset},ok,${LOG_PATH}" >> "${RESULTS_CSV}"
    else
      echo "${dataset},failed,${LOG_PATH}" >> "${RESULTS_CSV}"
      echo "[run_paper] failed: ablations dataset=${dataset}" | tee -a "${LOG_ROOT}/session.log" >&2
      exit 1
    fi
  done
  echo "[run_paper] sequential ablation run complete"
  echo "[run_paper] logs: ${LOG_ROOT}"
  exit 0
fi

if [[ "${PLAN_ONLY}" -eq 1 ]]; then
  write_manifests
  echo "[run_paper] plan written under ${OUT_ROOT}"
  exit 0
fi

if [[ "${CLEAN}" -eq 1 && "${INSIDE_TMUX}" -eq 0 ]]; then
  clean_generated_artifacts
fi

if [[ "${NO_TMUX}" -eq 0 && "${INSIDE_TMUX}" -eq 0 ]]; then
  spawn_tmux "--no-clean"
fi

LOG_ROOT="${REPO_ROOT}/logs/${SESSION}_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_ROOT}"

echo "[run_paper] full paper run: initialization, competitiveness, ablations" | tee -a "${LOG_ROOT}/session.log"
PYTHON_BIN="${PYTHON_BIN}" OUT_ROOT="${OUT_ROOT}" DATA_ROOT="${DATA_ROOT}" EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT}" THREADS="${THREADS}" \
  SEEDS="${SEEDS}" bash "${REPO_ROOT}/scripts/run_paper.sh" --initialization-only --datasets "${DATASETS}" --seeds "${SEEDS}" --no-clean --no-tmux > >(tee "${LOG_ROOT}/initialization.log") 2>&1
PYTHON_BIN="${PYTHON_BIN}" OUT_ROOT="${OUT_ROOT}" DATA_ROOT="${DATA_ROOT}" EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT}" THREADS="${THREADS}" \
  SEEDS="${SEEDS}" M_MAX="${M_MAX}" bash "${REPO_ROOT}/scripts/run_paper.sh" --competitiveness-only --datasets "${DATASETS}" --seeds "${SEEDS}" --no-clean --no-tmux > >(tee "${LOG_ROOT}/competitiveness.log") 2>&1
PYTHON_BIN="${PYTHON_BIN}" OUT_ROOT="${OUT_ROOT}" DATA_ROOT="${DATA_ROOT}" EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT}" THREADS="${THREADS}" \
  SEEDS="${SEEDS}" M_MAX="${M_MAX}" bash "${REPO_ROOT}/scripts/run_paper.sh" --ablations-only --datasets "${DATASETS}" --seeds "${SEEDS}" --no-clean --no-tmux > >(tee "${LOG_ROOT}/ablations.log") 2>&1
write_manifests
echo "[run_paper] full run complete"
echo "[run_paper] logs: ${LOG_ROOT}"
exit 0

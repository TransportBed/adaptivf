#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHARED_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-${SHARED_ROOT}/data}"

if [[ $# -eq 0 ]]; then
  set -- glove sift gist deep1m
fi

args=("$@")
need_glove=false

for token in "${args[@]}"; do
  IFS=',' read -r -a parts <<< "${token}"
  for part in "${parts[@]}"; do
    key="$(printf '%s' "${part}" | tr '[:upper:]' '[:lower:]' | xargs)"
    if [[ "${key}" == "glove10k" ]]; then
      need_glove=true
    fi
  done
done

if [[ "${need_glove}" == true && ! -f "${DATA_ROOT}/glove/glove-100-angular.hdf5" ]]; then
  echo "[download_datasets] glove10k requires glove; downloading glove first"
  args=("glove" "${args[@]}")
fi

PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m datasets --download --data-root "${DATA_ROOT}" "${args[@]}"

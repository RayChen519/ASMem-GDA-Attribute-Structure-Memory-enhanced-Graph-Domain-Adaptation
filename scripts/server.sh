#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../code"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=2
exec ../.venv/bin/python "$@"

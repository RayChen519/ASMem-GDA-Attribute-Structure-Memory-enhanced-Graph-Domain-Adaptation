#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON:=python3.13}"
: "${TORCH_INDEX_URL:?Set the official PyTorch wheel index compatible with this server CUDA stack}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.0 --index-url "$TORCH_INDEX_URL"
.venv/bin/python -m pip install -r code/requirements-da.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > requirements-server-resolved.txt

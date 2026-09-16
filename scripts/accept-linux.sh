#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ACCEPT_ID:?Set a new acceptance ID}"
: "${DEVICE:=cuda:0}"
bash scripts/server.sh -m utils.runtime.preflight --device "$DEVICE" --output "artifacts/reports/integration/$ACCEPT_ID-preflight.json"
bash scripts/server.sh -m verification.gate.run --device "$DEVICE" --output "artifacts/reports/integration/$ACCEPT_ID" --runs "artifacts/runs/smoke/$ACCEPT_ID"

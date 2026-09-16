#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GATE:?Set GATE relative to code/ or absolute}"
: "${DEVICE:=cuda:0}"
manifest=artifacts/datasets/manifests/dataset_manifest.json
mkdir -p code/artifacts/experiments
printf '{}\n' > code/artifacts/experiments/default-overlay.json
for variant in B0 B1 B2 B3 B4 B5 B6; do
  for source in ACMv9 Citationv1 DBLPv7; do
    for target in ACMv9 Citationv1 DBLPv7; do
      [[ "$source" == "$target" ]] && continue
      for rate in 0.01 0.03 0.05; do
        bash scripts/server.sh -m experiments development-run --manifest "$manifest" --integration-report "$GATE" --config artifacts/experiments/default-overlay.json --variant "$variant" --source "$source" --target "$target" --rate "$rate" --device "$DEVICE" --resume
      done
    done
  done
done
bash scripts/server.sh -m experiments collect-development --root artifacts/runs/development --output artifacts/experiments/development-candidate.json --budget 1
bash scripts/server.sh -m experiments lock-development --candidate artifacts/experiments/development-candidate.json --integration-report "$GATE" --output artifacts/experiments/development-lock.json

#!/usr/bin/env bash
# Full Experiment 1. Use the existing Colab Python/Torch/CUDA environment.
set -Eeuo pipefail
trap 'printf "ERROR: Experiment 1 stopped at line %s (exit %s). See the last stage error.\n" "$LINENO" "$?" >&2' ERR
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${KSRF_OUTPUT_DIR:-$ROOT/outputs/$(date -u +%Y%m%dT%H%M%SZ)}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# No pip/conda/apt, no virtual environment, no changes to Torch or CUDA.
"$PYTHON" -m ksrf_exp1 all --config "$ROOT/config.yaml" --output "$OUTPUT_DIR"
printf '\nFull Experiment 1 finished. Report: %s/EXP1_REPORT.md\n' "$OUTPUT_DIR"

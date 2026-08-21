#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/data_base}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config.yaml}"

cd "$SCRIPT_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python and run again." >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset directory not found: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 2
fi

"$PYTHON" -c 'import torch, scipy, yaml, numpy; print("Python dependencies: OK")'
"$PYTHON" -c 'import common; from mmfi_lib import mmfi, evaluate; print("PrivPose core imports: OK")'

echo "Dataset: $DATASET_ROOT"
echo "Config:  $CONFIG_FILE"
echo "Quick check passed; no training was started."


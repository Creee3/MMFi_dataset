#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/data_base}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config.yaml}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the checkpoint directory}"
MODEL_TYPE="${MODEL_TYPE:-lupi}"
CHECKPOINT="${CHECKPOINT:-best.pth}"
SPLIT="${SPLIT:-four_way_split}"
OUTPUT="${OUTPUT:-}"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_final_test_mpjpe.py"

cd "$SCRIPT_DIR"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Evaluation script not found: $EVAL_SCRIPT" >&2
  exit 2
fi

args=(
  "$DATASET_ROOT" "$CONFIG_FILE"
  --run_dir "$RUN_DIR"
  --model_type "$MODEL_TYPE"
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --device "${DEVICE:-cuda}"
  --num_workers "${NUM_WORKERS:-0}"
)
if [[ -n "$OUTPUT" ]]; then
  args+=(--output "$OUTPUT")
fi

exec "$PYTHON" -u "$EVAL_SCRIPT" "${args[@]}" "$@"

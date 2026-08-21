#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/data_base}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config.yaml}"

cd "$SCRIPT_DIR"

exec "$PYTHON" -u train_t1_rgb_teacher_mpjpe.py \
  "$DATASET_ROOT" "$CONFIG_FILE" \
  --split "${SPLIT:-cross_subject_split}" \
  --epochs "${EPOCHS:-30}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --val_batch_size "${VAL_BATCH_SIZE:-64}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --eval_num_workers "${EVAL_NUM_WORKERS:-0}" \
  --skip_final_test \
  "$@"


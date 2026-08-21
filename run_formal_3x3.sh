#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/data_base}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config.yaml}"
SELECTION_FILE="${SELECTION_FILE:-$SCRIPT_DIR/strict_offline_runs/S2P2_cmc_lr_bs_tuning_w84200_resume_clean2/selected_for_3x3.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/strict_offline_runs/S2P2_3x3_rank2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

cd "$SCRIPT_DIR"

if [[ ! -f "$SELECTION_FILE" ]]; then
  echo "Selection manifest not found: $SELECTION_FILE" >&2
  echo "Set SELECTION_FILE=/path/to/selected_for_3x3.json before running." >&2
  exit 2
fi

exec "$PYTHON" -u run_s2p2_3x3_rank2.py \
  "$DATASET_ROOT" "$CONFIG_FILE" \
  --selected_file "$SELECTION_FILE" \
  --results_root "$RESULTS_ROOT" \
  --num_workers "$NUM_WORKERS" \
  --eval_num_workers "$EVAL_NUM_WORKERS" \
  --worker_fallbacks 4 2 1 0 \
  --min_free_gpu_mib "$MIN_FREE_GPU_MIB" \
  --gpu_poll_seconds "$GPU_POLL_SECONDS" \
  "$@"


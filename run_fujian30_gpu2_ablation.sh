#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/yanyijin/STdiff"
PYTHON_BIN="/root/miniconda3/envs/holiday/bin/python"
TRAIN_PY="$ROOT/Holidiff/train.py"
CFG="$ROOT/Holidiff/configs/fujian30/holidiff_h12.yaml"
LOG_DIR="/root/autodl-tmp/STdiff_runs/logs_by_dataset_model/fujian30/holidiff_ablation"
mkdir -p "$LOG_DIR"

run_one() {
  local tag="$1"
  shift 1
  local log_path="$LOG_DIR/${tag}.log"
  echo "[START] $tag"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" -u "$TRAIN_PY" --config "$CFG" --version "$tag" "$@" \
    | tee "$log_path"
  echo "[DONE] $tag"
}

run_one "ablation_wo_dca_single__h12" \
  --sample_times 1 \
  --vs_times 1 \
  --test_times 1 \
  --aggregation_mode single \
  --train_val_aggregation_mode single \
  --test_aggregation_mode single

run_one "ablation_wo_dca_mean__h12" \
  --aggregation_mode single \
  --train_val_aggregation_mode single \
  --test_aggregation_mode single

run_one "ablation_wo_dca_median__h12" \
  --aggregation_mode median \
  --train_val_aggregation_mode median \
  --test_aggregation_mode median

run_one "ablation_wo_dca_mom__h12" \
  --aggregation_mode mom \
  --train_val_aggregation_mode mom \
  --test_aggregation_mode mom

echo "Fujian30 H12 DCA ablation runs completed."

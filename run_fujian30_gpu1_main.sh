#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/yanyijin/STdiff"
PYTHON_BIN="/root/miniconda3/envs/holiday/bin/python"
TRAIN_PY="$ROOT/Holidiff/train.py"
LOG_ROOT="/root/autodl-tmp/STdiff_runs/logs_by_dataset_model/fujian30"

run_one() {
  local model_dir="$1"
  local cfg="$2"
  local tag="$3"
  shift 3
  local log_dir="$LOG_ROOT/$model_dir"
  local log_path="$log_dir/${tag}.log"
  mkdir -p "$log_dir"
  echo "[START] $tag"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -u "$TRAIN_PY" --config "$cfg" --version "$tag" "$@" \
    | tee "$log_path"
  echo "[DONE] $tag"
}

run_one "holidiff" "$ROOT/Holidiff/configs/fujian30/holidiff_h12.yaml" "holidiff_main_gpu1__h12" \
  --train_val_aggregation_mode single \
  --test_aggregation_mode dca
run_one "holidiff" "$ROOT/Holidiff/configs/fujian30/holidiff_h24.yaml" "holidiff_main_gpu1__h24" \
  --train_val_aggregation_mode single \
  --test_aggregation_mode dca
run_one "holidiff" "$ROOT/Holidiff/configs/fujian30/holidiff_h36.yaml" "holidiff_main_gpu1__h36" \
  --train_val_aggregation_mode single \
  --test_aggregation_mode dca
run_one "dlinear" "$ROOT/Holidiff/configs/fujian30/dlinear_h36.yaml" "dlinear_main_gpu1__h36"

echo "All Fujian30 GPU1 main runs completed."

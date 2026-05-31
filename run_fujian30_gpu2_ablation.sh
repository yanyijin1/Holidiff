#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/yanyijin/STdiff"
PYTHON_BIN="/root/miniconda3/envs/holiday/bin/python"
TRAIN_PY="$ROOT/Holidiff/train.py"
CFG_DIR="$ROOT/Holidiff/configs/fujian30/ablation"
LOG_DIR="/root/autodl-tmp/STdiff_runs/logs_by_dataset_model/fujian30/holidiff_ablation"
mkdir -p "$LOG_DIR"

run_one() {
  local cfg="$1"
  local tag="$2"
  local log_path="$LOG_DIR/${tag}.log"
  echo "[START] $tag"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u "$TRAIN_PY" --config "$cfg" --version "$tag" \
    | tee "$log_path"
  echo "[DONE] $tag"
}

run_one "$CFG_DIR/wo_lstde_h12.yaml" "ablation_wo_lstde__h12"
run_one "$CFG_DIR/wo_sfcn_h12.yaml" "ablation_wo_sfcn__h12"
run_one "$CFG_DIR/wo_dca_single_h12.yaml" "ablation_wo_dca_single__h12"
run_one "$CFG_DIR/wo_dca_mean_h12.yaml" "ablation_wo_dca_mean__h12"
run_one "$CFG_DIR/wo_dca_median_h12.yaml" "ablation_wo_dca_median__h12"
run_one "$CFG_DIR/wo_dca_mom_h12.yaml" "ablation_wo_dca_mom__h12"

echo "Fujian30 H12 modular ablation runs completed."

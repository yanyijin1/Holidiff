#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/yanyijin/STdiff/Holidiff"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/fujian30/holidiff_h12.yaml}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/STdiff_runs/checkpoints/fujian30/final_holidiff__h12/checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/STdiff_runs/exp7_sensitivity/fujian30_h12}"
SEED="${SEED:-2026}"
REPEAT="${REPEAT:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

/root/miniconda3/bin/conda run -n holiday python exp/exp7sensitivity/exp7_main.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --k_list 1 5 10 15 20 30 40 50 \
  --repeat "${REPEAT}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

/root/miniconda3/bin/conda run -n holiday python exp/exp7sensitivity/plot_k_sensitivity.py \
  --csv "${OUTPUT_DIR}/k_sensitivity_summary.csv" \
  --out "${OUTPUT_DIR}/k_sensitivity_mae_time.png"
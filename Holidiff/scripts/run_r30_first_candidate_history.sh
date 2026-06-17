#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/yanyijin/STdiff/Holidiff"
TRAIN_PY="${ROOT_DIR}/train.py"
DATA_ROOT="/root/autodl-tmp/STdiff_data/datasets"
RUN_ROOT="/root/autodl-tmp/STdiff_runs/history_boundary_r30_first"
LOG_ROOT="/root/autodl-tmp/STdiff_runs/launch_logs"
SEED="${SEED:-2026}"
SEQ_LEN="96"
LABEL_LEN="48"
PRED_LEN="12"
TRAIN_RATIO="0.7"
VAL_RATIO="0.1"
TRAIN_EPOCHS="30"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

DATASETS=(
  "PEMS03-R30-001"
  "PEMS04-R30-001"
  "PEMS08-R30-001"
)

HISTORY_RATIOS=("0.3" "0.4" "0.5" "0.6" "0.7")
MODELS=("iTransformer" "HoliDiff")

ITR_CFG="${ROOT_DIR}/configs/fujian30/itransformer_h12.yaml"
HOLI_CFG="${ROOT_DIR}/configs/fujian30/holidiff_h12.yaml"

for DATASET in "${DATASETS[@]}"; do
  ROOT_PATH="${DATA_ROOT}/${DATASET}"
  DATA_PATH="warehouse/clean.csv"
  ADJ_PATH="${ROOT_PATH}/${DATASET}.csv"

  if [[ ! -f "${ROOT_PATH}/${DATA_PATH}" ]]; then
    echo "Missing dataset csv: ${ROOT_PATH}/${DATA_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${ADJ_PATH}" ]]; then
    echo "Missing adjacency csv: ${ADJ_PATH}" >&2
    exit 1
  fi

  for HR in "${HISTORY_RATIOS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      if [[ "${MODEL}" == "iTransformer" ]]; then
        CONFIG_PATH="${ITR_CFG}"
        MODEL_NAME="iTransformer"
        MODEL_ID="${DATASET}_96_12_iTransformer"
        DES="iTransformer"
      else
        CONFIG_PATH="${HOLI_CFG}"
        MODEL_NAME="HoliDiff"
        MODEL_ID="${DATASET}_96_12_HoliDiff"
        DES="HoliDiff"
      fi

      HR_TAG="${HR/./p}"
      SAVE_DIR="${RUN_ROOT}/${DATASET}/${MODEL}/hr${HR_TAG}_ep${TRAIN_EPOCHS}_seed${SEED}"

      echo "[launch] dataset=${DATASET} model=${MODEL} history_ratio=${HR} epochs=${TRAIN_EPOCHS}"
      conda run -n holiday python "${TRAIN_PY}" \
        --config "${CONFIG_PATH}" \
        --model "${MODEL_NAME}" \
        --model_id "${MODEL_ID}" \
        --des "${DES}" \
        --dataset_name "${DATASET}" \
        --root_path "${ROOT_PATH}" \
        --data_path "${DATA_PATH}" \
        --adj_path "${ADJ_PATH}" \
        --seq_len "${SEQ_LEN}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${PRED_LEN}" \
        --train_ratio "${TRAIN_RATIO}" \
        --val_ratio "${VAL_RATIO}" \
        --history_ratio "${HR}" \
        --train_epochs "${TRAIN_EPOCHS}" \
        --seed "${SEED}" \
        --save_dir "${SAVE_DIR}"
    done
  done
done

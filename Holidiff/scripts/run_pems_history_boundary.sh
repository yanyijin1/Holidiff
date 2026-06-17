#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/yanyijin/STdiff/Holidiff"
DATA_ROOT="${ROOT_DIR}/data"
SEED="${SEED:-2026}"
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LEN="${PRED_LEN:-12}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.1}"

SOURCES=("PEMS03" "PEMS04" "PEMS08")
REGIONS=("001" "002" "003" "004" "005")
HISTORY_RATIOS=("0.30" "0.35" "0.40" "0.45" "0.50" "0.55" "0.60" "0.65" "0.70")
MODELS=("iTransformer" "RegDiff")

ITR_CFG="${ROOT_DIR}/configs/fujian30/itransformer_h12.yaml"
REG_CFG="${ROOT_DIR}/configs/fujian30/holidiff_h12.yaml"

for SRC in "${SOURCES[@]}"; do
  for RID in "${REGIONS[@]}"; do
    DATASET="${SRC}-R30-${RID}"
    ROOT_PATH="${DATA_ROOT}/${DATASET}"
    DATA_PATH="warehouse/clean.csv"
    ADJ_PATH="${ROOT_PATH}/${DATASET}.csv"

    if [[ ! -f "${ROOT_PATH}/${DATA_PATH}" ]]; then
      echo "Skip missing dataset csv: ${ROOT_PATH}/${DATA_PATH}"
      continue
    fi
    if [[ ! -f "${ADJ_PATH}" ]]; then
      echo "Skip missing adjacency: ${ADJ_PATH}"
      continue
    fi

    for HR in "${HISTORY_RATIOS[@]}"; do
      for MODEL in "${MODELS[@]}"; do
        if [[ "${MODEL}" == "iTransformer" ]]; then
          CONFIG_PATH="${ITR_CFG}"
          MODEL_NAME="iTransformer"
          MODEL_ID="${DATASET}_96_12_iTransformer"
          DES="iTransformer"
        else
          CONFIG_PATH="${REG_CFG}"
          MODEL_NAME="RegDiff"
          MODEL_ID="${DATASET}_96_12_RegDiff"
          DES="RegDiff"
        fi

        SAVE_DIR="/root/yanyijin/STdiff/runs/history_boundary/${DATASET}/${MODEL}/hr${HR}_seed${SEED}"

        echo "Running dataset=${DATASET}, model=${MODEL}, history_ratio=${HR}"

        python "${ROOT_DIR}/train.py" \
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
          --seed "${SEED}" \
          --save_dir "${SAVE_DIR}"
      done
    done
  done
done

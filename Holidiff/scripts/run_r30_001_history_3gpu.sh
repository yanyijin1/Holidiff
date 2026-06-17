#!/usr/bin/env bash
# run_r30_001_history_3gpu.sh
# Three datasets in parallel on GPU 0/1/2.
# Each worker serializes: HoliDiff then iTransformer, across all history_ratios.
# All outputs go to autodl-tmp only.
set -euo pipefail

ROOT_DIR="/root/yanyijin/STdiff/Holidiff"
DATA_ROOT="/root/autodl-tmp/STdiff_data/datasets"
RUN_ROOT="/root/autodl-tmp/STdiff_runs/history_boundary_r30_001"
LAUNCH_LOG_ROOT="/root/autodl-tmp/STdiff_runs/launch_logs/r30_001_3gpu"

SEED="${SEED:-2026}"
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LEN="${PRED_LEN:-12}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
CONDA_ENV="${CONDA_ENV:-holiday}"

ITR_CFG="${ROOT_DIR}/configs/fujian30/itransformer_h12.yaml"
HOLI_CFG="${ROOT_DIR}/configs/fujian30/holidiff_h12.yaml"

HISTORY_RATIOS=("0.30" "0.35" "0.40" "0.45" "0.50" "0.55" "0.60" "0.65" "0.70")
MODELS=("HoliDiff" "iTransformer")
DATASETS=("PEMS03-R30-001" "PEMS04-R30-001" "PEMS08-R30-001")
GPUS=("0" "1" "2")

mkdir -p "${RUN_ROOT}" "${LAUNCH_LOG_ROOT}"

hr_tag() { echo "${1/./p}"; }

run_one_dataset() {
  local dataset="$1"
  local gpu="$2"
  local worker_log="${LAUNCH_LOG_ROOT}/${dataset}_gpu${gpu}_ep${TRAIN_EPOCHS}_seed${SEED}.launch.log"
  local root_path="${DATA_ROOT}/${dataset}"
  local data_path="warehouse/clean.csv"
  local adj_path="${root_path}/${dataset}.csv"

  {
    echo "[worker] started_at=$(date '+%F %T')  dataset=${dataset}  gpu=${gpu}"

    if [[ ! -f "${root_path}/${data_path}" ]]; then
      echo "[error] missing csv: ${root_path}/${data_path}"; exit 1
    fi
    if [[ ! -f "${adj_path}" ]]; then
      echo "[error] missing adj: ${adj_path}"; exit 1
    fi

    for hr in "${HISTORY_RATIOS[@]}"; do
      local tag; tag="$(hr_tag "${hr}")"

      for model in "${MODELS[@]}"; do
        local config_path model_name model_id des run_name save_dir

        if [[ "${model}" == "iTransformer" ]]; then
          config_path="${ITR_CFG}"
          model_name="iTransformer"
          model_id="${dataset}_96_12_iTransformer"
          des="iTransformer"
        else
          config_path="${HOLI_CFG}"
          model_name="HoliDiff"
          model_id="${dataset}_96_12_HoliDiff"
          des="HoliDiff"
        fi

        run_name="${dataset}_${model}_hr${tag}_ep${TRAIN_EPOCHS}_seed${SEED}"
        save_dir="${RUN_ROOT}/${dataset}/${model}/${run_name}"

        echo "[run] start  dataset=${dataset}  model=${model}  hr=${hr}  gpu=${gpu}  run=${run_name}"

        CUDA_VISIBLE_DEVICES="${gpu}" \
          conda run -n "${CONDA_ENV}" python "${ROOT_DIR}/train.py" \
            --config    "${config_path}" \
            --version   "${run_name}" \
            --model     "${model_name}" \
            --model_id  "${model_id}" \
            --des       "${des}" \
            --dataset_name "${dataset}" \
            --root_path "${root_path}" \
            --data_path "${data_path}" \
            --adj_path  "${adj_path}" \
            --seq_len   "${SEQ_LEN}" \
            --label_len "${LABEL_LEN}" \
            --pred_len  "${PRED_LEN}" \
            --train_ratio  "${TRAIN_RATIO}" \
            --val_ratio    "${VAL_RATIO}" \
            --history_ratio "${hr}" \
            --train_epochs  "${TRAIN_EPOCHS}" \
            --seed      "${SEED}" \
            --save_dir  "${save_dir}"

        echo "[run] done   dataset=${dataset}  model=${model}  hr=${hr}  gpu=${gpu}  run=${run_name}"
      done
    done

    echo "[worker] finished_at=$(date '+%F %T')  dataset=${dataset}  gpu=${gpu}"
  } >> "${worker_log}" 2>&1
}

PIDS=()
for idx in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$idx]}"
  gpu="${GPUS[$idx]}"
  echo "[launcher] spawning  dataset=${dataset}  gpu=${gpu}"
  run_one_dataset "${dataset}" "${gpu}" &
  pid=$!
  PIDS+=("${pid}")
  echo "[launcher] pid=${pid}  log=${LAUNCH_LOG_ROOT}/${dataset}_gpu${gpu}_ep${TRAIN_EPOCHS}_seed${SEED}.launch.log"
done

PID_FILE="${LAUNCH_LOG_ROOT}/r30_001_3gpu_ep${TRAIN_EPOCHS}_seed${SEED}.pids"
printf '%s\n' "${PIDS[@]}" > "${PID_FILE}"
echo "[launcher] pid_file=${PID_FILE}"
echo "[launcher] waiting for all workers..."

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[launcher] all workers finished at $(date '+%F %T')"

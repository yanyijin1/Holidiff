#!/bin/bash
# LWRGAT Phase 1 训练脚本 - 带日志保存
# 用法: bash run_lwrgat_train.sh [可选: 实验名称后缀]

set -e  # 遇到错误立即退出

# ========== 配置 ==========
CONFIG="config/lwrdiff_phase1_pcgk.yaml"
EXP_SUFFIX="${1:-}"  # 可选：实验名称后缀
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs"
RESULTS_DIR="./results"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 构建实验名称
EXP_NAME="lwrgat_phase1_${TIMESTAMP}"
if [ -n "$EXP_SUFFIX" ]; then
    EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
fi

LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

# ========== 打印配置 ==========
echo "=========================================="
echo "LWRGAT Phase 1 Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Log file: $LOG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="
echo ""

# ========== 检查依赖 ==========
echo "[CHECK] 检查数据文件..."
if [ ! -f "data/msst_speed.csv" ]; then
    echo "[ERROR] data/msst_speed.csv not found!"
    exit 1
fi

if [ ! -f "data/msst_flow.csv" ]; then
    echo "[ERROR] data/msst_flow.csv not found!"
    exit 1
fi

if [ ! -f "data/adjacent_gantry.csv" ]; then
    echo "[ERROR] data/adjacent_gantry.csv not found!"
    exit 1
fi

echo "[OK] All data files found"
echo ""

# ========== 检查 GPU ==========
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU] Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""
fi

# ========== 运行训练 ==========
echo "[TRAIN] Starting training..."
echo "[TRAIN] Command: python train_from_yaml.py --config $CONFIG --gpu 0"
echo ""

# 使用 tee 同时输出到终端和日志文件
python train_from_yaml.py \
    --config "$CONFIG" \
    --gpu 0 \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# ========== 保存训练结果 ==========
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[SUCCESS] Training completed!"
    echo "Log saved to: $LOG_FILE"
    echo "=========================================="
    
    # 提取关键指标
    echo ""
    echo "[METRICS SUMMARY]"
    if [ -f "result_long_term_forecast.txt" ]; then
        echo "Latest results:"
        tail -5 "result_long_term_forecast.txt"
    fi
else
    echo ""
    echo "=========================================="
    echo "[FAILED] Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check log: $LOG_FILE"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

#!/bin/bash
# 训练结果对比脚本 - LWRGAT vs LWRDiff 基线
# 用法: bash compare_results.sh

set -e

RESULTS_FILE="result_long_term_forecast.txt"
LOG_DIR="./logs"

echo "=========================================="
echo "LWRGAT vs LWRDiff Results Comparison"
echo "=========================================="
echo ""

# 1. 显示最近的 LWRGAT 结果
echo ">>> [1] 最近 LWRGAT Phase1 训练结果 <<<"
echo ""
if [ -d "$LOG_DIR" ]; then
    echo "Available LWRGAT logs:"
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5
    echo ""
fi

# 2. 从日志中提取训练指标
echo ">>> [2] 训练指标对比 <<<"
echo ""

# 查找 LWRGAT 日志
LWREAT_LOG=$(ls -t "$LOG_DIR"/lwrgat_phase1_*.log 2>/dev/null | head -1)

if [ -n "$LWREAT_LOG" ]; then
    echo "Latest LWRGAT log: $LWREAT_LOG"
    echo ""
    
    echo "--- Loss 趋势 ---"
    grep -E "Train Loss:|Vali Loss:" "$LWREAT_LOG" | tail -10
    echo ""
    
    echo "--- 最终测试指标 ---"
    grep -E "mse:|mae:|rmse:" "$LWREAT_LOG" | tail -5
    echo ""
    
    echo "--- 物理参数监控 ---"
    grep -E "v_critical|regime|decouple" "$LWREAT_LOG" | tail -10
    echo ""
else
    echo "[INFO] No LWRGAT log found in $LOG_DIR"
fi

# 3. 结果文件对比
echo ">>> [3] 结果文件对比 <<<"
echo ""

if [ -f "$RESULTS_FILE" ]; then
    echo "Content of $RESULTS_FILE:"
    echo "---"
    tail -20 "$RESULTS_FILE"
    echo "---"
    echo ""
else
    echo "[INFO] No result file found: $RESULTS_FILE"
fi

# 4. Checkpoint 信息
echo ">>> [4] Checkpoint 信息 <<<"
echo ""
if [ -d "./checkpoints" ]; then
    echo "Available checkpoints:"
    find ./checkpoints -name "*.pth" -mtime -7 2>/dev/null | head -10
    echo ""
fi

echo "=========================================="

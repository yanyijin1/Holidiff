#!/bin/bash
# SimDiff 36步长程预测 - 快速测试脚本

cd /root/yanyijin/STdiff/baseline/SimDiff

echo "=============================================="
echo "SimDiff 36步预测 - 测试运行 (2 epochs)"
echo "=============================================="

python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id simdiff_36step \
  --model SimDiff \
  --data custom \
  --root_path ./dataset/ \
  --data_path msst_speed.csv \
  --features M \
  --target s0 \
  --freq t \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 36 \
  --enc_in 30 \
  --dec_in 30 \
  --c_out 30 \
  --e_layers 1 \
  --d_model 64 \
  --num_heads 4 \
  --d_ff 256 \
  --dropout 0.1 \
  --skip_dropout 0.1 \
  --stride 6 \
  --patch_len 6 \
  --batch_size 32 \
  --train_epochs 30 \
  --patience 8 \
  --learning_rate 0.0005 \
  --loss_type MAE \
  --is_diff 1 \
  --diff_steps 1000 \
  --s_steps 20 \
  --skip_type time_uniform \
  --method multistep \
  --order 2 \
  --lower_order_final true \
  --sample_times 5 \
  --vs_times 5 \
  --use_mom 1 \
  --new_norm 1 \
  --rmom 20 \
  --n_b 5 \
  --coss 0.008 \
  --itr 1 \
  --des phase0_baseline_36step \
  --use_dtw false

echo "=============================================="
echo "测试完成"
echo "=============================================="
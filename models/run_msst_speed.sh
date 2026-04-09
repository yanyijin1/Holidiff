#!/bin/bash
# Run LWRDiff on MSST Speed dataset
# Exact parameters from need/LWRdiff/script/train_msst_speed.yaml

cd /home/yanyijin/research/MSST-CL/models

LOG_FILE="./logs/lwrdiff_train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ./logs

/home/yulong.chen/miniconda3/envs/holiday/bin/python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id msst_speed_96_12 \
    --model LWRdiff \
    --data custom \
    --root_path ./data/ \
    --data_path msst_speed.csv \
    --features M \
    --target s0 \
    --freq t \
    --enc_in 30 \
    --dec_in 30 \
    --c_out 30 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 12 \
    --e_layers 1 \
    --d_model 128 \
    --num_heads 8 \
    --stride 3 \
    --patch_len 6 \
    --dropout 0.0 \
    --d_ff 512 \
    --seed 42 \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 8 \
    --learning_rate 0.00004 \
    --loss_type MAE \
    --is_diff 1 \
    --diff_steps 100 \
    --s_steps 2 \
    --skip_type time_uniform \
    --method multistep \
    --order 2 \
    --lower_order_final true \
    --sample_times 20 \
    --vs_times 5 \
    --use_mom 1 \
    --mom_mode mom \
    --mom_tau 5.0 \
    --mom_clusters 2 \
    --new_norm 1 \
    --rmom 20 \
    --n_b 5 \
    --coss 0.008 \
    --embed timeF \
    --num_workers 4 \
    --des msst_speed_baseline > "$LOG_FILE" 2>&1 &

echo "Log: $LOG_FILE"
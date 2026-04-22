#!/usr/bin/env python
"""
训练入口脚本
"""
import os
import torch
import sys

sys.path.insert(0, '.')

from trainer import Exp_Long_Term_Forecast
import yaml

print("="*60)
print("[TRAIN] Loading config...")
with open('config/lwrdiff_phase1_pcgk.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

class Args:
    pass

args = Args()

# Model switch
model_switch = cfg.get('model_switch', {})
args.use_stformer = model_switch.get('use_stformer', False)
args.use_lwrgat = model_switch.get('use_lwrgat', True)
args.st_layers = model_switch.get('st_layers', 2)

# Data
data_cfg = cfg.get('data', {})
args.data = data_cfg.get('data', 'custom')
args.root_path = data_cfg.get('root_path', './data/')
args.data_path = data_cfg.get('data_path', 'msst_speed.csv')
args.flow_path = data_cfg.get('flow_path', None)
args.features = data_cfg.get('features', 'M')
args.target = data_cfg.get('target', 'OT')
args.freq = data_cfg.get('freq', 't')
args.enc_in = data_cfg.get('enc_in', 30)
args.dec_in = data_cfg.get('dec_in', 30)
args.c_out = data_cfg.get('c_out', 30)

# Graph
graph_cfg = cfg.get('graph', {})
args.graph_enabled = graph_cfg.get('enabled', False)
args.graph_adj_path = graph_cfg.get('adj_path', None)
args.graph_num_nodes = graph_cfg.get('num_nodes', None)

# Window
window_cfg = cfg.get('window', {})
args.seq_len = window_cfg.get('seq_len', 96)
args.label_len = window_cfg.get('label_len', 48)
args.pred_len = window_cfg.get('pred_len', 12)

# Model
model_cfg = cfg.get('model', {})
args.e_layers = model_cfg.get('e_layers', 1)
args.d_model = model_cfg.get('d_model', 64)
args.num_heads = model_cfg.get('num_heads', 4)
args.d_ff = model_cfg.get('d_ff', 256)
args.dropout = model_cfg.get('dropout', 0.1)
args.stride = model_cfg.get('stride', 3)
args.patch_len = model_cfg.get('patch_len', 6)

# Train
train_cfg = cfg.get('train', {})
args.num_workers = train_cfg.get('num_workers', 0)
args.itr = train_cfg.get('itr', 1)
args.train_epochs = train_cfg.get('train_epochs', 10)
args.batch_size = train_cfg.get('batch_size', 32)
args.patience = train_cfg.get('patience', 3)
args.learning_rate = train_cfg.get('learning_rate', 0.001)
args.use_amp = train_cfg.get('use_amp', False)

# Diffusion
diff_cfg = cfg.get('diffusion', {})
args.is_diff = diff_cfg.get('is_diff', 1)
args.diff_steps = diff_cfg.get('diff_steps', 100)
args.skip_type = diff_cfg.get('skip_type', 'time_uniform')
args.method = diff_cfg.get('method', 'multistep')
args.order = diff_cfg.get('order', 2)
args.lower_order_final = diff_cfg.get('lower_order_final', 'true')
args.s_steps = diff_cfg.get('s_steps', 2)
args.sample_times = diff_cfg.get('sample_times', 20)

# Loss
loss_cfg = cfg.get('loss', {})
args.loss_type = loss_cfg.get('loss_type', 'MAE')
args.vs_times = loss_cfg.get('vs_times', 5)
args.rmom = loss_cfg.get('rmom', 20)
args.n_b = loss_cfg.get('n_b', 5)
args.coss = loss_cfg.get('coss', 0.008)

# Norm
args.new_norm = cfg.get('new_norm', 1)

# TimeF embedding
args.embed = cfg.get('embed', 'timeF')
args.seasonal_patterns = cfg.get('seasonal_patterns', 'Monthly')

# Set device
args.use_gpu = True
args.use_multi_gpu = False
args.device = 'cuda:0'

print(f"[TRAIN] Config: seq_len={args.seq_len}, pred_len={args.pred_len}")
print(f"[TRAIN] Model: use_lwrgat={args.use_lwrgat}, st_layers={args.st_layers}")
print(f"[TRAIN] Diffusion: steps={args.diff_steps}, s_steps={args.s_steps}")
print("="*60)

if __name__ == '__main__':
    exp = Exp_Long_Term_Forecast(args)
    print('[TRAIN] Starting training...')
    exp.train()

#!/usr/bin/env python
import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from LWRGAT.Model import Model
from data_loader import data_provider
import yaml

# 加载配置
with open('config/lwrdiff_phase1_pcgk.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# 构建 args 对象
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

# Model args
model_args = cfg.get('model_args', {})
args.e_layers = model_args.get('e_layers', 1)
args.d_model = model_args.get('d_model', 64)
args.num_heads = model_args.get('num_heads', 4)
args.stride = model_args.get('stride', 3)
args.patch_len = model_args.get('patch_len', 6)
args.dropout = model_args.get('dropout', 0.1)
args.skip_dropout = model_args.get('skip_dropout', 0.1)
args.d_ff = model_args.get('d_ff', 256)

# Optimization
opt_cfg = cfg.get('optimization', {})
args.batch_size = opt_cfg.get('batch_size', 32)
args.num_workers = opt_cfg.get('num_workers', 4)

# Diffusion
args.is_diff = cfg.get('sampling', {}).get('is_diff', 1)
args.diff_steps = cfg.get('sampling', {}).get('diff_steps', 100)
args.s_steps = cfg.get('sampling', {}).get('s_steps', 2)
args.skip_type = cfg.get('sampling', {}).get('skip_type', 'time_uniform')
args.method = cfg.get('sampling', {}).get('method', 'multistep')
args.order = cfg.get('sampling', {}).get('order', 2)
args.lower_order_final = cfg.get('sampling', {}).get('lower_order_final', 'true')
args.sample_times = cfg.get('sampling', {}).get('sample_times', 20)
args.new_norm = cfg.get('sampling', {}).get('new_norm', 1)
args.rmom = cfg.get('sampling', {}).get('rmom', 20)
args.n_b = cfg.get('sampling', {}).get('n_b', 5)
args.coss = cfg.get('sampling', {}).get('coss', 0.008)

# Other
args.embed = 'timeF'
args.freq = 't'
args.seasonal_patterns = 'Monthly'

setting = 'long_term_forecast_lwrdiff_phase1_pcgk_LWRdiff_custom_ftM_sl96_ll48_pl12_dm64_nh4_el1_dl1_df256_ebtimeF_pcgk_gat_phase1_0'

# 加载模型
print('Loading model...')
model = Model(args)
model = model.cuda()
model.eval()
model.load_state_dict(torch.load(f'checkpoints/{setting}/checkpoint.pth', map_location='cuda:0'))
print('Model loaded!')

# 加载测试数据
test_data, test_loader = data_provider(args, 'test')
print(f'Test samples: {len(test_data)}, batches: {len(test_loader)}')

# metric 函数
def MAE(pred, true): return np.mean(np.abs(pred - true))
def MSE(pred, true): return np.mean((pred - true) ** 2)
def RMSE(pred, true): return np.sqrt(MSE(pred, true))
def MAPE(pred, true): return np.mean(np.abs((pred - true) / (true + 1e-5)))
def MSPE(pred, true): return np.mean(np.square((pred - true) / (true + 1e-5)))

# 测试
print('Running test...')
preds, trues = [], []
for i, batch in enumerate(test_loader):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
    
    batch_x = batch_x.float().cuda()
    batch_y = batch_y.float().cuda()
    batch_x_mark = batch_x_mark.float().cuda()
    batch_y_mark = batch_y_mark.float().cuda()
    
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
    
    with torch.no_grad():
        outputs, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=1)
    
    outputs = outputs[:, -args.pred_len:, :].detach().cpu().numpy()
    batch_y_np = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
    
    preds.append(outputs)
    trues.append(batch_y_np)
    
    if (i + 1) % 50 == 0:
        print(f'  batch {i+1}/{len(test_loader)}')

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)
print(f'Test shape: preds={preds.shape}, trues={trues.shape}')

mae, mse, rmse, mape, mspe = MAE(preds, trues), MSE(preds, trues), RMSE(preds, trues), MAPE(preds, trues), MSPE(preds, trues)
print(f'\n=== Test Results ===')
print(f'MSE:  {mse:.6f}')
print(f'RMSE: {rmse:.6f}')
print(f'MAE:  {mae:.6f}')
print(f'MAPE: {mape:.6f}')
print(f'MSPE: {mspe:.6f}')

# 保存结果
result_line = f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}'
with open('logs/result_long_term_forecast.txt', 'a') as f:
    f.write(f'{setting}  \n')
    f.write(result_line + '\n')
print(f'Results saved!')

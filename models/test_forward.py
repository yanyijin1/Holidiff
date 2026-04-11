"""快速测试模型前向传播"""
import sys
sys.path.insert(0, '/root/yanyijin/STdiff/models')

import torch
import yaml
from fourier.model import Model

# 加载配置
with open('/root/yanyijin/STdiff/models/config/lwrdiff.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 转换为 Namespace
class Args:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Args(v))
            else:
                setattr(self, k, v)

def get_val(cfg, keys, default=None):
    for k in keys:
        if isinstance(cfg, dict):
            cfg = cfg.get(k, default)
        else:
            return getattr(cfg, k, default)
    return cfg

args = Args({
    'task_name': 'long_term_forecast',
    'model': 'LWRdiff',
    'model_id': 'lwrdiff_96_12',
    'data': 'ETTh1',
    'root_path': '/root/yanyijin/STdiff/models/data/',
    'data_path': 'msst_speed.csv',
    'features': 'M',
    'target': 'OT',
    'freq': 'h',
    'enc_in': 30,
    'dec_in': 30,
    'c_out': 30,
    'seq_len': 96,
    'label_len': 48,
    'pred_len': 96,
    'e_layers': 2,
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 1024,
    'dropout': 0.1,
    'skip_dropout': 0.1,
    'stride': 8,
    'patch_len': 16,
    'embed': 'timeF',
    'activation': 'gelu',
    'is_diff': 1,
    'diff_steps': 100,
    's_steps': 5,
    'skip_type': 'time_uniform',
    'method': 'multistep',
    'order': 2,
    'lower_order_final': 'true',
    'sample_times': 50,
    'use_mom': 1,
    'mom_mode': 'mom',
    'new_norm': 1,
    'rmom': 20,
    'n_b': 5,
    'coss': 0.008,
    'batch_size': 32,
    'graph_enabled': True,
    'graph_adj_path': '/root/yanyijin/STdiff/models/data/adjacent_gantry.csv',
    'graph_num_nodes': 30,
})

# 创建模型
print("创建模型...")
model = Model(args).float()
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 测试输入
B, L, N = 4, 96, 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_enc = torch.randn(B, L, N).to(device)
x_dec = torch.randn(B, 48+96, N).to(device)
x_mark_enc = torch.randn(B, L, 4).to(device)
x_mark_dec = torch.randn(B, 144, 4).to(device)

# 训练模式前向
model = model.to(device)
model.train()
print("\n训练模式测试...")
try:
    out, weight = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"训练输出: {out.shape}, weight: {weight.shape}")
    print("训练模式通过 ✓")
except Exception as e:
    import traceback
    print(f"训练模式失败: {e}")
    traceback.print_exc()

# 推理模式测试
model.eval()
print("\n推理模式测试...")
with torch.no_grad():
    try:
        out, all_outs = model(x_enc, x_mark_enc, x_dec, x_mark_dec, sample_times=2)
        print(f"推理输出: {out.shape}, all_outs: {all_outs.shape}")
        print("推理模式通过 ✓")
    except Exception as e:
        import traceback
        print(f"推理模式失败: {e}")
        traceback.print_exc()

print("\n测试完成！")

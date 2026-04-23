"""
调试 flow_x 是否正确传递
"""
import torch
import sys
sys.path.insert(0, '/root/yanyijin/STdiff/models')
sys.path.insert(0, '/root/yanyijin/STdiff/models/data_provider')

from data_loader import Dataset_Custom
from torch.utils.data import DataLoader

# 创建数据集
root_path = '/root/yanyijin/STdiff/models/data/'
data_path = 'msst_speed.csv'
flow_path = 'msst_flow.csv'

ds = Dataset_Custom(
    root_path=root_path,
    data_path=data_path,
    flow_path=flow_path,
    flag='train',
    size=[96, 48, 12],
    features='M'
)

loader = DataLoader(ds, batch_size=4, shuffle=False)

# 获取一个 batch
batch = next(iter(loader))
print(f"Batch 长度: {len(batch)}")
print(f"batch_x shape: {batch[0].shape}")
print(f"batch_flow_x shape: {batch[4].shape}")
print(f"batch_flow_x requires_grad: {batch[4].requires_grad}")

# 检查 Model.forward_train 中的数据处理
x_enc = batch[0]  # (B, L, N)
flow_x = batch[4]  # (B, L, N)

print(f"\nx_enc: mean={x_enc.mean():.4f}, std={x_enc.std():.4f}")
print(f"flow_x: mean={flow_x.mean():.4f}, std={flow_x.std():.4f}")

# 模拟 Model 中的处理
q_obs = flow_x.permute(0, 2, 1)  # (B, N, L)
print(f"\nq_obs (after permute): mean={q_obs.mean():.4f}, std={q_obs.std():.4f}")

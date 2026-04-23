"""
测试 PCGK 梯度流
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 PCGK
import sys
sys.path.insert(0, '/root/yanyijin/STdiff/models')
from LWRGAT.stlwr_gat import PhysicsComputedGraphKernel

def test_pcgk_gradient():
    print("=" * 60)
    print("测试 PCGK 梯度流")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建 PCGK
    pcgk = PhysicsComputedGraphKernel(d_model=64, num_nodes=30, v_critical_init=0.0).to(device)
    print(f"\nPCGK 参数:")
    print(f"  raw_v_critical: {pcgk.raw_v_critical.data}")
    print(f"  raw_temperature: {pcgk.raw_temperature.data}")
    print(f"  v_critical property: {pcgk.v_critical}")
    print(f"  temperature property: {pcgk.temperature}")
    
    # 创建测试数据
    B, N, P, d = 8, 30, 16, 64
    q_embed = torch.randn(B, N, P, d, device=device, requires_grad=True)
    q_obs = torch.randn(B, N, P, device=device, requires_grad=True)  # 需要梯度！
    A_down = torch.rand(N, N, device=device).fill_diagonal_(1)
    A_up = A_down.T
    
    print(f"\n数据:")
    print(f"  q_obs mean: {q_obs.mean().item():.4f}, std: {q_obs.std().item():.4f}")
    
    # 前向传播
    print("\n前向传播...")
    A_eff, regime = pcgk(q_embed, q_obs, A_down, A_up, return_regime=True)
    print(f"  A_eff shape: {A_eff.shape}")
    print(f"  regime shape: {regime.shape}")
    print(f"  regime mean: {regime.mean().item():.4f}")
    
    # 模拟损失（使用 regime）
    loss = regime.mean()  # 简化损失
    
    print(f"\n损失: {loss.item():.6f}")
    
    # 反向传播
    print("\n反向传播...")
    loss.backward()
    
    # 检查梯度
    print(f"\n梯度检查:")
    print(f"  q_embed.grad exists: {q_embed.grad is not None}")
    print(f"  raw_v_critical.grad: {pcgk.raw_v_critical.grad}")
    print(f"  raw_temperature.grad: {pcgk.raw_temperature.grad}")
    
    # 手动验证梯度
    print("\n手动验证:")
    x = (q_obs - pcgk.v_critical) * pcgk.temperature
    sig_grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))
    print(f"  sigmoid gradient mean: {sig_grad.mean().item():.6f}")
    
    return pcgk


def test_with_stlwr_layer():
    """测试完整的 STLWRGATLayer"""
    print("\n" + "=" * 60)
    print("测试完整 STLWRGATLayer")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from LWRGAT.stlwr_gat import STLWRGATLayer
    
    # 创建层
    layer = STLWRGATLayer(d_model=64, num_heads=4, num_nodes=30).to(device)
    
    # 创建输入
    B, N, P, d = 8, 30, 16, 64
    h = torch.randn(B, N, P, d, device=device, requires_grad=True)
    q_obs = torch.randn(B, N, P, device=device)
    A_down = torch.rand(N, N, device=device).fill_diagonal_(1)
    A_up = A_down.T
    
    print(f"\n输入 h requires_grad: {h.requires_grad}")
    print(f"h.grad before forward: {h.grad}")
    
    # 前向传播
    H_out, A_kernel, regime = layer(h, A_down, A_up, q_obs=q_obs, v_obs=q_obs)
    
    print(f"\n输出:")
    print(f"  H_out shape: {H_out.shape}")
    print(f"  A_kernel shape: {A_kernel.shape}")
    print(f"  regime shape: {regime.shape}")
    
    # 损失
    loss = H_out.mean() + regime.mean() * 0.1
    print(f"\n损失: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    print(f"\nh.grad after backward: {h.grad is not None}")
    if h.grad is not None:
        print(f"  h.grad norm: {h.grad.norm().item():.6f}")
    
    # 检查 PCGK 梯度
    pcgk = layer.pcgk
    print(f"\nPCGK 梯度:")
    print(f"  raw_v_critical.grad: {pcgk.raw_v_critical.grad}")
    print(f"  raw_temperature.grad: {pcgk.raw_temperature.grad}")


if __name__ == '__main__':
    test_pcgk_gradient()
    test_with_stlwr_layer()

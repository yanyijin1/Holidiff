"""
调试：追踪 PCGK 梯度流
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DebugPCGK(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_v_critical = nn.Parameter(torch.tensor(0.0))
        self.raw_temperature = nn.Parameter(torch.log(torch.exp(torch.tensor(1.0)) - 1))
        
    @property
    def temperature(self):
        return F.softplus(self.raw_temperature)
    
    @property
    def v_critical(self):
        return torch.clamp(self.raw_v_critical, min=-2.0, max=3.0)
    
    def compute_regime(self, v_obs):
        x = (v_obs - self.v_critical) * self.temperature
        regime = torch.sigmoid(x).mean(dim=-1, keepdim=True)
        return regime, x
    
    def forward(self, h, v_obs, q_obs, A_down, A_up):
        regime, x = self.compute_regime(v_obs)
        
        # 简化：直接用 regime 的均值影响输出
        regime_scalar = regime.mean()
        h_out = h * regime_scalar  # regime 影响输出
        
        return h_out, regime, None


def test_full_flow():
    print("=" * 60)
    print("测试完整前向-反向流程")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    pcgk = DebugPCGK().to(device)
    
    # 创建测试数据
    B, N, P, d = 32, 30, 16, 64
    h = torch.randn(B, N, P, d, device=device, requires_grad=True)
    v_obs = torch.randn(B, N, P, device=device)  # 随机数据
    q_obs = torch.randn(B, N, P, device=device)
    A_down = torch.rand(N, N, device=device).fill_diagonal_(1)
    A_up = A_down.T
    
    print(f"\n数据范围:")
    print(f"  v_obs: mean={v_obs.mean().item():.3f}, std={v_obs.std().item():.3f}")
    print(f"  v_obs: min={v_obs.min().item():.3f}, max={v_obs.max().item():.3f}")
    print(f"  v_critical (property): {pcgk.v_critical.item():.3f}")
    print(f"  temperature (property): {pcgk.temperature.item():.3f}")
    
    # 前向传播
    h_out, regime, A_kernel = pcgk(h, v_obs, q_obs, A_down, A_up)
    
    print(f"\n前向结果:")
    print(f"  regime mean: {regime.mean().item():.4f}")
    print(f"  regime min: {regime.min().item():.4f}, max: {regime.max().item():.4f}")
    if A_kernel is not None:
        print(f"  A_kernel mean: {A_kernel.mean().item():.4f}")
    
    # 损失
    loss = h_out.mean()
    print(f"\n损失: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    print(f"\n梯度:")
    print(f"  h.grad: {h.grad.norm().item():.6f}")
    print(f"  pcgk.raw_v_critical.grad: {pcgk.raw_v_critical.grad}")
    print(f"  pcgk.raw_temperature.grad: {pcgk.raw_temperature.grad}")
    
    # 验证梯度计算
    print(f"\n梯度验证:")
    print(f"  ∂loss/∂v_critical = ∂loss/∂regime × ∂regime/∂v_critical")
    print(f"  regime = sigmoid((v_obs - v_critical) * temp)")
    print(f"  ∂regime/∂v_critical = sigmoid'(x) * (-temp)")
    
    x = (v_obs - pcgk.v_critical) * pcgk.temperature
    sig_grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))
    print(f"  sigmoid gradient mean: {sig_grad.mean().item():.6f}")
    
    # 检查是否所有 v_obs 都远大于/小于 v_critical
    v_diff = v_obs - pcgk.v_critical
    print(f"\n  v_obs - v_critical: mean={v_diff.mean().item():.3f}, std={v_diff.std().item():.3f}")
    print(f"  x = (v_obs - v_critical) * temp: mean={x.mean().item():.3f}")
    
    # 如果 x 太大，sigmoid 梯度会消失
    if x.abs().mean() > 3:
        print(f"  ⚠️ 警告: x 的绝对值太大，sigmoid 梯度接近 0!")
    
    return pcgk


def test_with_real_data():
    """使用更真实的 v_obs 值"""
    print("\n" + "=" * 60)
    print("测试场景：v_obs 在归一化空间 [-2, 3] 内")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pcgk = DebugPCGK().to(device)
    
    B, N, P = 32, 30, 16
    # 模拟归一化数据：通常在 [-2, 3] 范围内
    v_obs = torch.rand(B, N, P, device=device) * 5 - 2  # [-2, 3]
    h = torch.randn(B, 30, P, 64, device=device, requires_grad=True)
    q_obs = torch.randn(B, N, P, device=device)
    A_down = torch.rand(30, 30, device=device).fill_diagonal_(1)
    A_up = A_down.T
    
    print(f"\nv_obs 范围: [{v_obs.min().item():.3f}, {v_obs.max().item():.3f}]")
    
    h_out, regime, _ = pcgk(h, v_obs, q_obs, A_down, A_up)
    loss = h_out.mean()
    loss.backward()
    
    print(f"regime 范围: [{regime.min().item():.4f}, {regime.max().item():.4f}]")
    print(f"raw_v_critical.grad: {pcgk.raw_v_critical.grad.item():.8f}")
    print(f"raw_temperature.grad: {pcgk.raw_temperature.grad.item():.8f}")


if __name__ == '__main__':
    test_full_flow()
    test_with_real_data()

"""
端到端测试梯度流
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 模拟简化的 FormerBone
class SimpleFormerBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_len = 6
        self.stride = 3
        self.enc_in = 30
        self.d_model = 64
        
        # PCGK
        self.pcgk_raw_v_critical = nn.Parameter(torch.tensor(0.0))
        self.pcgk_raw_temperature = nn.Parameter(torch.tensor(0.54))
        
        self.input_proj = nn.Linear(6, 64)
        self.time_embed = nn.Sequential(nn.Linear(100, 64), nn.ReLU())
        
    @property
    def v_critical(self):
        return torch.clamp(self.pcgk_raw_v_critical, min=-2.0, max=3.0)
    
    @property
    def temperature(self):
        return F.softplus(self.pcgk_raw_temperature)
    
    def compute_regime(self, v_obs):
        x = (v_obs - self.v_critical) * self.temperature
        regime = torch.sigmoid(x).mean(dim=-1, keepdim=True)
        return regime
    
    def forward(self, x, timesteps, cond_ts):
        B_N, T = x.shape
        N = self.enc_in
        B = B_N // N
        
        x = rearrange(x, '(b n) t -> b n t', n=N)
        cond_ts = rearrange(cond_ts, '(b n) t -> b n t', n=N)
        
        # Patch
        cond_patches = cond_ts.unfold(-1, size=self.patch_len, step=self.stride)
        patches = self.input_proj(cond_patches)
        
        # Time embed
        t_flat = timesteps.reshape(B * N)
        time_embed = self.time_embed(t_flat).reshape(B, N, 1, self.d_model)
        h = patches + time_embed
        
        # v_obs
        v_obs = cond_ts.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)
        print(f"  v_obs.requires_grad: {v_obs.requires_grad}")
        
        # Regime
        regime = self.compute_regime(v_obs)
        print(f"  regime.requires_grad: {regime.requires_grad}")
        
        return h, regime


def test_e2e():
    print("=" * 60)
    print("端到端测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleFormerBone().to(device)
    
    # 测试数据
    B, N, seq_len, pred_len = 4, 30, 96, 12
    x_enc = torch.randn(B, seq_len, N, requires_grad=True, device=device).float()
    x_dec = torch.randn(B, seq_len + pred_len, N, requires_grad=True, device=device).float()
    timesteps = torch.randint(0, 100, size=[B * N], device=device).long()
    
    print(f"x_enc.requires_grad: {x_enc.requires_grad}")
    
    # 简化前向
    cond_ts = x_enc.permute(0, 2, 1)  # (B, N, seq_len)
    x_target = x_dec[:, -pred_len:, :]
    x_flat = x_target.permute(0, 2, 1).reshape(B * N, pred_len)
    
    print(f"cond_ts.shape: {cond_ts.shape}")
    print(f"cond_ts.requires_grad: {cond_ts.requires_grad}")
    
    # 前向
    h, regime = model(x_flat, timesteps, cond_ts.reshape(B * N, seq_len))
    print(f"h.shape: {h.shape}")
    
    # 损失
    loss = regime.mean()
    print(f"\nloss: {loss.item():.6f}")
    print(f"loss.requires_grad: {loss.requires_grad}")
    
    # 反向
    loss.backward()
    
    print(f"\n梯度:")
    print(f"  x_enc.grad: {x_enc.grad is not None}")
    print(f"  raw_v_critical.grad: {model.pcgk_raw_v_critical.grad}")
    print(f"  raw_temperature.grad: {model.pcgk_raw_temperature.grad}")


if __name__ == '__main__':
    test_e2e()

"""
STLWR Layer: LWRSpatialConv + TemporalAttention + FFN

核心修正：
- LWRSpatialConv 使用显式相态门控，替代 softmax 归一化图卷积
- 上游/下游分别聚合，regime 直接控制混合比例
- 物理参数梯度路径：regime -> h_phys -> loss（直接无衰减）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pcgk import PhysicsComputedGraphKernel


class LWRSpatialConv(nn.Module):
    """
    LWR 空间卷积：物理矩阵乘法 + 显式相态门控

    方向逻辑（已修正）：
    - 自由流 (regime -> 1)：信息向下游传播，节点 i 受上游邻居影响
    - 拥堵流 (regime -> 0)：拥堵波向上游回溢，节点 i 受下游邻居影响
    - 因此：自由流时上游聚合主导，拥堵流时下游聚合主导
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 上游/下游独立投影
        self.W_up = nn.Linear(d_model, d_model)
        self.W_down = nn.Linear(d_model, d_model)

        # 物理主导系数（残差连接权重）
        self.beta_phys = nn.Parameter(torch.tensor(0.3))

    def forward(self, h, A_up, A_down, regime, W_up_kernel=None, W_down_kernel=None):
        """
        Args:
            h:              (B, N, P, d) 输入特征
            A_up:           (N, N)       上游物理拓扑
            A_down:         (N, N)       下游物理拓扑
            regime:         (B, N, 1)    相态（1=自由流，0=拥堵）
            W_up_kernel:    (B, N, N)    可选边权重（上游）
            W_down_kernel:  (B, N, N)    可选边权重（下游）

        Returns:
            h_out: (B, N, P, d)  残差更新: h + beta * h_phys
        """
        B, N, P, d = h.shape

        # reshape for batch matmul: (B*P, N, d)
        h_2d = h.permute(0, 2, 1, 3).reshape(B * P, N, d)

        # === 上游聚合 ===
        A_up_exp = A_up.unsqueeze(0).unsqueeze(1).expand(B, P, N, N).reshape(B * P, N, N)
        if W_up_kernel is not None:
            W_up_exp = W_up_kernel.unsqueeze(1).expand(B, P, N, N).reshape(B * P, N, N)
            A_up_exp = A_up_exp * W_up_exp

        # 行归一化（稳定特征尺度）：每行除以该行所有边权重之和
        A_up_sum = A_up_exp.sum(dim=-1, keepdim=True) + 1e-6
        A_up_exp = A_up_exp / A_up_sum

        h_up = torch.bmm(A_up_exp, h_2d)           # (B*P, N, d)
        h_up = self.W_up(h_up)

        # === 下游聚合 ===
        A_down_exp = A_down.unsqueeze(0).unsqueeze(1).expand(B, P, N, N).reshape(B * P, N, N)
        if W_down_kernel is not None:
            W_down_exp = W_down_kernel.unsqueeze(1).expand(B, P, N, N).reshape(B * P, N, N)
            A_down_exp = A_down_exp * W_down_exp

        # 行归一化
        A_down_sum = A_down_exp.sum(dim=-1, keepdim=True) + 1e-6
        A_down_exp = A_down_exp / A_down_sum

        h_down = torch.bmm(A_down_exp, h_2d)
        h_down = self.W_down(h_down)

        # === 显式相态门控 ===
        # 自由流(regime->1): 上游主导；拥堵流(regime->0): 下游主导
        # regime: (B, N, 1) -> (B*P, N, 1)
        regime_exp = regime.unsqueeze(1).expand(B, P, N, 1).reshape(B * P, N, 1)
        h_phys = regime_exp * h_up + (1.0 - regime_exp) * h_down

        # reshape back: (B, N, P, d)
        h_phys = h_phys.reshape(B, P, N, d).permute(0, 2, 1, 3)

        beta = torch.sigmoid(self.beta_phys)
        return h + beta * h_phys


class TemporalAttention(nn.Module):
    """Patch 间自注意力（时间维度）"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        B, N, P, d = h.shape

        h_2d = h.reshape(B * N, P, d)
        q = self.W_q(h_2d).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(h_2d).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(h_2d).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)

        scale = self.d_head ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        h_out = (attn @ v).transpose(1, 2).reshape(B * N, P, d)
        h_out = self.W_o(h_out)
        return h_out.reshape(B, N, P, d)


class STLWRLayer(nn.Module):
    """
    STLWR Layer: PCGK -> LWRSpatialConv -> TemporalAttention -> FFN
    """

    def __init__(self, d_model, n_heads, num_nodes, dropout=0.1):
        super().__init__()
        self.pcgk = PhysicsComputedGraphKernel(d_model, num_nodes)
        self.spatial = LWRSpatialConv(d_model)
        self.temporal = TemporalAttention(d_model, n_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

        self.norm_spatial = nn.LayerNorm(d_model)
        self.norm_temporal = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, A_phys_down, A_phys_up,
                cached_kernel=None, cached_regime=None,
                q_obs=None, v_obs=None):
        B, N, P, d = h.shape

        # fallback: 从特征提取代理信号
        if q_obs is None or v_obs is None:
            q_strength = h.norm(p=2, dim=-1).mean(dim=-1, keepdim=True)
            q_obs = q_strength.expand(B, N, P)
            v_obs = q_obs

        # Step 1: PCGK（支持缓存）
        if cached_kernel is not None and cached_regime is not None:
            W_up, W_down = cached_kernel
            regime = cached_regime
        else:
            W_up, W_down, regime = self.pcgk(
                h, q_obs, A_phys_down, A_phys_up,
                return_regime=True, v_obs=v_obs
            )

        # Step 2: LWR 空间传播（显式门控，无 softmax）
        h_spatial = self.spatial(
            h, A_phys_up, A_phys_down, regime, W_up, W_down
        )
        h_spatial = self.norm_spatial(h_spatial)
        h_spatial = self.dropout(h_spatial)

        # Step 3: 时间注意力
        h_temp = self.temporal(h_spatial)
        h_temp = self.norm_temporal(h_temp)

        # Step 4: FFN + 残差
        h_out = h_temp + self.ffn(h_temp)
        return self.dropout(h_out), (W_up, W_down), regime

"""
PCGK: Physics-Computed Graph Kernel (修正版)

核心修正：
1. 彻底移除 softmax 行归一化，改用稀疏边权重
2. 输出上游/下游边权重 (W_up, W_down) + 相态 regime
3. v_critical / temperature 梯度路径直接，无数值抵消
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsComputedGraphKernel(nn.Module):
    def __init__(self, d_model, num_nodes, v_critical_init=0.0):
        super().__init__()
        self.num_nodes = num_nodes

        # 可学习临界速度（归一化空间）：支持通过参数初始化
        self.raw_v_critical = nn.Parameter(
            torch.tensor(0.5, dtype=torch.float32)
        )

        self.v_critical_min = -2.0
        self.v_critical_max = 3.0

        # 混合系数（物理 vs 自适应邻接）
        self.raw_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # 温度系数：固定 buffer = 1.0，消除 sigmoid 饱和问题
        self.register_buffer('temperature', torch.tensor(1.0, dtype=torch.float32))

        # 节点特征投影（用于特征相似度门控）
        self.q_proj = nn.Linear(d_model, 16)

        print(f"  [PCGK] v_critical_init=0.5 (fixed), temperature=1.0 (fixed)")

    @property
    def v_critical(self):
        return torch.clamp(
            self.raw_v_critical,
            min=self.v_critical_min,
            max=self.v_critical_max
        )

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def compute_regime(self, v_obs):
        """
        相态检测：自由流(->1) vs 拥堵流(->0)
        v_obs: (B, N, P) 速度观测（已归一化）
        return: (B, N, 1)
        """
        x = (v_obs - self.v_critical) * self.temperature
        regime = torch.sigmoid(x).mean(dim=-1, keepdim=True)
        return regime

    def forward(self, h_embed, q_obs, A_phys_down, A_phys_up,
                A_adp=None, return_regime=False, v_obs=None):
        """
        Args:
            h_embed:    (B, N, P, d) 嵌入特征
            q_obs:      (B, N, P)    流量观测（用于强度门控）
            A_phys_down:(N, N)       下游物理邻接
            A_phys_up: (N, N)       上游物理邻接
            v_obs:      (B, N, P)    速度观测（用于相态检测）

        Returns:
            (W_up, W_down, regime) 或 (W_up, W_down)
            W_up/W_down: (B, N, N) 边权重（未做行归一化！）
            regime:      (B, N, 1) 相态概率
        """
        B = q_obs.shape[0]
        N = self.num_nodes

        # Step 1: 相态检测
        if v_obs is not None:
            regime = self.compute_regime(v_obs)
        else:
            regime = torch.ones(B, N, 1, device=q_obs.device) * 0.5

        # Step 2: 节点特征相似度
        q_node = h_embed.mean(dim=2)                 # (B, N, d)
        q_feat = self.q_proj(q_node)
        q_feat = F.normalize(q_feat, p=2, dim=-1)

        q_i = q_feat.unsqueeze(2)                    # (B, N, 1, 16)
        q_j = q_feat.unsqueeze(1)                    # (B, 1, N, 16)
        feat_sim = (q_i * q_j).sum(dim=-1)           # (B, N, N)

        # Step 3: 流量强度门控
        q_strength = q_obs.mean(dim=-1, keepdim=True)  # (B, N, 1)
        q_i_s = q_strength.unsqueeze(2)              # (B, N, 1, 1)
        q_j_s = q_strength.unsqueeze(1)                # (B, 1, N, 1)
        min_q = torch.minimum(q_i_s.squeeze(-1), q_j_s.squeeze(-1))
        max_q = torch.maximum(q_i_s.squeeze(-1), q_j_s.squeeze(-1))
        intensity_sim = min_q / (max_q + 1e-6)         # (B, N, N)

        # 综合系数
        c = 0.5 + 0.3 * feat_sim + 0.2 * intensity_sim   # (B, N, N)

        # Step 4: 拓扑掩码（只保留物理存在的边）
        A_mask = (A_phys_down + A_phys_up).clamp(0, 1)   # (N, N)

        # Step 5: 上游/下游边权重（无 softmax！无行归一化！）
        W_up = A_phys_up.unsqueeze(0) * c * A_mask.unsqueeze(0)      # (B, N, N)
        W_down = A_phys_down.unsqueeze(0) * c * A_mask.unsqueeze(0)  # (B, N, N)

        # Step 6: 与自适应邻接混合（如果有）
        if A_adp is not None:
            W_up = self.alpha * W_up + (1.0 - self.alpha) * A_adp
            W_down = self.alpha * W_down + (1.0 - self.alpha) * A_adp

        return (W_up, W_down, regime) if return_regime else (W_up, W_down)

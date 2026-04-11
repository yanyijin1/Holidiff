"""
时空联合门控 - STJointLayer

特征：
- 时序分支：TemporalAttention
- 空间分支：SpatialPropagation
- 向量级门控：每维度独立决定"微/宏"属性

数学形式：
- gate = σ(W_g · Concat[H_T, H_S])
- H_new = H + gate ⊗ H_T + (1-gate) ⊗ H_S
"""
import torch
import torch.nn as nn
from .temporal import TemporalAttention
from .spatial import SpatialPropagation


class STJointLayer(nn.Module):
    """
    时空联合层
    
    整合时序分支 + 空间分支 + 向量级门控
    
    输入: (B, N, P, d)
    输出: (B, N, P, d)
    """
    def __init__(self, d_model, num_heads, num_nodes, adj_path=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 时序分支
        self.temporal_branch = TemporalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 空间分支
        self.spatial_branch = SpatialPropagation(
            d_model=d_model,
            num_nodes=num_nodes,
            adj_path=adj_path,
            dropout=dropout
        )
        
        # 门控投影
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, h):
        """
        Args:
            h: (B, N, P, d) 输入特征
        Returns:
            out: (B, N, P, d) 输出特征
        """
        # ========== 时序分支 ==========
        h_t = self.temporal_branch(h)
        
        # ========== 空间分支 ==========
        h_s = self.spatial_branch(h)
        
        # ========== 向量级门控 ==========
        # Concat: (B, N, P, 2d) -> (B, N, P, d)
        gate = self.gate_proj(torch.cat([h_t, h_s], dim=-1))
        
        # H_new = H + gate * H_T + (1 - gate) * H_S
        out = h + gate * h_t + (1 - gate) * h_s
        
        return out

"""
空间分支 - SpatialPropagation

特征：
- D @ Q 拓扑 LWR 传播
- 无需距离信息，纯拓扑
- 逐 patch 独立处理

D 矩阵物理意义：
- D = I - S^T
- S^T: 下游传播权重（上游客流流向）
- D @ q: 上游 deficit（拥堵波沿拓扑边上行传播）
"""
import torch
import torch.nn as nn
from utils.graph import build_d_matrix_from_adjacency


class SpatialPropagation(nn.Module):
    """
    空间传播分支
    
    核心：D @ Q 操作（LWR 物理先验）
    
    输入: (B, N, P, d) - B=batch, N=节点数, P=patch数, d=维度
    输出: (B, N, P, d)
    
    沿 N 维度做空间传播，保持 P 维度独立
    """
    def __init__(self, d_model, num_nodes, adj_path=None, alpha=0.1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.alpha = alpha  # 空间传播强度（可学习）
        
        # Q 投影
        self.q_proj = nn.Linear(d_model, d_model)
        
        # 更新投影
        self.update = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 可学习的 alpha
        self.spatial_alpha = nn.Parameter(torch.tensor(0.1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # D 矩阵（拓扑 LWR 算子）
        if adj_path is not None:
            D = build_d_matrix_from_adjacency(adj_path, num_nodes)
            self.register_buffer('D', D)
        else:
            # 如果没有图结构，用单位矩阵（纯时序）
            self.register_buffer('D', torch.eye(num_nodes))
    
    def forward(self, h):
        """
        Args:
            h: (B, N, P, d) 输入特征
        Returns:
            out: (B, N, P, d) 输出特征
        """
        B, N, P, d = h.shape
        
        # ========== Q 投影 ==========
        q = self.q_proj(h)  # (B, N, P, d)
        
        # ========== D @ Q（沿 N 维度传播）==========
        # D: (N, N), q: (B, N, P, d) -> (B, N, P, d)
        # einsum: 'nm,bnpd->bmpd' 表示 D 每个节点对 q 的加权
        q_spatial = torch.einsum('nm,bnpd->bmpd', self.D, q)
        
        # ========== 更新投影 ==========
        delta_h = self.update(q_spatial)
        
        # ========== 残差 + alpha ==========
        # alpha 控制空间传播强度
        alpha = torch.sigmoid(self.spatial_alpha)
        out = h + alpha * self.dropout(delta_h)
        
        return out

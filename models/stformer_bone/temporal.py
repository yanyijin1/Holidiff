"""
时序分支 - TemporalAttention

特征：
- Pre-LN 结构（更稳定）
- RotaryEmbedding（无因果掩码）
- 全注意力（纯扩散，不限制时序）
"""
import torch
import torch.nn as nn
import math
from layers.rotaryembedding import RotaryEmbedding


class TemporalAttention(nn.Module):
    """
    时序注意力分支
    
    架构：Pre-LN → RotaryAttn → Output Projection
    
    与原 Attenion 的区别：
    1. 无因果掩码（纯扩散，注意力全局）
    2. Pre-LN 结构（更稳定）
    3. 只做时序维度 Attention（patch 内）
    
    输入: (B, N, P, d) - B=batch, N=节点数, P=patch数, d=维度
    输出: (B, N, P, d)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 确保维度可整除
        assert d_model % num_heads == 0, f"d_model={d_model} 不能被 num_heads={num_heads} 整除"
        
        # QKV 投影
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        
        # 输出投影
        self.proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Rotary Embedding
        self.rotary = RotaryEmbedding(dim=self.head_dim // 2)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Dropout for residual
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, P, d) 输入特征
        Returns:
            out: (B, N, P, d) 输出特征
        """
        B, N, P, d = x.shape
        
        # ========== Pre-LN ==========
        x_norm = x
        
        # ========== QKV 投影 ==========
        # (B, N, P, d) -> (B, N, P, 3*d) -> (B, N, P, 3, num_heads, head_dim)
        qkv = self.qkv(x_norm).reshape(B, N, P, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # (3, B, N, H, P, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 (B, N, H, P, head_dim)
        
        # ========== Rotary Embedding ==========
        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)
        
        # ========== 注意力计算（无掩码） ==========
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Flash Attention（更快）
            # reshape: (B, N, H, P, head_dim) -> (B*N*H, P, head_dim)
            q_flat = q.reshape(B * N * self.num_heads, P, self.head_dim)
            k_flat = k.reshape(B * N * self.num_heads, P, self.head_dim)
            v_flat = v.reshape(B * N * self.num_heads, P, self.head_dim)
            
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q_flat, k_flat, v_flat, dropout_p=self.dropout.p if self.training else 0.0
            )
            attn_out = attn_out.reshape(B, N, self.num_heads, P, self.head_dim)
        else:
            # 手动计算
            scale = 1.0 / math.sqrt(self.head_dim)
            q_flat = q.reshape(B * N * self.num_heads, P, self.head_dim)
            k_flat = k.reshape(B * N * self.num_heads, self.head_dim, P)
            v_flat = v.reshape(B * N * self.num_heads, P, self.head_dim)
            
            attn = torch.bmm(q_flat, k_flat) * scale  # (B*N*H, P, P)
            attn = torch.softmax(attn, dim=-1)
            attn_out = torch.bmm(attn, v_flat)  # (B*N*H, P, head_dim)
            attn_out = attn_out.reshape(B, N, self.num_heads, P, self.head_dim)
        
        # reshape: (B, N, H, P, head_dim) -> (B, N, P, d)
        attn_out = attn_out.permute(0, 1, 3, 2, 4).reshape(B, N, P, d)
        attn_out = self.dropout(attn_out)
        
        # ========== 输出投影 + 残差 ==========
        out = x + self.dropout2(self.proj(attn_out))
        
        # ========== FFN + 残差 ==========
        out = out + self.dropout3(self.ffn(out))
        
        return out

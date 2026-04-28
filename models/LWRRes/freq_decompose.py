"""
Frequency Decomposition Module

将输入序列分解为三个频带：
- Low frequency (趋势): 指数滑动平均，捕捉宏观趋势
- Mid frequency (周期): 去除趋势后的中频成分
- High frequency (波动): 去除低频+中频后的残差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqDecomposer(nn.Module):
    """
    频域分解器：输入 -> (low_freq, mid_freq, high_freq)

    使用固定系数（不参与梯度），避免 EMA 循环的 in-place 版本冲突
    """
    def __init__(self, seq_len, alpha_low=0.1, alpha_mid=0.3):
        super().__init__()
        self.alpha_low = alpha_low
        self.alpha_mid = alpha_mid

    def forward(self, x):
        """
        Args:
            x: (B*N, T) 输入序列
        Returns:
            low: (B*N, T) 低频趋势
            mid: (B*N, T) 中频周期
            high: (B*N, T) 高频波动
        """
        # ========== 低频：趋势 ==========
        low = self._ema_smooth(x, self.alpha_low)

        # ========== 中频：去除趋势后的中频 ==========
        detrended = x - low
        mid = self._ema_smooth(detrended, self.alpha_mid)

        # ========== 高频：去除低频+中频 ==========
        high = x - low - mid

        return low, mid, high

    def _ema_smooth(self, x, alpha):
        """
        指数滑动平均（纯循环，无 in-place）
        alpha 是固定值，不参与梯度
        """
        T = x.shape[1]
        y = torch.empty_like(x)
        y[:, 0] = x[:, 0]
        for t in range(1, T):
            y[:, t] = alpha * x[:, t] + (1 - alpha) * y[:, t - 1]
        return y


class FreqFusion(nn.Module):
    """
    频带融合层：将三个频带的特征融合为统一表示
    
    方案：加权求和 + 可学习的融合权重
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 三路独立的投影（保持各自特性）
        self.low_proj = nn.Linear(d_model, d_model)
        self.mid_proj = nn.Linear(d_model, d_model)
        self.high_proj = nn.Linear(d_model, d_model)
        
        # 可学习的融合权重（频带重要性）
        self.freq_weights = nn.Parameter(torch.ones(3) / 3)  # 初始均匀
        
        self.act = nn.GELU()
    
    def forward(self, low_feat, mid_feat, high_feat):
        """
        Args:
            low_feat: (B, N, P, d) 低频特征
            mid_feat: (B, N, P, d) 中频特征
            high_feat: (B, N, P, d) 高频特征
        Returns:
            fused: (B, N, P, d) 融合后的特征
        """
        # 各路投影 + activation
        low = self.act(self.low_proj(low_feat))
        mid = self.act(self.mid_proj(mid_feat))
        high = self.act(self.high_proj(high_feat))
        
        # 归一化权重
        weights = F.softmax(self.freq_weights, dim=0)  # (3,)
        
        # 加权融合
        fused = weights[0] * low + weights[1] * mid + weights[2] * high
        
        return fused

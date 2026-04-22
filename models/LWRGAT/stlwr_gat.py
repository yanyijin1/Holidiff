"""
ST-LWR-GAT: 物理约束时空图注意力网络

核心设计：
1. PCGK: 物理计算图核，用速度阈值判断相态，决定信息传播方向
2. 空间GAT + 时间Attn: 串行时空传播
3. 差分形式更新: H_out = H_gat + beta * (H_temp - H_gat)
4. v_critical 可学习，自动适配数据集
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== PCGK: 物理计算图核 ==============
class PhysicsComputedGraphKernel(nn.Module):
    """
    PCGK: Physics-Computed Graph Kernel
    
    核心设计：
    - 相态判断：v > v_critical → 自由流；v ≤ v_critical → 拥堵流
    - 图核方向：自由流加强下游连接，拥堵流加强上游连接
    - v_critical 可学习，自动适配数据集
    """

    def __init__(self, d_model, num_nodes, v_critical_init=30.0):
        super().__init__()

        self.num_nodes = num_nodes

        # v_critical 可学习（用 softplus 保证 > 5 km/h）
        self.raw_v_critical = nn.Parameter(torch.tensor(v_critical_init - 5.0))
        self.v_critical_min = 5.0  # 最小临界速度

        # 混合系数（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.7))

        # q 嵌入投影（用于节点特征相似度）
        self.q_proj = nn.Linear(d_model, 16)

        print(f"  [PCGK] v_critical_init={v_critical_init} (learnable)")

    @property
    def v_critical(self):
        """可学习的临界速度（softplus 保证 > 5 km/h）"""
        return F.softplus(self.raw_v_critical) + self.v_critical_min

    def compute_regime(self, v_obs):
        """
        计算相态：自由流 vs 拥堵流
        
        Args:
            v_obs: (B, N, P) 速度观测
        
        Returns:
            regime: (B, N, 1) 相态概率，自由流=1，拥堵流=0
        """
        v_clipped = torch.clamp(v_obs, 0, 150)  # 速度不超过 150 km/h
        regime = (v_clipped > self.v_critical).float().mean(dim=-1, keepdim=True)
        return regime

    def forward(self, q_embed, q_obs, A_phys_down, A_phys_up, A_adp=None, return_regime=False, v_obs=None):
        """
        Args:
            q_embed: (B, N, P, d) 嵌入特征
            q_obs: (B, N, P) 流量观测
            A_phys_down: (N, N) 下游物理邻接
            A_phys_up: (N, N) 上游物理邻接
            return_regime: 是否返回相态
            v_obs: (B, N, P) 速度观测
        
        Returns:
            A_eff: (B, N, N) 有效邻接矩阵
            regime: (B, N, 1) 相态（仅当 return_regime=True 时）
        """
        B = q_obs.shape[0]
        N, P = self.num_nodes, q_obs.shape[-1]

        # ========== Step 1: 相态检测 ==========
        if v_obs is not None:
            regime = self.compute_regime(v_obs)
        else:
            # Fallback：偏向自由流
            regime = torch.ones(B, N, 1, device=q_obs.device) * 0.6

        # ========== Step 2: 节点特征相似度 ==========
        q_node = q_embed.mean(dim=2)
        q_feat = self.q_proj(q_node)
        q_feat = F.normalize(q_feat, p=2, dim=-1)
        
        q_i = q_feat.unsqueeze(2)
        q_j = q_feat.unsqueeze(1)
        feat_sim = (q_i * q_j).sum(dim=-1)

        # ========== Step 3: 相态方向加权 ==========
        downstream_weight = regime * A_phys_down.unsqueeze(0)
        upstream_weight = (1 - regime) * A_phys_up.unsqueeze(0)
        direction_weight = downstream_weight + upstream_weight

        # ========== Step 4: 流量强度门控 ==========
        q_strength = q_obs.mean(dim=-1, keepdim=True)
        q_strength_i = q_strength.unsqueeze(2)
        q_strength_j = q_strength.unsqueeze(1)

        intensity_sim = torch.minimum(q_strength_i.squeeze(-1), q_strength_j.squeeze(-1)) / \
                       (torch.maximum(q_strength_i.squeeze(-1), q_strength_j.squeeze(-1)) + 1e-6)
        intensity_sim = intensity_sim.squeeze(-1)

        # 合并：方向 × 特征相似度 × 强度相似度
        w_kernel = direction_weight * (0.5 + 0.3 * feat_sim + 0.2 * intensity_sim)

        # ========== Step 5: 拓扑掩码 + 归一化 ==========
        A_mask = (A_phys_down + A_phys_up).clamp(0, 1)
        w_kernel = w_kernel * A_mask.unsqueeze(0)
        w_kernel = w_kernel.masked_fill(w_kernel == 0, -1e9)
        A_kernel = F.softmax(w_kernel, dim=-1)
        A_kernel = A_kernel * A_mask.unsqueeze(0)

        # ========== Step 6: 可学习混合 ==========
        alpha = torch.sigmoid(self.alpha)
        if A_adp is not None:
            A_eff = alpha * A_kernel + (1 - alpha) * A_adp.unsqueeze(0)
        else:
            A_eff = A_kernel

        return (A_eff, regime) if return_regime else A_eff


# ============== 空间 GAT 层 ==============
class SpatialGATLayer(nn.Module):
    """空间 GAT 层 - 向量化版本"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.log_A_weight = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, h, A_graph):
        """
        Args:
            h: (B, N, P, d) 输入特征
            A_graph: (B, N, N) 图邻接矩阵
        
        Returns:
            h_out: (B, N, P, d) 更新后特征
        """
        B, N, P, d = h.shape
        A_graph = A_graph.to(h.device)

        # === 1. QKV 投影 ===
        q = self.W_q(h)  # (B, N, P, d)
        k = self.W_k(h)
        v = self.W_v(h)

        # === 2. reshape: (B, N, P, d) → (B*P, N, d) → (B*P, H, N, d_h) ===
        # 把 Patch 维度 batch 化，消除循环
        q = q.permute(0, 2, 1, 3).reshape(B * P, N, d)   # (B*P, N, d)
        k = k.permute(0, 2, 1, 3).reshape(B * P, N, d)
        v = v.permute(0, 2, 1, 3).reshape(B * P, N, d)

        q = q.view(B * P, N, self.n_heads, self.d_head).transpose(1, 2)   # (B*P, H, N, d_h)
        k = k.view(B * P, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B * P, N, self.n_heads, self.d_head).transpose(1, 2)

        # === 3. attention: (B*P, H, N, N) ===
        scale = self.d_head ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale

        # === 4. 物理图核先验 ===
        # A_graph: (B, N, N) → (B*P, 1, N, N)
        A_exp = A_graph.unsqueeze(1).repeat(1, P, 1, 1).reshape(B * P, 1, N, N)
        log_A = torch.log(A_exp + 1e-10)
        attn = attn + torch.sigmoid(self.log_A_weight) * log_A

        # === 5. mask + softmax ===
        mask = (A_exp > 0).expand(-1, self.n_heads, -1, -1)
        attn = attn.masked_fill(~mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn) * mask.float()

        # === 6. 聚合 + 恢复形状 ===
        h_agg = (attn @ v).transpose(1, 2).reshape(B * P, N, d)   # (B*P, N, d)
        h_agg = self.W_o(h_agg)

        # (B*P, N, d) → (B, P, N, d) → (B, N, P, d)
        h_out = h_agg.reshape(B, P, N, d).permute(0, 2, 1, 3)

        # === 7. 残差 ===
        beta = torch.sigmoid(self.beta)
        return h + beta * (h_out - h)


# ============== 时间注意力层 ==============
class TemporalAttentionLayer(nn.Module):
    """时间注意力层"""

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
        """h: (B, N, P, d) → (B, N, P, d)"""
        B, N, P, d = h.shape

        h_2d = h.transpose(1, 2).reshape(B * N, P, d)

        q = self.W_q(h_2d).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(h_2d).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(h_2d).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)

        scale = self.d_head ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        h_out_2d = (attn @ v).transpose(1, 2).reshape(B * N, P, d)
        h_out_2d = self.W_o(h_out_2d)

        h_out = h_out_2d.reshape(B, N, P, d)

        return h_out


# ============== STLWRGAT 层 ==============
class STLWRGATLayer(nn.Module):
    """
    STLWRGATLayer - 单层时空 LWR 约束 GAT
    
    架构：空间 GAT → 时间 Attn → 差分更新
    """

    def __init__(self, d_model, n_heads, num_nodes, dropout=0.1):
        super().__init__()

        self.pcgk = PhysicsComputedGraphKernel(d_model, num_nodes)
        self.spatial_gat = SpatialGATLayer(d_model, n_heads, dropout)
        self.temporal_attn = TemporalAttentionLayer(d_model, n_heads, dropout)

        # 可学习的时间 refine 系数（差分形式）
        self.beta_temp = nn.Parameter(torch.tensor(0.2))

        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  [STLWRGATLayer] params: {total_params:,}")

    def forward(self, h, A_phys_down, A_phys_up,
                cached_A_kernel=None, cached_regime=None, q_obs=None, v_obs=None):
        """
        Args:
            h: (B, N, P, d) 输入嵌入
            A_phys_down: (N, N) 下游物理邻接
            A_phys_up: (N, N) 上游物理邻接
            cached_A_kernel: (B, N, N) 预计算的图核, None 则重新计算
            cached_regime: (B, N, 1) 预计算的相态, None 则重新计算
            q_obs: (B, N, P) 流量观测，如果传入则使用，否则从 h 提取
            v_obs: (B, N, P) 速度观测，如果传入则使用，否则从 h 提取
        
        Returns:
            h_out: (B, N, P, d) 更新后特征
            A_kernel: (B, N, N) 物理图核
            regime: (B, N, 1) 相态
        """
        B, N, P, d = h.shape
        
        # 从输入嵌入提取代理信号（如果没有传入）
        if q_obs is None or v_obs is None:
            q_strength = h.norm(p=2, dim=-1).mean(dim=-1, keepdim=True)
            v_strength = h.norm(p=2, dim=-1).mean(dim=-1, keepdim=True)
            q_obs = q_strength.expand(B, N, P)
            v_obs = v_strength.expand(B, N, P)
        
        # ========== Step 1: PCGK (支持缓存) ==========
        if cached_A_kernel is not None and cached_regime is not None:
            # 测试模式: 直接复用缓存
            A_kernel = cached_A_kernel
            regime = cached_regime
        else:
            # 训练模式: 重新计算
            A_kernel, regime = self.pcgk(h, q_obs, A_phys_down, A_phys_up, return_regime=True, v_obs=v_obs)
        
        # ========== Step 2: 空间 GAT ==========
        H_gat = self.spatial_gat(h, A_kernel)
        
        # ========== Step 3: 时间 Attention ==========
        H_temp = self.temporal_attn(H_gat)
        
        # ========== Step 4: 差分形式更新 ==========
        beta_temp = torch.sigmoid(self.beta_temp)
        H_diff = H_temp - H_gat
        H_out = H_gat + beta_temp * H_diff
        
        # ========== Step 5: FFN + Dropout ==========
        H_out = self.norm(H_out)
        H_out = H_out + self.ffn(H_out)
        H_out = self.dropout(H_out)

        return H_out, A_kernel, regime


# ============== 显式分解输出层 ==============
class PhysicsDecoupledOutput(nn.Module):
    """显式分解输出层 - pred_lwr + pred_res"""

    def __init__(self, d_model, pred_len, res_lr_factor=0.5):
        super().__init__()
        self.pred_len = pred_len
        self.res_lr_factor = res_lr_factor

        self.proj_lwr = nn.Linear(d_model, pred_len)
        self.proj_res = nn.Linear(d_model, pred_len)

        nn.init.zeros_(self.proj_lwr.weight)
        nn.init.zeros_(self.proj_lwr.bias)

    def forward(self, H_v):
        """
        Args:
            H_v: (B, N, P, d) 特征
        
        Returns:
            pred: (B, N, pred_len) 总预测
            pred_lwr: (B, N, pred_len) 物理骨架预测
            pred_res: (B, N, pred_len) 残差修正
        """
        h_v = H_v.mean(dim=2)

        pred_lwr = self.proj_lwr(h_v)
        pred_res = self.proj_res(h_v)

        return pred_lwr + pred_res, pred_lwr, pred_res


# ============== 双流模型 ==============
class DualStreamSTLWRGAT(nn.Module):
    """Dual-Stream Physics-Guided GAT"""

    def __init__(self, d_model, n_heads, num_nodes, pred_len, n_layers=2, dropout=0.1):
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_nodes = num_nodes
        self.pred_len = pred_len

        self.layers = nn.ModuleList([
            STLWRGATLayer(d_model, n_heads, num_nodes, dropout)
            for _ in range(n_layers)
        ])

        self.output = PhysicsDecoupledOutput(d_model, pred_len)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  [DualStreamSTLWRGAT] layers={n_layers}, params={total_params:,}")

    def precompute_pcgk(self, h_embed, q_obs, v_obs, A_phys_down, A_phys_up):
        """
        测试前置: 预计算所有层的 PCGK 图核
        返回每层缓存的 (A_kernel, regime)
        
        Args:
            h_embed: (B, N, P, d) 嵌入特征（从条件历史计算得到）
            q_obs: (B, N, P) 流量观测
            v_obs: (B, N, P) 速度观测
            A_phys_down: (N, N) 下游物理邻接
            A_phys_up: (N, N) 上游物理邻接
        
        Returns:
            pcgk_cache: list of [(A_k, regime), ...] 每层一个
        """
        cached = []
        H = h_embed
        for layer in self.layers:
            # 直接调用 PCGK 计算（不经过 spatial/temporal）
            A_k, regime = layer.pcgk(H, q_obs, A_phys_down, A_phys_up, return_regime=True, v_obs=v_obs)
            cached.append((A_k, regime))
        return cached

    def forward(self, v_embed, q_obs, v_obs, A_phys_down, A_phys_up,
                pcgk_cache=None):
        """
        Args:
            v_embed: (B, N, P, d) 速度嵌入
            q_obs: (B, N, P) 流量观测
            v_obs: (B, N, P) 速度观测
            A_phys_down: (N, N) 下游物理邻接
            A_phys_up: (N, N) 上游物理邻接
            pcgk_cache: list of [(A_k, regime), ...] 每层一个, None 则重新计算
        
        Returns:
            pred: (B, N, pred_len) 总预测
            pred_lwr: (B, N, pred_len) 物理骨架预测
            pred_res: (B, N, pred_len) 残差修正
            all_regimes: 每层的相态
            all_A_kernels: 每层的图核
        """
        all_regimes = []
        all_A_kernels = []
        
        H = v_embed
        for i, layer in enumerate(self.layers):
            if pcgk_cache is not None and i < len(pcgk_cache):
                # 测试: 用缓存
                A_k_cached, regime_cached = pcgk_cache[i]
                H, A_kernel, regime = layer(H, A_phys_down, A_phys_up,
                                          cached_A_kernel=A_k_cached,
                                          cached_regime=regime_cached,
                                          q_obs=q_obs, v_obs=v_obs)
            else:
                # 训练: 重新计算
                H, A_kernel, regime = layer(H, A_phys_down, A_phys_up,
                                          q_obs=q_obs, v_obs=v_obs)
            all_regimes.append(regime)
            all_A_kernels.append(A_kernel)
        
        pred, pred_lwr, pred_res = self.output(H)
        
        return pred, pred_lwr, pred_res, all_regimes, all_A_kernels


# ============== 工具函数 ==============
def compute_lwr_residual_ratio(pred, pred_lwr, pred_res):
    """
    计算 LWR 残差比
    
    Returns:
        ratio: |pred_res| / |pred_lwr|
    """
    lwr_norm = pred_lwr.norm()
    res_norm = pred_res.norm()
    return res_norm / (lwr_norm + 1e-6)


def get_physics_summary(model):
    """
    获取物理参数摘要
    
    Args:
        model: DualStreamSTLWRGAT
    
    Returns:
        dict: 物理参数
    """
    layer = model.layers[0]
    return {
        'v_critical': layer.pcgk.v_critical.item(),
        'alpha': torch.sigmoid(layer.pcgk.alpha).item(),
        'beta_temp': torch.sigmoid(layer.beta_temp).item(),
    }

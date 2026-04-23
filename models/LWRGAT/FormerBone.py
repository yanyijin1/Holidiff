"""
STFormerBone - Phase 1: PCGK-GAT

架构：Patch Embed → Time Add → [STLWRGAT × n] → Output
"""
import torch
import torch.nn as nn
from einops import rearrange
from .stlwr_gat import STLWRGATLayer
from utils.graph import build_upstream_downstream_adjacency as build_phys_adjacency


class TimeEmbedding(nn.Module):
    """时间步嵌入 - 加法注入版本"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, timesteps):
        """timesteps: (B*N,), 输出: (B*N, d_model)"""
        half_d = self.d_model // 2
        t = timesteps.float().unsqueeze(-1) / 10000.0
        emb = torch.cat([torch.sin(t), torch.cos(t)], dim=-1).repeat(1, half_d)
        return self.mlp(emb)


class STFormerBone(nn.Module):
    """
    STFormerBone 主干网络 - Phase 1: PCGK-GAT
    
    架构：Patch Embed → Time Add → [STLWRGAT × n] → Output
    """
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.n_layers = getattr(configs, 'st_layers', 2)
        self.num_heads = configs.num_heads
        
        print(f"\n{'='*60}")
        print(f"[STFormerBone-Phase1] Model Config")
        print(f"  patch_len={self.patch_len}, stride={self.stride}, d_model={self.d_model}")
        print(f"  seq_len={self.seq_len}, pred_len={self.pred_len}, enc_in={self.enc_in}")
        print(f"  STLWRGAT layers={self.n_layers}, num_heads={self.num_heads}")
        
        # Patch Embedding
        patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        patch_num_forecast = int((self.pred_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.patch_num_forecast = patch_num_forecast
        
        # 输入投影：patch_len -> d_model
        self.input_proj = nn.Linear(self.patch_len, self.d_model)
        self.input_dropout = nn.Dropout(configs.dropout)
        
        # 时间嵌入（加法注入）
        self.time_embed = TimeEmbedding(self.d_model)
        
        # 图结构配置
        self.graph_enabled = getattr(configs, 'graph_enabled', False)
        self.graph_adj_path = getattr(configs, 'graph_adj_path', None)
        
        # 物理邻接矩阵（从 CSV 构建）
        if self.graph_enabled and self.graph_adj_path is not None:
            A_down, A_up = build_phys_adjacency(self.graph_adj_path, self.enc_in)
            self.register_buffer('A_phys_down', A_down)
            self.register_buffer('A_phys_up', A_up)
            print(f"  Graph: loaded {int(A_down.sum().item())} downstream edges from adjacency")
        
        # STLWRGAT Layer × n (Phase 1: 单层/双层)
        self.st_layers = nn.ModuleList([
            STLWRGATLayer(
                d_model=self.d_model,
                n_heads=self.num_heads,
                num_nodes=self.enc_in,
                dropout=configs.dropout
            )
            for _ in range(self.n_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(patch_num_forecast * self.d_model, self.pred_len)
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total_params:,}")
        print(f"{'='*60}\n")
    
    def get_physical_params(self):
        """获取 PCGK 物理参数"""
        pcgk = self.st_layers[0].pcgk
        return {'v_f': pcgk.v_f.item(), 'k_j': pcgk.k_j.item()}
    
    def precompute_pcgk(self, cond_ts):
        """
        测试前置: 预计算所有层的 PCGK 图核
        
        Args:
            cond_ts: (B*N, seq_len) 条件序列
        
        Returns:
            pcgk_cache: list of [(A_k, regime), ...] 每层一个
        """
        B_N, seq_len = cond_ts.shape
        N = self.enc_in
        B = B_N // N
        
        # Rearrange
        cond_ts = rearrange(cond_ts, '(b n) t -> b n t', n=N)
        
        # Patch Embedding
        cond_patches = cond_ts.unfold(-1, size=self.patch_len, step=self.stride)
        h = self.input_proj(cond_patches)
        
        # 从条件历史提取 q_obs 和 v_obs
        q_obs = cond_ts.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)  # (B, N, P_cond)
        v_obs = q_obs  # 复用
        
        # 预计算每层的 PCGK
        pcgk_cache = []
        H = h
        for st_layer in self.st_layers:
            A_k, regime = st_layer.pcgk(H, q_obs, self.A_phys_down, self.A_phys_up, 
                                        return_regime=True, v_obs=v_obs)
            pcgk_cache.append((A_k, regime))
        
        return pcgk_cache
    
    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, pcgk_cache=None,
                q_obs=None, v_obs=None, **kwargs):
        """
        Forward - Phase 1
        
        Args:
            x: (B*N, pred_len) 输入噪声
            timesteps: (B*N,) 时间步
            cond_ts: (B*N, seq_len) 条件序列
            pcgk_cache: list of [(A_k, regime), ...] 每层一个, None 则重新计算
            q_obs: (B*N, seq_len) 流量观测，用于 PCGK 相态检测
            v_obs: (B*N, seq_len) 速度观测，用于 PCGK 相态检测
        """
        # ========== Step 1: Rearrange ==========
        B_N, T = x.shape
        N = self.enc_in
        B = B_N // N
        
        x = rearrange(x, '(b n) t -> b n t', n=N)
        cond_ts = rearrange(cond_ts, '(b n) t -> b n t', n=N)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=N).unsqueeze(-1)
        
        # ========== Step 2: Patch Embedding ==========
        cond_patches = cond_ts.unfold(-1, size=self.patch_len, step=self.stride)
        x_patches = x.unfold(-1, size=self.patch_len, step=self.stride)
        patches = torch.cat([cond_patches, x_patches], dim=-2)
        
        h = self.input_proj(patches)
        h = self.input_dropout(h)
        
        # ========== Step 3: Time Embedding ==========
        target_device = x.device
        t_flat = timesteps.reshape(B * N).to(target_device)
        time_embed = self.time_embed(t_flat)
        time_embed = time_embed.reshape(B, N, 1, self.d_model).to(target_device)
        h = h + time_embed
        
        # 从外部传入的 q_obs/v_obs 或从条件历史提取
        # q_obs/v_obs: (B*N, seq_len) -> (B, N, P_cond)
        if q_obs is not None:
            q_obs = rearrange(q_obs, '(b n) t -> b n t', n=N)
            q_obs = q_obs.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)  # (B, N, P_cond)
        else:
            q_obs = cond_ts.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)
        
        if v_obs is not None:
            v_obs = rearrange(v_obs, '(b n) t -> b n t', n=N)
            v_obs = v_obs.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)
        else:
            v_obs = cond_ts.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)
        
        # ========== Step 4: STLWRGAT Layers ==========
        all_regimes = []
        for i, st_layer in enumerate(self.st_layers):
            if pcgk_cache is not None and i < len(pcgk_cache):
                # 测试: 用缓存
                A_k_cached, regime_cached = pcgk_cache[i]
                h_out, A_kernel, regime = st_layer(h, self.A_phys_down, self.A_phys_up,
                                                  cached_A_kernel=A_k_cached,
                                                  cached_regime=regime_cached,
                                                  q_obs=q_obs, v_obs=v_obs)
            else:
                # 训练: 重新计算
                h_out, A_kernel, regime = st_layer(h, self.A_phys_down, self.A_phys_up,
                                                  q_obs=q_obs, v_obs=v_obs)
            h = h_out
            all_regimes.append(regime)
        
        self.last_regime = all_regimes[-1]
        
        # ========== Step 5: Output Projection ==========
        P_cond = cond_patches.shape[2]
        h_forecast = h[:, :, P_cond:, :]
        h_flat = h_forecast.reshape(B, N, -1).reshape(B * N, -1)
        z_out = self.output_proj(h_flat)
        
        return z_out


class PatchUVIT_STFormer(nn.Module):
    """
    PatchUVIT Wrapper for STFormerBone - 适配 Model 的前向传播接口
    """
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = STFormerBone(configs)
        self.enc_in = configs.enc_in
    
    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, **kwargs):
        # 修复：传递 q_obs 和 v_obs
        return self.model(x, timesteps, cond_ts, x_mark_enc, **kwargs)
    
    def forward_for_diffusion(self, x, timesteps, cond_ts, x_mark_enc=None, **kwargs):
        """给 DPM-Solver 调用的接口，支持 pcgk_cache"""
        pcgk_cache = kwargs.get('pcgk_cache', None)
        return self.model(x, timesteps, cond_ts, x_mark_enc, pcgk_cache=pcgk_cache)
    
    def precompute_pcgk(self, cond_ts):
        """预计算 PCGK"""
        return self.model.precompute_pcgk(cond_ts)

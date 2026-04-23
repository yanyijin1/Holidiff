"""
STFormerBone - LWR 物理约束版本 (修正版)

架构：Patch Embed → Time Add → [STLWRLayer × n] → Output

关键修正：
- 使用 STLWRLayer 替代原 STLWRGATLayer
- precompute_pcgk 缓存格式改为 ((W_up, W_down), regime)
- 保持与 Model.py / Trainer 的接口兼容
"""

import torch
import torch.nn as nn
from einops import rearrange
from .stlwr_layer import STLWRLayer
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
        half_d = self.d_model // 2
        t = timesteps.float().unsqueeze(-1) / 10000.0
        emb = torch.cat([torch.sin(t), torch.cos(t)], dim=-1).repeat(1, half_d)
        return self.mlp(emb)


class STFormerBone(nn.Module):
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
        self.dropout = configs.dropout

        # Patch 数量
        self.patch_num_cond = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.patch_num_pred = int((self.pred_len - self.patch_len) / self.stride + 1)
        if self.patch_num_pred < 1:
            self.patch_num_pred = 1
        self.total_patches = self.patch_num_cond + self.patch_num_pred

        # 输入投影
        self.input_proj = nn.Linear(self.patch_len, self.d_model)
        self.input_dropout = nn.Dropout(self.dropout)

        # 时间嵌入
        self.time_embed = TimeEmbedding(self.d_model)

        # 图结构
        self.graph_enabled = getattr(configs, 'graph_enabled', False)
        self.graph_adj_path = getattr(configs, 'graph_adj_path', None)

        if self.graph_enabled and self.graph_adj_path is not None:
            A_down, A_up = build_phys_adjacency(self.graph_adj_path, self.enc_in)
            self.register_buffer('A_phys_down', A_down)
            self.register_buffer('A_phys_up', A_up)
            print(f"  [Graph] loaded {int(A_down.sum().item())} downstream edges")

        # STLWRLayers
        self.st_layers = nn.ModuleList([
            STLWRLayer(self.d_model, self.num_heads, self.enc_in, self.dropout)
            for _ in range(self.n_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(self.patch_num_pred * self.d_model, self.pred_len)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[STFormerBone] params: {total_params:,}")

    def get_physical_params(self):
        """获取 PCGK 物理参数（用于监控）"""
        pcgk = self.st_layers[0].pcgk
        return {
            'v_critical': pcgk.v_critical.item(),
            'temperature': pcgk.temperature.item(),
            'alpha': pcgk.alpha.item()
        }

    def precompute_pcgk(self, cond_ts):
        """
        预计算所有层的 PCGK 图核（测试时缓存）
        基于条件历史的初始嵌入，逐层预计算
        """
        B_N, L = cond_ts.shape
        N = self.enc_in
        B = B_N // N

        cond_ts = rearrange(cond_ts, '(b n) t -> b n t', n=N)

        # Patch Embedding
        cond_patches = cond_ts.unfold(-1, size=self.patch_len, step=self.stride)
        h = self.input_proj(cond_patches)

        # 提取 q_obs / v_obs
        q_obs = cond_ts.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)
        v_obs = q_obs

        # 逐层预计算（不经过 spatial/temporal，只算 PCGK）
        cached = []
        for layer in self.st_layers:
            W_up, W_down, regime = layer.pcgk(
                h, q_obs, self.A_phys_down, self.A_phys_up,
                return_regime=True, v_obs=v_obs
            )
            cached.append(((W_up, W_down), regime))

        return cached

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None,
                pcgk_cache=None, q_obs=None, v_obs=None, **kwargs):
        """
        Args:
            x:          (B*N, pred_len)     噪声/输入
            timesteps:  (B*N,)              扩散时间步
            cond_ts:    (B*N, seq_len)      条件历史
            pcgk_cache: list of [((W_up, W_down), regime), ...]
            q_obs:      (B*N, seq_len)      流量观测
            v_obs:      (B*N, seq_len)      速度观测
        """
        B_N, T = x.shape
        N = self.enc_in
        B = B_N // N

        # Rearrange
        x = rearrange(x, '(b n) t -> b n t', n=N)
        cond_ts = rearrange(cond_ts, '(b n) t -> b n t', n=N)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=N).unsqueeze(-1)

        # Patch Embedding
        cond_patches = cond_ts.unfold(-1, size=self.patch_len, step=self.stride)
        x_patches = x.unfold(-1, size=self.patch_len, step=self.stride)
        patches = torch.cat([cond_patches, x_patches], dim=-2)

        h = self.input_proj(patches)
        h = self.input_dropout(h)

        # Time Embedding（加法注入）
        t_flat = timesteps.reshape(B * N).to(x.device)
        time_embed = self.time_embed(t_flat)
        time_embed = time_embed.reshape(B, N, 1, self.d_model).to(x.device)
        h = h + time_embed

        # 处理 q_obs / v_obs
        if q_obs is not None:
            q_obs = rearrange(q_obs, '(b n) t -> b n t', n=N)
            q_obs = q_obs.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)
        else:
            q_obs = cond_ts.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)

        if v_obs is not None:
            v_obs = rearrange(v_obs, '(b n) t -> b n t', n=N)
            v_obs = v_obs.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)
        else:
            v_obs = cond_ts.unfold(-1, size=self.patch_len, step=self.stride).mean(dim=-1)

        # STLWRLayers
        all_regimes = []
        for i, layer in enumerate(self.st_layers):
            if pcgk_cache is not None and i < len(pcgk_cache):
                (W_up_c, W_down_c), regime_c = pcgk_cache[i]
                h, _, regime = layer(
                    h, self.A_phys_down, self.A_phys_up,
                    cached_kernel=(W_up_c, W_down_c),
                    cached_regime=regime_c,
                    q_obs=q_obs, v_obs=v_obs
                )
            else:
                h, _, regime = layer(
                    h, self.A_phys_down, self.A_phys_up,
                    q_obs=q_obs, v_obs=v_obs
                )
            all_regimes.append(regime)

        self.last_regime = all_regimes[-1]

        # Output Projection
        P_cond = self.patch_num_cond
        h_forecast = h[:, :, P_cond:, :]
        h_flat = h_forecast.reshape(B, N, -1).reshape(B * N, -1)
        return self.output_proj(h_flat)


class PatchUVIT_STFormer(nn.Module):
    """包装器，适配 Model 的前向传播接口"""

    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = STFormerBone(configs)
        self.enc_in = configs.enc_in

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, **kwargs):
        return self.model(x, timesteps, cond_ts, x_mark_enc, **kwargs)

    def forward_for_diffusion(self, x, timesteps, cond_ts, x_mark_enc=None, **kwargs):
        """给 DPM-Solver 调用的接口，支持 pcgk_cache"""
        pcgk_cache = kwargs.get('pcgk_cache', None)
        return self.model(x, timesteps, cond_ts, x_mark_enc, pcgk_cache=pcgk_cache)

    def precompute_pcgk(self, cond_ts):
        return self.model.precompute_pcgk(cond_ts)

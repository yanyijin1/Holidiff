"""
U-Net 主干网络模块 - FormerBone 和 PatchUVIT

架构：T-Down × e_layers → T-Mid → T-Up × e_layers → S-LWR → T-Fusion
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from einops import rearrange
from layers.rotaryembedding import RotaryEmbedding
from utils.graph import build_d_matrix_from_adjacency


class Transpose(nn.Module):
    """用于 BatchNorm1d 的转置"""
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.transpose(*self.dims)


class Attenion(nn.Module):
    """SimDiff 风格的 Attention，带 RotaryEmbedding"""
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.num_heads = config.num_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.num_heads
        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear(config.d_model, config.d_model)
        self.norm_post1 = nn.Sequential(
            Transpose((1, 2)), 
            nn.BatchNorm1d(config.d_model), 
            Transpose((1, 2))
        )
        self.norm_attn = nn.Sequential(
            Transpose((1, 2)), 
            nn.BatchNorm1d(config.d_model), 
            Transpose((1, 2))
        )
        self.ff_1 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True)
        )
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim // 2)

    def forward(self, src, *configs, **kwargs):
        B, nvars, H, C = src.shape
        qkv = self.qkv(src).reshape(B, nvars, H, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = (
            qkv[0].reshape(B * nvars, self.num_heads, -1, self.head_dim),
            qkv[1].reshape(B * nvars, self.num_heads, -1, self.head_dim),
            qkv[2].reshape(B * nvars, self.num_heads, -1, self.head_dim)
        )
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1)
            x = torch.matmul(attn, v)
        
        output1 = rearrange(x, '(b n) h e d -> b n e (h d)', b=B)
        src2 = self.ff_1(output1)
        src = src + src2
        src = src.reshape(B * nvars, -1, self.num_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.num_heads * self.head_dim)
        return src


class SpatialLWROperator(nn.Module):
    """S-层：LWR空间传播

    核心思想：在 N 维度做空间传播
    D = I - S^T （后向差分算子）

    输入: (B, N, T, d)
    输出: (B, N, T, d)
    """
    def __init__(self, d_model, num_nodes, dropout=0.1, 
                 graph_enabled=False, graph_adj_path=None, graph_num_nodes=None):
        super().__init__()
        self.num_nodes = num_nodes

        self.q_proj = nn.Linear(d_model, d_model)
        self.update = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        
        # 根据配置决定是否使用图结构
        if graph_enabled and graph_adj_path and graph_num_nodes:
            from utils.graph import build_d_matrix_from_adjacency
            D = build_d_matrix_from_adjacency(graph_adj_path, graph_num_nodes)
            self.register_buffer('D', D)
        else:
            self.register_buffer('D', torch.eye(num_nodes))

    def forward(self, h):
        B, N, T, d = h.shape
        q_tilde = self.q_proj(h)
        D = self.D.to(h.device)
        q_spatial = torch.einsum('nm,bntd->bmtd', D, q_tilde)
        delta_h = self.update(q_spatial)
        h_new = h + torch.sigmoid(self.gate) * delta_h
        return self.dropout(h_new)


class FormerBone(nn.Module):
    """SimDiff 风格的 FormerBone，使用 unfold + Linear

    架构：T-Down × e_layers → T-Mid → T-Up × e_layers → S-LWR → T-Fusion
    """
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        # 计算 patch 数量
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        patch_num_forecast = int((configs.pred_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.patch_num_forecast = patch_num_forecast

        configs.d_ff = configs.d_model * 2

        # SimDiff 风格的输入投影
        self.W_input_projection = nn.Linear(self.patch_len, configs.d_model)
        self.input_dropout = nn.Dropout(configs.dropout)

        # 时间步 token
        self.cls = nn.Sequential(nn.Linear(1, configs.d_model))

        # 输出投影
        self.W_outs = nn.Linear((patch_num + 1 + patch_num_forecast) * configs.d_model, configs.pred_len)

        # SimDiff 风格的 Attention 层级（U-Net 结构）
        self.Attentions_over_token = nn.ModuleList([Attenion(configs) for i in range(configs.e_layers)])
        self.Attentions_over_token_mid = Attenion(configs)
        self.Attentions_over_token_up = nn.ModuleList([Attenion(configs) for i in range(configs.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(configs.d_model * 2, configs.d_model) for i in range(configs.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(configs.skip_dropout) for i in range(configs.e_layers)])
        self.Attentions_dropout_mid = nn.Dropout(configs.skip_dropout)
        self.Attentions_dropout_up = nn.ModuleList([nn.Dropout(configs.skip_dropout) for i in range(configs.e_layers)])
        self.Attentions_norm = nn.ModuleList([
            nn.Sequential(
                Transpose((1, 2)), 
                nn.BatchNorm1d(configs.d_model), 
                Transpose((1, 2))
            )
            for i in range(configs.e_layers)
        ])

        # S-LWR 空间传播层
        self.spatial_layer = SpatialLWROperator(
            d_model=configs.d_model,
            num_nodes=configs.enc_in,
            dropout=configs.dropout,
            graph_enabled=getattr(configs, 'graph_enabled', False),
            graph_adj_path=getattr(configs, 'graph_adj_path', None),
            graph_num_nodes=getattr(configs, 'graph_num_nodes', None)
        )

        # T-Fusion：最后的时间融合层
        self.t_fusion = Attenion(configs)

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):
        """
        架构：T-Down × e_layers → T-Mid → T-Up × e_layers → S-LWR → T-Fusion
        
        Args:
            x: (B*N, T) - 输入噪声序列 (展平后)
            timesteps: (B*N,) - 时间步
            cond_ts: (B*N, L) - 条件序列
        
        Returns:
            z_out: (B*N, pred_len) - 输出预测
        """
        print("\n========== FormerBone.forward ==========")
        print(f"[INPUT] x: {x.shape}, timesteps: {timesteps.shape}, cond_ts: {cond_ts.shape}")
        
        b, c, s = x.shape
        print(f"[AFTER SHAPE] b={b}, c={c}, s={s}")

        # SimDiff 风格：使用 unfold 进行 patch 化
        zcube0 = cond_ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        zcube1 = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        print(f"[UNFOLD] zcube0: {zcube0.shape}, zcube1: {zcube1.shape}")
        zcube = torch.cat([zcube0, zcube1], dim=-2)
        print(f"[CAT] zcube: {zcube.shape}")

        z_embed = self.input_dropout(self.W_input_projection(zcube))
        print(f"[PROJECTION] z_embed: {z_embed.shape}")

        # 时间 token
        time_token = self.cls(timesteps.float())
        print(f"[TIME_TOKEN] time_token: {time_token.shape}")
        z_embed = torch.cat((time_token, z_embed), dim=-2)
        print(f"[CAT TIME] z_embed: {z_embed.shape}")

        inputs = z_embed
        b, c, t, h = inputs.shape
        print(f"[AFTER CAT] b={b}, c={c}, t={t}, h={h}")

        # ========== T-Down：编码阶段 ==========
        skip = []
        for i, (a_2, mlp, drop, norm) in enumerate(zip(
            self.Attentions_over_token, self.Attentions_mlp,
            self.Attentions_dropout, self.Attentions_norm
        )):
            output = a_2(inputs)
            inputs = drop(output)
            skip.append(inputs)

        # ========== T-Mid：最深层 ==========
        inputs = self.Attentions_over_token_mid(inputs)
        inputs = self.Attentions_dropout_mid(inputs)

        # ========== T-Up：解码阶段 + skip connection ==========
        for i, (a_2, mlp, drop, norm) in enumerate(zip(
            self.Attentions_over_token_up, self.Attentions_mlp,
            self.Attentions_dropout_up, self.Attentions_norm
        )):
            prev = skip.pop()
            outputs = drop(mlp(torch.cat((prev, inputs), dim=-1)))
            outputs = norm(outputs.reshape(b * c, t, -1)).reshape(b, c, t, -1)
            output = a_2(inputs)
            inputs = drop(output)

        # ========== S-LWR：空间传播 ==========
        print(f"[BEFORE S-LWR] inputs: {inputs.shape}")
        inputs = self.spatial_layer(inputs)
        print(f"[AFTER S-LWR] inputs: {inputs.shape}")

        # ========== T-Fusion：时间融合 ==========
        print(f"[BEFORE T-FUSION] inputs: {inputs.shape}")
        output = self.t_fusion(inputs)
        print(f"[AFTER T-FUSION] output: {output.shape}")

        # 输出映射
        print(f"[BEFORE W_OUTS] output: {output.shape}")
        z_out = self.W_outs(output[:, :, :, :].reshape(b, c, -1)).reshape(b * c, -1)
        print(f"[FINAL] z_out: {z_out.shape}")

        return z_out


class PatchUVIT(nn.Module):
    """Patch UViT Wrapper - 用于适配 Model 的前向传播接口"""
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = FormerBone(configs)
        self.enc_in = configs.enc_in

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):
        """
        PatchUVIT Wrapper Forward
        输入: x=(B*N, T), timesteps=(B*N,), cond_ts=(B*N, L)
        输出: z_out=(B*N, pred_len)
        """
        print("\n========== PatchUVIT.forward ==========")
        print(f"[INPUT] x: {x.shape}, timesteps: {timesteps.shape}, cond_ts: {cond_ts.shape}")
        
        x = rearrange(x, '(b n) h -> b n h', n=self.enc_in)
        cond_ts = rearrange(cond_ts, '(b n) h -> b n h', n=self.enc_in)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in).unsqueeze(-1).unsqueeze(-1)
        print(f"[AFTER REARRANGE] x: {x.shape}, cond_ts: {cond_ts.shape}, timesteps: {timesteps.shape}")
        
        x = self.model(x, timesteps, cond_ts)
        print(f"[AFTER MODEL] x: {x.shape}")
        return x

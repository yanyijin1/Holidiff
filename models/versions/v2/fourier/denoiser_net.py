# Version: v0.1-denoiser-net
# Date: 2026-04-07
# Description: 去噪网络 Denoiser Network

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SinusoidalStepEmbedding(nn.Module):
    """Sinusoidal embedding for timesteps"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t):
        device = t.device
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class PatchEmbed(nn.Module):
    """Patch embedding layer with padding and sliding window"""
    
    def __init__(self, in_ch, d_model, patch_size=3, stride=1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Linear(in_ch * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        b, n, _, c = x.shape
        pad = self.patch_size - 1
        left = x[:, :, 1:pad + 1, :].flip(dims=[2])
        right = x[:, :, -(pad + 1):-1, :].flip(dims=[2])
        xp = torch.cat([left, x, right], dim=2)

        patches = []
        i = 0
        while i + self.patch_size <= xp.shape[2]:
            p = xp[:, :, i:i + self.patch_size, :].reshape(b, n, self.patch_size * c)
            patches.append(p)
            i += self.stride
        patches = torch.stack(patches, dim=2)
        return self.norm(self.proj(patches))


class PatchUnfold(nn.Module):
    """Reverse operation of PatchEmbed - reconstructs from patches"""
    
    def __init__(self, d_model, patch_size=3, stride=1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.pad = patch_size - 1
        self.proj = nn.Linear(d_model, patch_size)

    def forward(self, x, t_len):
        b, n, p, _ = x.shape
        out = torch.zeros(b, n, t_len, device=x.device, dtype=x.dtype)
        cnt = torch.zeros(t_len, device=x.device, dtype=x.dtype)
        vals = self.proj(x)
        for i in range(p):
            for j in range(self.patch_size):
                ti = i * self.stride + j - self.pad
                if 0 <= ti < t_len:
                    out[:, :, ti] += vals[:, :, i, j]
                    cnt[ti] += 1
        return (out / cnt.clamp(min=1)).unsqueeze(-1)


class AsymmetricALiBiAttention(nn.Module):
    """
    Asymmetric ALiBi (Attention with Linear Biases) attention mechanism
    
    Allows different attention patterns for forward and backward directions.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.log_alpha_fwd = nn.Parameter(torch.zeros(n_heads))
        self.log_alpha_bwd = nn.Parameter(torch.log(torch.ones(n_heads) * 1.5))

    def forward(self, x):
        bn, p, d = x.shape
        qkv = self.qkv(x).reshape(bn, p, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        i = torch.arange(p, device=x.device).unsqueeze(1)
        j = torch.arange(p, device=x.device).unsqueeze(0)
        diff = i - j
        fwd = diff.clamp(min=0)
        bwd = (-diff).clamp(min=0)

        alpha_fwd = self.log_alpha_fwd.exp().view(1, self.n_heads, 1, 1)
        alpha_bwd = self.log_alpha_bwd.exp().view(1, self.n_heads, 1, 1)
        bias = -(alpha_fwd * fwd + alpha_bwd * bwd)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn + bias, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bn, p, d)
        return self.out(out)


class PatchDenoiserBlock(nn.Module):
    """
    Single denoiser block with:
    - Self-attention (ALiBi)
    - Cross-attention (conditioning)
    - Feed-forward network
    - LWR operator
    """
    
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = AsymmetricALiBiAttention(d_model, n_heads, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.drop = nn.Dropout(dropout)
        
        # Import LWR operator from core
        from core.lwr_solver import PatchLWROperatorLayer
        self.lwr_op = PatchLWROperatorLayer(d_model)

    def forward(self, x, h_c, d_op):
        b, n, p, d = x.shape
        xb = x.reshape(b * n, p, d)
        hb = h_c.reshape(b * n, 1, d)
        
        # Self-attention with ALiBi
        xb = xb + self.drop(self.self_attn(self.norm1(xb)))
        
        # Cross-attention with conditioning
        x2, _ = self.cross_attn(self.norm2(xb), hb, hb)
        xb = xb + self.drop(x2)
        
        # Feed-forward
        xb = xb + self.drop(self.ffn(self.norm3(xb)))
        
        # LWR physics operation
        return self.lwr_op(xb.reshape(b, n, p, d), d_op)


class FormerBone(nn.Module):
    """
    Main denoiser network (FormerBone)
    
    Architecture:
    - Patch embedding
    - Timestep embedding
    - Condition projection
    - Multiple denoiser blocks
    - Patch unfolding
    """
    
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.num_heads = configs.num_heads
        self.pred_len = configs.pred_len
        self.cond_len = configs.seq_len
        self.enc_in = configs.enc_in

        # Embedding layers
        self.patch_embed = PatchEmbed(2, self.d_model, self.patch_len, self.stride)
        self.step_emb = SinusoidalStepEmbedding(self.d_model)
        self.step_proj = nn.Linear(self.d_model, self.d_model)
        self.cond_proj = nn.Linear(self.cond_len, self.d_model)
        
        # Denoiser blocks
        self.blocks = nn.ModuleList([
            PatchDenoiserBlock(self.d_model, self.num_heads, self.d_model * 2, configs.dropout)
            for _ in range(self.e_layers)
        ])
        
        # Output layer
        self.patch_unfold = PatchUnfold(self.d_model, self.patch_len, self.stride)

        # D matrix for LWR (spatial difference operator)
        d = -torch.eye(self.enc_in)
        if self.enc_in > 1:
            d[1:, :-1] += torch.eye(self.enc_in - 1)
        self.register_buffer('d_op', d)

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None):
        b, n, l2 = x.shape
        
        # Conditioning from the past
        qt = cond_ts[:, :, -l2:]
        x = torch.stack([x, qt], dim=-1)
        h = self.patch_embed(x)

        # Timestep embedding
        t = timesteps.reshape(b * n)
        t_emb = self.step_proj(self.step_emb(t)).reshape(b, n, 1, self.d_model)
        h = h + t_emb

        # Condition projection
        c = self.cond_proj(cond_ts).unsqueeze(2)
        
        # Process through blocks
        for block in self.blocks:
            h = block(h, c, self.d_op)

        # Unfold patches to get final output
        z_out = self.patch_unfold(h, l2).squeeze(-1)
        return z_out.reshape(b * n, l2)


class PatchUVIT(nn.Module):
    """
    Wrapper for the denoiser network that handles batch reshaping
    """
    
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = FormerBone(configs)
        self.enc_in = configs.enc_in

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):
        # Reshape for per-channel processing
        x = rearrange(x, '(b n) h -> b n h', n=self.enc_in)
        cond_ts = rearrange(cond_ts, '(b n) h -> b n h', n=self.enc_in)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in)
        x = self.model(x, timesteps, cond_ts)
        return x

import math

import torch
import torch.nn as nn
from einops import rearrange

from Holidiff.layers.rotaryembedding import RotaryEmbedding


class TensorTranspose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class LocalPatchEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_projection = nn.Linear(configs.patch_len, configs.d_model)

    def forward(self, patch_tokens):
        return self.input_projection(patch_tokens)


class TCPAttention(nn.Module):
    def __init__(self, config, over_hidden=False, trianable_smooth=False, untoken=False, *configs, **kwargs):
        super().__init__()
        self.over_hidden = over_hidden
        self.untoken = untoken
        self.num_heads = config.num_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.num_heads
        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear(config.d_model, config.d_model)
        self.norm_post1 = nn.Sequential(TensorTranspose(1, 2), nn.BatchNorm1d(config.d_model), TensorTranspose(1, 2))
        self.norm_attn = nn.Sequential(TensorTranspose(1, 2), nn.BatchNorm1d(config.d_model), TensorTranspose(1, 2))
        self.ff_1 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True),
        )
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim // 2)

    def forward(self, src, *configs, **kwargs):
        B, nvars, H, C = src.shape
        qkv = self.qkv(src).reshape(B, nvars, H, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q = qkv[0].reshape(B * nvars, self.num_heads, -1, self.head_dim)
        k = qkv[1].reshape(B * nvars, self.num_heads, -1, self.head_dim)
        v = qkv[2].reshape(B * nvars, self.num_heads, -1, self.head_dim)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        output1 = rearrange(x, '(b n) h e d -> b n e (h d)', b=B)
        src2 = self.ff_1(output1)
        src = src + src2
        src = src.reshape(B * nvars, -1, self.num_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.num_heads * self.head_dim)
        return src


class STEKBackbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.n_blocks = configs.n_b
        self.frequency_conditioner_dim = int(getattr(configs, 'frequency_conditioner_dim', 8))
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        patch_num_forecast = max(1, int((configs.pred_len - self.patch_len) / self.stride + 1))
        self.patch_num = patch_num
        self.patch_num_forecast = patch_num_forecast
        configs.d_ff = configs.d_model * 2
        self.use_sfcn = bool(getattr(configs, 'use_sfcn', True))
        self.patch_embedding = LocalPatchEmbedding(configs)
        self.input_dropout = nn.Dropout(configs.dropout)
        self.cls = nn.Sequential(nn.Linear(1, configs.d_model))
        self.W_outs = nn.Linear((patch_num + 1 + patch_num_forecast) * configs.d_model, configs.pred_len)
        self.Attentions_over_token = nn.ModuleList([TCPAttention(configs) for _ in range(configs.e_layers)])
        self.Attentions_over_token_mid = TCPAttention(configs)
        self.Attentions_over_token_up = nn.ModuleList([TCPAttention(configs) for _ in range(configs.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(configs.d_model * 2, configs.d_model) for _ in range(configs.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(configs.skip_dropout) for _ in range(configs.e_layers)])
        self.Attentions_dropout_mid = nn.Dropout(configs.skip_dropout)
        self.Attentions_dropout_up = nn.ModuleList([nn.Dropout(configs.skip_dropout) for _ in range(configs.e_layers)])
        self.Attentions_norm = nn.ModuleList([
            nn.Sequential(TensorTranspose(1, 2), nn.BatchNorm1d(configs.d_model), TensorTranspose(1, 2))
            for _ in range(configs.e_layers)
        ])

    def forward(self, micro_realization, timesteps, hist_macro_state, x_mark_enc=None, raw_history=None, physical_injection=None, frequency_patch_embedding=None):
        b, c, _ = micro_realization.shape
        if hist_macro_state.shape[-1] < self.patch_len:
            hist_macro_state = torch.nn.functional.pad(hist_macro_state, (0, self.patch_len - hist_macro_state.shape[-1]), mode='replicate')
        if micro_realization.shape[-1] < self.patch_len:
            micro_realization = torch.nn.functional.pad(micro_realization, (0, self.patch_len - micro_realization.shape[-1]), mode='replicate')
        hist_tokens = hist_macro_state.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        future_tokens = micro_realization.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        state_tokens = torch.cat([hist_tokens, future_tokens], dim=-2)
        state_embeddings = self.input_dropout(self.patch_embedding(state_tokens))
        if self.use_sfcn and frequency_patch_embedding is not None:
            if frequency_patch_embedding.size(-2) != state_embeddings.size(-2):
                min_tokens = min(frequency_patch_embedding.size(-2), state_embeddings.size(-2))
                frequency_patch_embedding = frequency_patch_embedding[:, :, :min_tokens, :]
                state_embeddings = state_embeddings[:, :, :min_tokens, :]
            state_embeddings = state_embeddings + frequency_patch_embedding.to(device=state_embeddings.device, dtype=state_embeddings.dtype)
        time_token = self.cls(timesteps.float())
        state_embeddings = torch.cat((time_token, state_embeddings), dim=-2)
        inputs = state_embeddings
        b, c, t, h = inputs.shape
        skip = []
        for attention_layer, _, dropout_layer, _ in zip(self.Attentions_over_token, self.Attentions_mlp, self.Attentions_dropout, self.Attentions_norm):
            output = attention_layer(inputs)
            inputs = dropout_layer(output)
            skip.append(inputs)
        inputs = self.Attentions_over_token_mid(inputs)
        inputs = self.Attentions_dropout_mid(inputs)
        for attention_layer, mlp, dropout_layer, norm in zip(self.Attentions_over_token_up, self.Attentions_mlp, self.Attentions_dropout_up, self.Attentions_norm):
            prev = skip.pop()
            outputs = dropout_layer(mlp(torch.cat((prev, inputs), dim=-1)))
            outputs = norm(outputs.reshape(b * c, t, -1)).reshape(b, c, t, -1)
            output = attention_layer(inputs)
            inputs = dropout_layer(output)
        flat = inputs[:, :, :, :].reshape(b, c, -1)
        target_in = self.W_outs.in_features
        if flat.shape[-1] > target_in:
            flat = flat[..., :target_in]
        elif flat.shape[-1] < target_in:
            flat = torch.nn.functional.pad(flat, (0, target_in - flat.shape[-1]))
        micro_estimate = self.W_outs(flat).reshape(b * c, -1)
        return micro_estimate
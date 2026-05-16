import torch
import torch.nn as nn
from einops import rearrange

from Holidiff.layers.rotaryembedding import RotaryEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class Attenion(nn.Module):
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
        self.norm_post1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
        self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
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
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        output1 = rearrange(x, '(b n) h e d -> b n e (h d)', b=B)
        src2 = self.ff_1(output1)
        src = src + src2
        src = src.reshape(B * nvars, -1, self.num_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.num_heads * self.head_dim)
        return src


class FormerBone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.n_blocks = configs.n_b
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        patch_num_forecast = max(1, int((configs.pred_len - self.patch_len) / self.stride + 1))
        self.patch_num = patch_num
        self.patch_num_forecast = patch_num_forecast
        configs.d_ff = configs.d_model * 2
        self.W_input_projection = nn.Linear(self.patch_len, configs.d_model)
        self.input_dropout = nn.Dropout(configs.dropout)
        self.cls = nn.Sequential(nn.Linear(1, configs.d_model))
        self.W_outs = nn.Linear((patch_num + 1 + patch_num_forecast) * configs.d_model, configs.pred_len)
        self.Attentions_over_token = nn.ModuleList([Attenion(configs) for _ in range(configs.e_layers)])
        self.Attentions_over_token_mid = Attenion(configs)
        self.Attentions_over_token_up = nn.ModuleList([Attenion(configs) for _ in range(configs.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(configs.d_model * 2, configs.d_model) for _ in range(configs.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(configs.skip_dropout) for _ in range(configs.e_layers)])
        self.Attentions_dropout_mid = nn.Dropout(configs.skip_dropout)
        self.Attentions_dropout_up = nn.ModuleList([nn.Dropout(configs.skip_dropout) for _ in range(configs.e_layers)])
        self.Attentions_norm = nn.ModuleList([
            nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
            for _ in range(configs.e_layers)
        ])

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None):
        b, c, s = x.shape
        if cond_ts.shape[-1] < self.patch_len:
            cond_ts = torch.nn.functional.pad(cond_ts, (0, self.patch_len - cond_ts.shape[-1]), mode='replicate')
        if x.shape[-1] < self.patch_len:
            x = torch.nn.functional.pad(x, (0, self.patch_len - x.shape[-1]), mode='replicate')
        zcube0 = cond_ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        zcube1 = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        zcube = torch.cat([zcube0, zcube1], dim=-2)
        z_embed = self.input_dropout(self.W_input_projection(zcube))
        time_token = self.cls(timesteps.float())
        z_embed = torch.cat((time_token, z_embed), dim=-2)
        inputs = z_embed
        b, c, t, h = inputs.shape
        skip = []
        for a_2, mlp, drop, norm in zip(self.Attentions_over_token, self.Attentions_mlp, self.Attentions_dropout, self.Attentions_norm):
            output = a_2(inputs)
            inputs = drop(output)
            skip.append(inputs)
        inputs = self.Attentions_over_token_mid(inputs)
        inputs = self.Attentions_dropout_mid(inputs)
        for a_2, mlp, drop, norm in zip(self.Attentions_over_token_up, self.Attentions_mlp, self.Attentions_dropout, self.Attentions_norm):
            prev = skip.pop()
            outputs = drop(mlp(torch.cat((prev, inputs), dim=-1)))
            outputs = norm(outputs.reshape(b * c, t, -1)).reshape(b, c, t, -1)
            output = a_2(inputs)
            inputs = drop(output)
        flat = inputs[:, :, :, :].reshape(b, c, -1)
        target_in = self.W_outs.in_features
        if flat.shape[-1] > target_in:
            flat = flat[..., :target_in]
        elif flat.shape[-1] < target_in:
            flat = torch.nn.functional.pad(flat, (0, target_in - flat.shape[-1]))
        z_out = self.W_outs(flat).reshape(b * c, -1)
        return z_out

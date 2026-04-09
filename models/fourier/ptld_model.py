import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
import torch.nn as nn
from torch import nn, einsum
from torch.nn.modules import loss
from einops import rearrange, repeat
import math

from layers.RevIN import RevIN
from layers.rotaryembedding import RotaryEmbedding
from layers.samplers.dpm_sampler import DPMSolverSampler
from utils.diffusion_utils import *



def cosine_beta_schedule(timesteps, s=5):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.diff_steps = configs.diff_steps
        self.stride=configs.stride
        self.patch_len=configs.patch_len
        self.d_model=configs.d_model
        self.e_layers=configs.e_layers
        self.num_heads=configs.num_heads
        self.rmom_n = configs.rmom
        self.n_blocks = configs.n_b
        u_net = PatchUVIT(configs)
            
        self.enc_in = configs.enc_in
        self.batch_size =configs.batch_size

        self.beta_start = 1e-4 # 1e4
        self.beta_end = 1e-1#2e-2
        self.beta_schedule = 'cosine'
        self.v_posterior = 0.0
        self.loss_type = "l1"
        self.set_new_noise_schedule(None, self.beta_schedule, self.diff_steps, self.beta_start, self.beta_end)
        self.total_N = len(self.alphas_cumprod)
        self.T = 1.
        self.eps = 1e-5
        self.nn = u_net
        self.sampler = DPMSolverSampler(configs,self.nn, self.device,self.alphas_cumprod,self.betas.device)
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):
        if self.training:
            return self.forward_train(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                             enc_self_mask, dec_self_mask, dec_enc_mask)
        else:
            return self.forward_val_test(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                            enc_self_mask, dec_self_mask, dec_enc_mask, sample_times)


    def forward_train(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # === 1. 取 target（只取 pred_len，与 SimDiff 完全一致）===
        x_target = x_dec[:, -self.configs.pred_len:, :]   # (B, 12, N)

        # === 2. 归一化 ===
        x = x_target.permute(0, 2, 1)                    # (B, N, 12)
        cond_ts = x_enc.permute(0, 2, 1)                 # (B, N, 96)

        mean_ = cond_ts.mean(dim=(1, 2), keepdim=True)
        std_  = cond_ts.std(dim=(1, 2), keepdim=True) + 1e-5
        x_norm = (x - mean_) / std_                      # (B, N, 12)
        cond_norm = (cond_ts - mean_) / std_            # (B, N, 96)

        # === 3. 加噪 ===
        B, N, dec_L = x_norm.shape   # B=32, N=30, dec_L=12
        t = torch.randint(0, self.num_timesteps,
                          size=[B * N // 2]).long().to(self.device)
        t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)  # (B*N,)
        noise = torch.randn_like(x_norm)  # (B, N, 12)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(B, N, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].reshape(B, N, 1)
        x_k = sqrt_alpha * x_norm + sqrt_one_minus * noise   # (B, N, 12)

        # === 4. FormerBone_TST 前向 ===
        x_k_flat = x_k.reshape(B * N, dec_L)                # (B*N, 12)
        cond_flat = cond_norm.reshape(B * N, -1)            # (B*N, 96)
        model_out_flat = self.nn(x_k_flat, t, cond_flat)    # (B*N, 12)

        # === 5. 反归一化 ===
        model_out = model_out_flat.reshape(B, N, dec_L)     # (B, N, 12)
        model_out = model_out * std_ + mean_                 # (B, N, 12)

        weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(B * N, 1, 1)

        # trainer 期望 (B, pred_len, N)，与 forward_val_test 一致
        model_out = model_out.permute(0, 2, 1)               # (B, pred_len, N)

        return model_out, weight_tmp

    def forward_val_test(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):
        x_future = x_dec[:,-self.configs.pred_len:,:].permute(0,2,1)
        x_past = x_enc.permute(0,2,1)
        f_dim = -1 if self.configs.features in ['MS'] else 0
        batchs, nF, nL = np.shape(x_past)[0], self.enc_in, self.pred_len
        batchs = batchs*self.enc_in
        if self.configs.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        all_outs = []
        B = np.shape(x_past)[0]
        N = self.configs.enc_in
        if self.configs.new_norm:
            x_past = x_past.permute(0,2,1) #(B,L,N)
            x_past = self.revin_layer(x_past,'norm')
            x_past = x_past.permute(0,2,1) #(B,N,L)
        else:
            mean_ = torch.mean(x_past, dim=-1,keepdims=True)
            std_ = torch.std(x_past, dim=-1,keepdims=True)
            x_past = (x_past-mean_)/(std_+0.00001)
        x_past = torch.reshape(x_past,(B*N,-1))
        x_past = x_past.to(device=self.device)
        import time
        for i in range(sample_times):
            start_code = torch.randn((batchs, nL), device=self.device)
            t0 = time.time()
            diff_samples ,_= self.sampler.sample(S=self.configs.s_steps,
                                             conditioning=x_past,
                                             x_mark_enc=x_mark_enc,
                                             batch_size=batchs,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=start_code)
            # if i == 0:
            #     print(f"  [forward_val_test] sample 0/{sample_times} done: {time.time()-t0:.1f}s, s_steps={self.configs.s_steps}")
            diff_samples=torch.reshape(diff_samples,(B,N,-1))
            if self.configs.new_norm:
                diff_samples = diff_samples.permute(0,2,1).to(self.device) #(B,TARGET L,N)
                diff_samples = self.revin_layer(diff_samples,'denorm')
            else:
                diff_samples = diff_samples.to(self.device)*(std_+0.00001) + mean_
                diff_samples = diff_samples.permute(0,2,1)
            # 只取最后 pred_len 步
            diff_samples = diff_samples[:, -self.configs.pred_len:, :]
            all_outs.append(diff_samples)
        all_outs = torch.stack(all_outs, dim=0)  # (M, B, pred_len, N)
        outs = self._aggregate_samples(all_outs)  # (B, pred_len, N)

        return outs, all_outs.permute(1, 0, 2, 3)  # (B, pred_len, N), (B, M, pred_len, N)

    def set_new_noise_schedule(self, given_betas=None, beta_schedule="linear", diff_steps=1000, beta_start=1e-4, beta_end=2e-2
    ):  

        betas = cosine_beta_schedule(diff_steps,self.configs.coss)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = beta_start
        self.linear_end = beta_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        lvlb_weights = 0.8 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))


        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all() 

    def noise_ts(self, x_start, t, noise=None):

        noise = default(noise, lambda: self.scaling_noise * torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def _emp_mean(self, seq):
        return torch.sum(seq, dim=0) / seq.size(0)

    def _median_of_means(self, tensor):
        if self.n_blocks > tensor.size(0):
            self.n_blocks = int(torch.ceil(tensor.size(0) / 2))

        indic = torch.randperm(tensor.size(0))
        tensor = tensor[indic]  # Shuffle the tensor according to indic
        block_size = tensor.size(0) // self.n_blocks

        means = []
        for i in range(self.n_blocks):
            start_index = i * block_size
            end_index = start_index + block_size if (i+1) < self.n_blocks else tensor.size(0)
            block = tensor[start_index:end_index]
            block_mean = self._emp_mean(block)
            means.append(block_mean)

        means = torch.stack(means)
        return torch.median(means, dim=0)[0]

    def _rob_median_of_means(self, outputs):
        results = []
        for _ in range(self.rmom_n):
            shuffled_outputs = outputs[torch.randperm(outputs.size(0))]
            result = self._median_of_means(shuffled_outputs)
            results.append(result)
        results = torch.stack(results)
        return self._emp_mean(results)

    def _aggregate_samples(self, all_outs):
        # all_outs: (M, B, T, N)
        import time
        t0 = time.time()
        m = all_outs.size(0)
        use_mom = bool(getattr(self.configs, 'use_mom', 1))
        mode = str(getattr(self.configs, 'mom_mode', 'mom')).lower()

        if m <= 1:
            result = all_outs.mean(0)
            print(f"  [_agg] m={m} <= 1, mean. t={time.time()-t0:.2f}s")
            return result

        if mode == 'mean':
            result = all_outs.mean(0)
            print(f"  [_agg] mode=mean, t={time.time()-t0:.2f}s")
            return result

        if use_mom:
            result = self._rob_median_of_means(all_outs)
            print(f"  [_agg] rob_mom m={m}, use_mom={use_mom}, rmom_n={self.rmom_n}, n_blocks={self.n_blocks}, t={time.time()-t0:.2f}s")
            return result
        result = all_outs.mean(0)
        print(f"  [_agg] fallback mean, t={time.time()-t0:.2f}s")
        return result


class PatchUVIT(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = FormerBone_TST(configs)
        self.enc_in = configs.enc_in

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):
        x = rearrange(x, '(b n) h -> b n h', n=self.enc_in)
        cond_ts = rearrange(cond_ts, '(b n) h -> b n h', n=self.enc_in)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in).unsqueeze(-1).unsqueeze(-1)
        x = self.model(x, timesteps, cond_ts)
        return x


class TemporalLayer(nn.Module):
    """T-层: 时序建模层"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (b, n, p, d)
        # print(f"  [T-Layer] Input shape: {x.shape}")
        residual = x
        out = self.norm(x + self.proj(x))
        # print(f"  [T-Layer] Output shape: {out.shape}")
        return out


class SpatialLWRLayer(nn.Module):
    """S-层: 基于LWR的空间传播层（简单版本）"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (b, n, p, d) - b=batch, n=vars/nodes, p=patches, d=dim
        # print(f"  [S-Layer] Input shape: {x.shape}")
        b, n, p, d = x.shape

        # 简单实现: 节点间信息传播（使用可学习的权重）
        pooled = x.mean(dim=2, keepdim=True)  # (b, n, 1, d)
        proj = self.proj(pooled)  # (b, n, 1, d)
        proj = proj.expand(-1, -1, p, -1)  # (b, n, p, d)

        out = self.norm(x + proj)
        # print(f"  [S-Layer] Output shape: {out.shape}")
        return out


class Transpose(nn.Module):
    """用于 BatchNorm1d 的转置"""
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.transpose(*self.dims)


# ============================================================
# TST-LWR 架构（时空联合扩散）
# ============================================================

class TemporalAttention_TST(nn.Module):
    """
    T-层：时序自注意力（RoPE）

    输入: (B, N, T, d)
    在 T 维度做自注意力，N 维度作为变量维度保持不变
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim // 2)

    def forward(self, x):
        """
        Args:
            x: (B, N, T, d)
        Returns:
            out: (B, N, T, d)
        """
        B, N, T, d = x.shape

        # QKV: (B, N, T, 3d) -> (B, N, T, 3, heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # (3, B, N, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE旋转
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # 注意力: (B, N, heads, T, head_dim)
        scale = q.shape[-1] ** 0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn = torch.matmul(attn_weights, v)

        # 合并头: (B, N, T, d)
        out = attn.permute(0, 2, 3, 1, 4).reshape(B, N, T, d)
        out = self.proj(out)
        return out


class SpatialLWROperator(nn.Module):
    """
    S-层：LWR空间传播

    核心思想：在 N 维度做空间传播
    D = I - S^T （后向差分算子）

    输入: (B, N, T, d)
    输出: (B, N, T, d)
    """
    def __init__(self, d_model, num_nodes, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes

        # 解码出 q 来做空间传播
        self.q_proj = nn.Linear(d_model, d_model)
        self.update = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def build_d_matrix(self, device):
        """构建后向差分算子 D = I - S^T"""
        D = torch.zeros(self.num_nodes, self.num_nodes, device=device)
        # D[i,j] = 1 if j == i, -1 if j == i-1 (upstream)
        D[0, 0] = 1.0
        for i in range(1, self.num_nodes):
            D[i, i] = 1.0
            D[i, i-1] = -1.0
        return D

    def forward(self, h):
        """
        Args:
            h: (B, N, T, d) - 隐表示
        Returns:
            h_new: (B, N, T, d) - 更新后的隐表示
        """
        B, N, T, d = h.shape

        # 第一步：从隐表示解码出 q
        q_tilde = self.q_proj(h)  # (B, N, T, d)

        # 第二步：构建 D 矩阵并做空间传播
        D = self.build_d_matrix(h.device)
        # q_spatial[n,t] = D[n,:] @ q_tilde[:,t,:]
        # D: (N,N), q_tilde: (B,N,T,d) → (B,N,T,d)
        q_spatial = torch.einsum('nm,bntd->bmtd', D, q_tilde)  # (B, N, T, d)

        # 第三步：计算 Δh
        delta_h = self.update(q_spatial)  # (B, N, T, d)

        # 第四步：直接递推
        h_new = h + torch.sigmoid(self.gate) * delta_h
        h_new = self.dropout(h_new)

        return h_new


class TSTBlock(nn.Module):
    """
    T-S-T Block（时空联合块）

    数据流：
    h → T-Attention → Cross-Attention → S-LWR → T-Attention → FFN → h
    """
    def __init__(self, d_model, num_heads, num_nodes, dropout=0.1):
        super().__init__()

        # T-层1：时序自注意
        self.norm1 = nn.LayerNorm(d_model)
        self.t_attn1 = TemporalAttention_TST(d_model, num_heads, dropout)

        # Cross-Attention：注入条件
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # S-层：LWR空间传播
        self.norm3 = nn.LayerNorm(d_model)
        self.s_lwr = SpatialLWROperator(d_model, num_nodes, dropout)

        # T-层2：时序融合
        self.norm4 = nn.LayerNorm(d_model)
        self.t_attn2 = TemporalAttention_TST(d_model, num_heads, dropout)

        # FFN
        self.norm5 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, h, cond):
        """
        Args:
            h: (B, N, T, d) - 输入隐表示
            cond: (B, N, L, d) - 条件序列
        Returns:
            out: (B, N, T, d)
        """
        # T-层1：时序自注意
        h = h + self.t_attn1(self.norm1(h))

        # Cross-Attention：条件注入（cond=None 时跳过，条件已在 zcube 里）
        if cond is not None:
            B, N, T, d = h.shape
            h_flat = h.reshape(B * N, T, d)  # (B*N, T, d)
            cond_flat = cond.reshape(B * N, -1, d)  # (B*N, L, d)
            cross_out, _ = self.cross_attn(h_flat, cond_flat, cond_flat)
            h = (h_flat + cross_out).reshape(B, N, T, d)

        # S-层：LWR空间传播
        h = h + self.s_lwr(self.norm3(h))

        # T-层2：时序融合
        h = h + self.t_attn2(self.norm4(h))

        # FFN
        h = h + self.ffn(self.norm5(h))

        return h


class FormerBone_TST(nn.Module):
    """
    TST-LWR 主干网络（替代 FormerBone）

    严格对齐 SimDiff FormerBone 的数据格式：
    - 输入: (B, N, T) 和 (B, N, L)
    - 输出: (B*N, T)

    数据流：
    1. Unfold patch 化：(B, N, T) → (B, N, num_patches, patch_len)
    2. Linear(patch_len, d_model) 投影
    3. 时间步 token 拼接
    4. TST Blocks × e_layers（每个 block 内部有 T-Attn + Cross-Attn + S-LWR）
    5. Output Projection: (B, N, T, d) → (B, N, T)
    6. Reshape: (B, N, T) → (B*N, T)
    """
    def __init__(self, configs):
        super().__init__()

        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.num_nodes = configs.enc_in
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.total_len = configs.label_len + configs.pred_len

        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # Patch 数量（以 pred_len 为基准，与 SimDiff FormerBone 完全一致）
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        patch_num_forecast = int((configs.pred_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.patch_num_forecast = patch_num_forecast
        total_patches = patch_num + 1 + patch_num_forecast  # cond_p + time_token + x_p

        # SimDiff 风格输出投影
        self.W_outs = nn.Linear(total_patches * configs.d_model, configs.pred_len)

        # SimDiff 风格的输入投影
        self.W_input_projection = nn.Linear(self.patch_len, configs.d_model)
        self.input_dropout = nn.Dropout(configs.dropout)

        # 时间步 token
        self.cls = nn.Sequential(nn.Linear(1, configs.d_model))

        # ========== TST Blocks（替代 SimDiff U-Net attention）==========
        # TSTBlock 接收 unfold 后的 patch 序列 (B,N,P,d) 作为 "时间维度" 做时空联合建模
        self.tst_blocks = nn.ModuleList([
            TSTBlock(
                d_model=configs.d_model,
                num_heads=configs.num_heads,
                num_nodes=configs.enc_in,
                dropout=configs.dropout
            )
            for _ in range(configs.e_layers)
        ])

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None):
        """
        Args:
            x: (B, N, T) 或 (B*N, T)
            timesteps: (B, N) 或 (B*N,)
            cond_ts: (B, N, L) 或 (B*N, L)

        Returns:
            out: (B*N, pred_len)
        """
        x_dim = len(x.shape)
        if x_dim == 2:
            B_all, T_total = x.shape
            L = cond_ts.shape[-1]
            N = self.num_nodes
            B = B_all // N
            x = x.view(B, N, T_total)
            cond_ts = cond_ts.view(B, N, L)
        else:
            B, N, T_total = x.shape
            B_all = B * N
            L = cond_ts.shape[-1]

        # SimDiff 风格 unfold
        zcube0 = cond_ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        zcube1 = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        zcube = torch.cat([zcube0, zcube1], dim=-2)  # (B, N, cond_p+x_p, patch_len)
        z_embed = self.input_dropout(self.W_input_projection(zcube))  # (B, N, cond_p+x_p, d)

        # 时间步 token
        time_token = self.cls(timesteps.float())  # (B, N, 1, d)
        z_embed = torch.cat((time_token, z_embed), dim=-2)  # (B, N, 1+P, d)

        # TST Blocks（时空联合建模替代 U-Net attention）
        inputs = z_embed
        for block in self.tst_blocks:
            inputs = block(inputs, cond=None)

        # W_outs 投影（与 SimDiff 完全一致）
        z_out = self.W_outs(inputs[:, :, :, :].reshape(B, N, -1)).reshape(B * N, -1)  # (B*N, pred_len)
        return z_out


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
        self.norm_post1 = nn.Sequential(Transpose((1, 2)), nn.BatchNorm1d(config.d_model), Transpose((1, 2)))
        self.norm_attn = nn.Sequential(Transpose((1, 2)), nn.BatchNorm1d(config.d_model), Transpose((1, 2)))
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


class FormerBone(nn.Module):
    """SimDiff 风格的 FormerBone，使用 unfold + Linear"""
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
            nn.Sequential(Transpose((1, 2)), nn.BatchNorm1d(configs.d_model), Transpose((1, 2)))
            for i in range(configs.e_layers)
        ])

        # ========== T/S 层（可插拔）==========
        if getattr(configs, 'use_tst_layer', False):
            self.temporal_layer = TemporalLayer(self.d_model, configs.num_heads)
            self.spatial_layer = SpatialLWRLayer(self.d_model)
        else:
            self.temporal_layer = None
            self.spatial_layer = None

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None):
        # print(f"\n[FormerBone] Input x shape: {x.shape}")
        # print(f"[FormerBone] cond_ts shape: {cond_ts.shape}")
        # print(f"[FormerBone] timesteps shape: {timesteps.shape}")

        b, c, s = x.shape

        # SimDiff 风格：使用 unfold 进行 patch 化
        zcube0 = cond_ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        zcube1 = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        zcube = torch.cat([zcube0, zcube1], dim=-2)

        # print(f"[FormerBone] After unfold - zcube shape: {zcube.shape}")

        z_embed = self.input_dropout(self.W_input_projection(zcube))
        # print(f"[FormerBone] After projection - z_embed shape: {z_embed.shape}")

        # 时间 token
        time_token = self.cls(timesteps.float())
        z_embed = torch.cat((time_token, z_embed), dim=-2)
        # print(f"[FormerBone] After time token concat - z_embed shape: {z_embed.shape}")

        inputs = z_embed
        b, c, t, h = inputs.shape

        # ========== T/S 层（可插拔）==========
        if self.temporal_layer is not None:
            inputs = self.temporal_layer(inputs)

        # U-Net 风格 Attention（向下）
        skip = []
        for i, (a_2, mlp, drop, norm) in enumerate(zip(
            self.Attentions_over_token, self.Attentions_mlp,
            self.Attentions_dropout, self.Attentions_norm
        )):
            # print(f"[FormerBone] Attention down layer {i}, input shape: {inputs.shape}")
            output = a_2(inputs)
            inputs = drop(output)
            skip.append(inputs)

        # Middle 层
        # print(f"[FormerBone] Attention middle layer, input shape: {inputs.shape}")
        inputs = self.Attentions_over_token_mid(inputs)
        inputs = self.Attentions_dropout_mid(inputs)

        # ========== S-层（可插拔）==========
        if self.spatial_layer is not None:
            inputs = self.spatial_layer(inputs)

        # U-Net 风格 Attention（向上）+ skip connection
        for i, (a_2, mlp, drop, norm) in enumerate(zip(
            self.Attentions_over_token_up, self.Attentions_mlp,
            self.Attentions_dropout_up, self.Attentions_norm
        )):
            # print(f"[FormerBone] Attention up layer {i}, input shape: {inputs.shape}")
            prev = skip.pop()
            outputs = drop(mlp(torch.cat((prev, inputs), dim=-1)))
            outputs = norm(outputs.reshape(b * c, t, -1)).reshape(b, c, t, -1)
            output = a_2(inputs)
            inputs = drop(output)

        # 输出映射
        z_out = self.W_outs(output[:, :, :, :].reshape(b, c, -1)).reshape(b * c, -1)
        # print(f"[FormerBone] Final output shape: {z_out.shape}\n")
        return z_out



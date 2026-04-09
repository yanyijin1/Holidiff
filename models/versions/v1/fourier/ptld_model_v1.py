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

        #print(np.shape(x_enc),np.shape(x_mark_enc)) # (B,L,N)
        x = x_dec[:,-self.configs.pred_len:,:].permute(0,2,1) 
        f_dim = -1 if self.configs.features in ['MS'] else 0
        x=x[:,f_dim:,:]
        cond_ts = x_enc#(B,L,N)
        if self.configs.new_norm:
            cond_ts = self.revin_layer(cond_ts,'norm')
            cond_ts = cond_ts.permute(0,2,1) #(B,N,L)
            lenth = np.shape(x)[1] 
            mean_ = torch.mean(x[:,-lenth:,:], dim=1).unsqueeze(1)
            std_ = torch.ones_like(torch.std(x, dim=1).unsqueeze(1))
            x = (x-mean_.repeat(1,lenth,1))/(std_.repeat(1,lenth,1)+0.00001)
            B = np.shape(x)[0]
            N = self.configs.enc_in
            L1 = np.shape(cond_ts)[2]
            L2 = np.shape(x)[2]
            cond_ts = torch.reshape(cond_ts,(B*N,L1))
            x = torch.reshape(x,(B*N,L2))
            t = torch.randint(0, self.num_timesteps, size=[B*N//2,]).long().to(self.device)
            t = torch.cat([t, self.num_timesteps-1-t], dim=0)
            #print(t,t.shape)
            noise = torch.randn_like(x)
            x_k = self.noise_ts(x_start=x, t=t, noise=noise)
            model_out= self.nn(x_k, t, cond_ts,x_mark_enc)
            model_out=torch.reshape(model_out,(B,N,L2))
            model_out = model_out.permute(0,2,1) #(B,TARGET L,N)
            model_out=self.revin_layer(model_out,'denorm')
            model_out = model_out.permute(0,2,1)  #(B,N,TARGET L)
            weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(model_out.shape[0],model_out.shape[1],1)
        else:
            cond_ts = cond_ts.permute(0,2,1) #(B,N,L) 
            mean_ = torch.mean(cond_ts, dim=-1,keepdims=True)
            std_ = torch.std(cond_ts, dim=-1,keepdims=True)
            cond_ts = (cond_ts-mean_)/(std_+0.00001)
            x = (x-mean_)/(std_+0.00001)
            B = np.shape(x)[0]
            N = self.configs.enc_in
            L1 = np.shape(cond_ts)[2]
            L2 = np.shape(x)[2]
            cond_ts = torch.reshape(cond_ts,(B*N,L1))
            x = torch.reshape(x,(B*N,L2))
            t = torch.randint(0, self.num_timesteps, size=[B*N//2,]).long().to(self.device)
            t = torch.cat([t, self.num_timesteps-1-t], dim=0)
            #print(t,t.shape)
            noise = torch.randn_like(x)
            x_k = self.noise_ts(x_start=x, t=t, noise=noise)
            model_out = self.nn(x_k, t, cond_ts,x_mark_enc)
            model_out = torch.reshape(model_out,(B,N,L2))
            model_out = model_out*(std_+0.00001) + mean_#(B,N,TARGET L)
            weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(model_out.shape[0],model_out.shape[1],1)
        return model_out,weight_tmp

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
        for i in range(sample_times):
            start_code = torch.randn((batchs, nL), device=self.device)
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
            diff_samples=torch.reshape(diff_samples,(B,N,-1))      
            if self.configs.new_norm:                       
                diff_samples = diff_samples.permute(0,2,1).to(self.device) #(B,TARGET L,N)
                diff_samples = self.revin_layer(diff_samples,'denorm')
            else:
                diff_samples = diff_samples.to(self.device)*(std_+0.00001) + mean_
                diff_samples = diff_samples.permute(0,2,1)
            all_outs.append(diff_samples)
        all_outs = torch.stack(all_outs, dim=0)
        outs = self._aggregate_samples(all_outs)
        
        return outs,all_outs.permute(1,0,2,3)

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

    def _cmom(self, outputs, tau=None):
        # outputs: (M, B, T, N)
        M = outputs.size(0)
        if M <= 1:
            return outputs.mean(0)

        k_groups = min(max(2, int(getattr(self.configs, 'n_b', 5))), M)
        indices = torch.randperm(M, device=outputs.device)
        groups = torch.chunk(indices, k_groups)

        means = []
        vars_ = []
        for g in groups:
            if g.numel() == 0:
                continue
            block = outputs[g]
            m = block.mean(0)
            s2 = ((block - m.unsqueeze(0)) ** 2).mean()
            means.append(m)
            vars_.append(s2)

        means = torch.stack(means, dim=0)
        vars_ = torch.stack(vars_)
        if tau is None:
            tau = float(getattr(self.configs, 'mom_tau', 5.0))
        w = torch.softmax(-tau * vars_, dim=0)
        return (w.view(-1, 1, 1, 1) * means).sum(0)

    def _mmmom(self, outputs, clusters=2, tau=None, iters=5):
        # outputs: (M, B, T, N)
        M = outputs.size(0)
        if M <= 1:
            return outputs.mean(0)

        if tau is None:
            tau = float(getattr(self.configs, 'mom_tau', 5.0))
        R = max(1, min(int(clusters), M))

        flat = outputs.reshape(M, -1)
        centers = flat[torch.linspace(0, M - 1, steps=R).long()]

        assign = torch.zeros(M, dtype=torch.long, device=outputs.device)
        for _ in range(iters):
            dist = torch.cdist(flat, centers)
            assign = dist.argmin(dim=1)
            new_centers = []
            for r in range(R):
                mask = assign == r
                if mask.any():
                    new_centers.append(flat[mask].mean(0))
                else:
                    new_centers.append(centers[r])
            centers = torch.stack(new_centers, dim=0)

        robust_centers = []
        dispersions = []
        sizes = []
        for r in range(R):
            mask = assign == r
            if not mask.any():
                continue
            clus = outputs[mask]
            center = self._rob_median_of_means(clus) if clus.size(0) > 1 else clus.mean(0)
            v = ((clus - center.unsqueeze(0)) ** 2).mean()
            robust_centers.append(center)
            dispersions.append(v)
            sizes.append(float(mask.sum().item()))

        if len(robust_centers) == 0:
            return outputs.mean(0)

        centers_t = torch.stack(robust_centers, dim=0)
        disp_t = torch.stack(dispersions)
        size_t = torch.tensor(sizes, device=outputs.device, dtype=disp_t.dtype)
        logits = torch.log(size_t + 1e-8) - tau * disp_t
        pi = torch.softmax(logits, dim=0)
        return (pi.view(-1, 1, 1, 1) * centers_t).sum(0)

    def _aggregate_samples(self, all_outs):
        # all_outs: (M, B, T, N)
        m = all_outs.size(0)
        use_mom = bool(getattr(self.configs, 'use_mom', 1))
        mode = str(getattr(self.configs, 'mom_mode', 'mom')).lower()

        if m <= 1:
            return all_outs.mean(0)

        if mode == 'mean':
            return all_outs.mean(0)

        if mode == 'cmom':
            return self._cmom(all_outs)

        if mode in ('mmmom', 'mm-mom', 'mm_mom'):
            clusters = int(getattr(self.configs, 'mom_clusters', 2))
            return self._mmmom(all_outs, clusters=clusters)

        if use_mom:
            return self._rob_median_of_means(all_outs)
        return all_outs.mean(0)


class PatchUVIT(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = FormerBone(configs)
        self.enc_in = configs.enc_in

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):
        x = rearrange(x, '(b n) h -> b n h', n=self.enc_in)
        cond_ts = rearrange(cond_ts, '(b n) h -> b n h', n=self.enc_in)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in)
        x = self.model(x, timesteps, cond_ts)
        return x


class SinusoidalStepEmbedding(nn.Module):
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


class PatchLWROperatorLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.decode = nn.Linear(d_model, 2)
        self.project = nn.Linear(1, d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(1)

    def forward(self, x, d_op, eps=1e-5):
        _, _, p, _ = x.shape
        dv_dq = self.decode(x)
        dv, dq = dv_dq[..., 0], dv_dq[..., 1]
        drho = dq / (dv.abs() + eps)
        drho_next = torch.zeros_like(drho)
        drho_next[:, :, 0] = drho[:, :, 0]
        for i in range(p - 1):
            drho_next[:, :, i + 1] = drho[:, :, i] + torch.einsum('nm,bm->bn', d_op, dq[:, :, i])
        proj = self.project(self.norm(drho_next.unsqueeze(-1)))
        return x + torch.sigmoid(self.gate) * proj


class PatchDenoiserBlock(nn.Module):
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
        self.lwr_op = PatchLWROperatorLayer(d_model)

    def forward(self, x, h_c, d_op):
        b, n, p, d = x.shape
        xb = x.reshape(b * n, p, d)
        hb = h_c.reshape(b * n, 1, d)
        xb = xb + self.drop(self.self_attn(self.norm1(xb)))
        x2, _ = self.cross_attn(self.norm2(xb), hb, hb)
        xb = xb + self.drop(x2)
        xb = xb + self.drop(self.ffn(self.norm3(xb)))
        return self.lwr_op(xb.reshape(b, n, p, d), d_op)


class FormerBone(nn.Module):
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

        self.patch_embed = PatchEmbed(2, self.d_model, self.patch_len, self.stride)
        self.step_emb = SinusoidalStepEmbedding(self.d_model)
        self.step_proj = nn.Linear(self.d_model, self.d_model)
        self.cond_proj = nn.Linear(self.cond_len, self.d_model)
        self.blocks = nn.ModuleList([
            PatchDenoiserBlock(self.d_model, self.num_heads, self.d_model * 2, configs.dropout)
            for _ in range(self.e_layers)
        ])
        self.patch_unfold = PatchUnfold(self.d_model, self.patch_len, self.stride)

        d = -torch.eye(self.enc_in)
        if self.enc_in > 1:
            d[1:, :-1] += torch.eye(self.enc_in - 1)
        self.register_buffer('d_op', d)

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None):
        b, n, l2 = x.shape
        qt = cond_ts[:, :, -l2:]
        x = torch.stack([x, qt], dim=-1)
        h = self.patch_embed(x)

        t = timesteps.reshape(b * n)
        t_emb = self.step_proj(self.step_emb(t)).reshape(b, n, 1, self.d_model)
        h = h + t_emb

        c = self.cond_proj(cond_ts).unsqueeze(2)
        for block in self.blocks:
            h = block(h, c, self.d_op)

        z_out = self.patch_unfold(h, l2).squeeze(-1)
        return z_out.reshape(b * n, l2)



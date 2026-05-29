import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()


def make_beta_schedule(timesteps, beta_end=0.1, schedule='linear'):
    if schedule == 'cosine':
        return cosine_beta_schedule(timesteps)
    if schedule == 'quad':
        return torch.linspace(1e-4 ** 0.5, beta_end ** 0.5, timesteps).pow(2).float()
    if schedule == 'const':
        return torch.full((timesteps,), float(beta_end))
    return torch.linspace(1e-4, beta_end, timesteps).float()


class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, steps):
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=steps.device) * -scale)
        emb = steps[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class MGResidualBlock(nn.Module):
    def __init__(self, channels, cond_dim, step_dim, dilation):
        super().__init__()
        self.step_proj = nn.Linear(step_dim, channels)
        self.cond_proj = nn.Conv1d(cond_dim, 2 * channels, kernel_size=1)
        self.dilated = nn.Conv1d(channels, 2 * channels, kernel_size=3, padding=dilation, dilation=dilation, padding_mode='circular')
        self.out_proj = nn.Conv1d(channels, 2 * channels, kernel_size=1)

    def forward(self, x, cond, step_emb):
        y = x + self.step_proj(step_emb).unsqueeze(-1)
        y = self.dilated(y) + self.cond_proj(cond)
        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        residual, skip = self.out_proj(y).chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class MGEpsilonTheta(nn.Module):
    def __init__(self, target_dim, cond_dim, residual_layers=8, residual_channels=64, dilation_cycle_length=4, step_dim=64):
        super().__init__()
        self.input_projection = nn.Conv1d(target_dim, residual_channels, kernel_size=1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalStepEmbedding(step_dim),
            nn.Linear(step_dim, step_dim),
            nn.SiLU(),
            nn.Linear(step_dim, step_dim),
            nn.SiLU(),
        )
        self.residual_layers = nn.ModuleList([
            MGResidualBlock(residual_channels, cond_dim, step_dim, dilation=2 ** (i % dilation_cycle_length))
            for i in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.output_projection = nn.Conv1d(residual_channels, target_dim, kernel_size=3, padding=1, padding_mode='circular')
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, time, cond):
        # x: (B,L,N), cond: (B,L,H)
        h = F.leaky_relu(self.input_projection(x.transpose(1, 2)), 0.4)
        cond = cond.transpose(1, 2)
        step_emb = self.diffusion_embedding(time)
        skips = []
        for layer in self.residual_layers:
            h, skip = layer(h, cond, step_emb)
            skips.append(skip)
        h = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(len(skips))
        h = F.leaky_relu(self.skip_projection(h), 0.4)
        return self.output_projection(h).transpose(1, 2)


class MGConditioner(nn.Module):
    def __init__(self, target_dim, hidden_dim=128, layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(target_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, sequence):
        out, _ = self.rnn(sequence)
        return self.proj(out)


class MultiGranGaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, diff_steps=100, share_ratio_list=None, weight_list=None, beta_end=0.1, beta_schedule='linear', loss_type='l2'):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(diff_steps)
        self.share_ratio_list = share_ratio_list or [1.0]
        self.weight_list = weight_list or [1.0]
        self.loss_type = loss_type
        betas = make_beta_schedule(self.num_timesteps, beta_end=beta_end, schedule=beta_schedule)
        self.register_buffer('betas', betas)
        for ratio in self.share_ratio_list:
            betas_sub = betas.clone()
            start_index = int(len(betas_sub) * (1 - ratio))
            betas_sub[:start_index] = 0.0
            alphas = 1.0 - betas_sub
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            suffix = int(ratio * 100)
            self.register_buffer(f'sqrt_alphas_cumprod_{suffix}', torch.sqrt(alphas_cumprod))
            self.register_buffer(f'sqrt_one_minus_alphas_cumprod_{suffix}', torch.sqrt(1.0 - alphas_cumprod))
            self.register_buffer(f'sqrt_recip_alphas_cumprod_{suffix}', torch.sqrt(1.0 / alphas_cumprod.clamp_min(1e-8)))
            self.register_buffer(f'sqrt_recipm1_alphas_cumprod_{suffix}', torch.sqrt(1.0 / alphas_cumprod.clamp_min(1e-8) - 1))
            posterior_variance = betas_sub * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp_min(1e-8)
            self.register_buffer(f'posterior_log_variance_clipped_{suffix}', torch.log(posterior_variance.clamp_min(1e-20)))
            self.register_buffer(f'posterior_mean_coef1_{suffix}', betas_sub * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp_min(1e-8))
            self.register_buffer(f'posterior_mean_coef2_{suffix}', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod).clamp_min(1e-8))

    def q_sample(self, x_start, t, share_ratio, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        suffix = int(share_ratio * 100)
        return extract(getattr(self, f'sqrt_alphas_cumprod_{suffix}'), t, x_start.shape) * x_start + extract(getattr(self, f'sqrt_one_minus_alphas_cumprod_{suffix}'), t, x_start.shape) * noise

    def predict_start_from_noise(self, x_t, t, noise, share_ratio):
        suffix = int(share_ratio * 100)
        return extract(getattr(self, f'sqrt_recip_alphas_cumprod_{suffix}'), t, x_t.shape) * x_t - extract(getattr(self, f'sqrt_recipm1_alphas_cumprod_{suffix}'), t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t, share_ratio):
        suffix = int(share_ratio * 100)
        mean = extract(getattr(self, f'posterior_mean_coef1_{suffix}'), t, x_t.shape) * x_start + extract(getattr(self, f'posterior_mean_coef2_{suffix}'), t, x_t.shape) * x_t
        log_var = extract(getattr(self, f'posterior_log_variance_clipped_{suffix}'), t, x_t.shape)
        return mean, log_var

    def loss_for_target(self, target, cond, share_ratio):
        b = target.shape[0]
        time = torch.randint(0, self.num_timesteps, (b,), device=target.device).long()
        noise = torch.randn_like(target)
        x_noisy = self.q_sample(target, time, share_ratio=share_ratio, noise=noise)
        pred_noise = self.denoise_fn(x_noisy, time, cond)
        if self.loss_type == 'l1':
            return F.l1_loss(pred_noise, noise)
        if self.loss_type == 'huber':
            return F.smooth_l1_loss(pred_noise, noise)
        return F.mse_loss(pred_noise, noise)

    def p_sample(self, x, cond, t, share_ratio):
        noise = self.denoise_fn(x, t, cond)
        x_recon = self.predict_start_from_noise(x, t, noise, share_ratio=share_ratio).clamp(-5.0, 5.0)
        mean, log_var = self.q_posterior(x_recon, x, t, share_ratio=share_ratio)
        nonzero = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * torch.randn_like(x)


class MGTSDAdapter(nn.Module):
    is_mgtsd = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.full_len = int(getattr(configs, 'mgtsd_seq_length', self.seq_len + self.pred_len))
        self.mg_factors = [int(x) for x in str(getattr(configs, 'mgtsd_mg_dict', '1_4_12')).split('_')]
        self.share_ratio_list = [float(x) for x in str(getattr(configs, 'mgtsd_share_ratio_list', '1_0.8_0.6')).split('_')]
        self.weight_list = [float(x) for x in str(getattr(configs, 'mgtsd_weight_list', '0.8_0.1_0.1')).split('_')]
        n = min(len(self.mg_factors), len(self.share_ratio_list), len(self.weight_list))
        self.mg_factors = self.mg_factors[:n]
        self.share_ratio_list = self.share_ratio_list[:n]
        weight_sum = sum(self.weight_list[:n]) or 1.0
        self.weight_list = [w / weight_sum for w in self.weight_list[:n]]
        hidden_dim = int(getattr(configs, 'mgtsd_hidden_dim', getattr(configs, 'd_model', 128)))
        self.conditioner = MGConditioner(
            target_dim=self.enc_in,
            hidden_dim=hidden_dim,
            layers=int(getattr(configs, 'mgtsd_rnn_layers', 1)),
            dropout=float(getattr(configs, 'mgtsd_dropout', 0.0)),
        )
        self.denoise_fn = MGEpsilonTheta(
            target_dim=self.enc_in,
            cond_dim=hidden_dim,
            residual_layers=int(getattr(configs, 'mgtsd_residual_layers', 8)),
            residual_channels=int(getattr(configs, 'mgtsd_residual_channels', 64)),
            dilation_cycle_length=int(getattr(configs, 'mgtsd_dilation_cycle_length', 4)),
            step_dim=int(getattr(configs, 'mgtsd_step_dim', 64)),
        )
        self.diffusion = MultiGranGaussianDiffusion(
            denoise_fn=self.denoise_fn,
            diff_steps=int(getattr(configs, 'mgtsd_diff_steps', getattr(configs, 'diff_steps', 100))),
            share_ratio_list=self.share_ratio_list,
            weight_list=self.weight_list,
            beta_end=float(getattr(configs, 'mgtsd_beta_end', 0.1)),
            beta_schedule=str(getattr(configs, 'mgtsd_beta_schedule', 'linear')),
            loss_type=str(getattr(configs, 'mgtsd_loss_type', 'l2')),
        )
        self.use_ema = bool(getattr(configs, 'mgtsd_use_ema', True))
        self.ema_decay = float(getattr(configs, 'mgtsd_ema_decay', 0.995))
        self.ema_update_every = int(getattr(configs, 'mgtsd_ema_update_every', 10))
        self.ema_conditioner = copy.deepcopy(self.conditioner)
        self.ema_denoise_fn = copy.deepcopy(self.denoise_fn)
        for module in [self.ema_conditioner, self.ema_denoise_fn]:
            for param in module.parameters():
                param.requires_grad_(False)
        self.register_buffer('ema_steps', torch.tensor(0, dtype=torch.long))

    def _build_full_series(self, x_enc, batch_y=None):
        future = torch.zeros(x_enc.size(0), self.pred_len, x_enc.size(-1), device=x_enc.device, dtype=x_enc.dtype) if batch_y is None else batch_y[:, -self.pred_len:, :]
        full = torch.cat([x_enc, future], dim=1)
        if full.size(1) > self.full_len:
            return full[:, :self.full_len, :]
        if full.size(1) < self.full_len:
            return F.pad(full, (0, 0, 0, self.full_len - full.size(1)))
        return full

    def _coarse_target(self, full, factor):
        if factor <= 1:
            return full
        x = full.transpose(1, 2)
        pad = (factor - x.size(-1) % factor) % factor
        if pad > 0:
            x = F.pad(x, (0, pad), mode='replicate')
        coarse = F.avg_pool1d(x, kernel_size=factor, stride=factor)
        coarse = coarse.repeat_interleave(factor, dim=-1)[..., :full.size(1)]
        return coarse.transpose(1, 2)

    def training_loss(self, x_enc, batch_y, batch_y_mask=None):
        full = self._build_full_series(x_enc, batch_y)
        cond = self.conditioner(full)
        losses = []
        for factor, ratio, weight in zip(self.mg_factors, self.share_ratio_list, self.weight_list):
            target = self._coarse_target(full, factor)
            losses.append(weight * self.diffusion.loss_for_target(target, cond, share_ratio=ratio))
        return sum(losses)

    @torch.no_grad()
    def after_optimizer_step(self):
        if not self.use_ema:
            return
        self.ema_steps += 1
        if int(self.ema_steps.item()) % max(self.ema_update_every, 1) != 0:
            return
        for ema_module, module in [(self.ema_conditioner, self.conditioner), (self.ema_denoise_fn, self.denoise_fn)]:
            for ema_param, param in zip(ema_module.parameters(), module.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)
            for ema_buffer, buffer in zip(ema_module.buffers(), module.buffers()):
                ema_buffer.copy_(buffer)

    @torch.no_grad()
    def _sample_once(self, target, mask, sample_conditioner, sample_denoiser):
        old_denoiser = self.diffusion.denoise_fn
        self.diffusion.denoise_fn = sample_denoiser
        try:
            cond = sample_conditioner(target)
            x = torch.randn_like(target)
            ratio = self.share_ratio_list[0]
            for i in reversed(range(0, self.diffusion.num_timesteps)):
                t = torch.full((target.shape[0],), i, device=target.device, dtype=torch.long)
                x = self.diffusion.p_sample(x, cond, t, share_ratio=ratio)
                x_known = self.diffusion.q_sample(target, t, share_ratio=ratio)
                x = torch.where(mask, x_known, x)
            x = torch.where(mask, target, x)
            return x
        finally:
            self.diffusion.denoise_fn = old_denoiser

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=1, holiday_flag=None):
        target = self._build_full_series(x_enc, None)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, :self.seq_len, :] = True
        sample_conditioner = self.ema_conditioner if self.use_ema and not self.training else self.conditioner
        sample_denoiser = self.ema_denoise_fn if self.use_ema and not self.training else self.denoise_fn
        samples = []
        for _ in range(max(1, int(sample_times))):
            generated = self._sample_once(target, mask, sample_conditioner, sample_denoiser)
            samples.append(generated[:, -self.pred_len:, :])
        all_samples = torch.stack(samples, dim=1)
        return all_samples.mean(dim=1), all_samples

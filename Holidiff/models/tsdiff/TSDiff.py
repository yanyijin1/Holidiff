import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BackboneModel


def extract(a, t, shape):
    return a.to(t.device).gather(-1, t).reshape(t.shape[0], *((1,) * (len(shape) - 1)))


def linear_beta(n):
    return torch.linspace(1e-4, 0.1, n).float()


def cosine_beta(n, s=0.008):
    x = torch.linspace(0, n, n + 1, dtype=torch.float64)
    ac = torch.cos(((x / n) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - ac[1:] / ac[:-1], 1e-4, 0.9999).float()


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.ctx = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.timesteps = int(getattr(configs, 'tsdiff_timesteps', 100))
        self.local_scaling = bool(getattr(configs, 'local_scaling_enable', getattr(configs, 'tsdiff_local_scaling', True)))

        self.backbone = BackboneModel(
            input_dim=self.enc_in,
            hidden_dim=int(getattr(configs, 'tsdiff_hidden_dim', 64)),
            output_dim=self.enc_in,
            step_emb=int(getattr(configs, 'tsdiff_step_emb', 128)),
            num_residual_blocks=int(getattr(configs, 'tsdiff_num_residual_blocks', 3)),
            num_features=0,
            residual_block=str(getattr(configs, 'tsdiff_residual_block', 's4')),
            dropout=float(getattr(configs, 'tsdiff_dropout', 0.0)),
            init_skip=bool(getattr(configs, 'tsdiff_init_skip', True)),
        )

        schedule = str(getattr(configs, 'tsdiff_beta_schedule', 'linear')).lower()
        betas = linear_beta(self.timesteps) if schedule == 'linear' else cosine_beta(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recip_alphas_cumprod', (1.0 / alphas_cumprod).sqrt())
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def _scale(self, x):
        if self.local_scaling:
            return x.abs().mean(1, keepdim=True).clamp_min(1e-5)
        return torch.ones_like(x[:, :1])

    def _concat_context_target(self, batch_x, batch_y=None):
        scale = self._scale(batch_x)
        if batch_y is None:
            future = torch.zeros(batch_x.size(0), self.pred_len, batch_x.size(-1), device=batch_x.device, dtype=batch_x.dtype)
        else:
            future = batch_y[:, -self.pred_len:, :].to(batch_x.device)
        return torch.cat([batch_x, future], dim=1) / scale, scale

    def training_loss(self, batch_x, batch_y, batch_y_mask=None, **kwargs):
        target = batch_y[:, -self.pred_len:, :].to(batch_x.device)
        series, _ = self._concat_context_target(batch_x, target)
        t = torch.randint(0, self.timesteps, (series.size(0),), device=series.device).long()
        noise = torch.randn_like(series)
        noisy = extract(self.sqrt_alphas_cumprod, t, series.shape) * series + extract(self.sqrt_one_minus_alphas_cumprod, t, series.shape) * noise
        pred_noise = self.backbone(noisy, t, None)
        return F.mse_loss(pred_noise, noise)

    def _denoise(self, noisy, t, noise):
        return (noisy - extract(self.sqrt_one_minus_alphas_cumprod, t, noisy.shape) * noise) / extract(self.sqrt_alphas_cumprod, t, noisy.shape)

    @torch.no_grad()
    def _sample_once(self, batch_x, sampling_steps=0, guidance_scale=1.0, guidance_clip=10.0):
        target, scale = self._concat_context_target(batch_x, None)
        known_mask = torch.zeros_like(target, dtype=torch.bool)
        known_mask[:, :self.ctx, :] = True
        seq = torch.randn_like(target)

        num_steps = max(1, min(int(sampling_steps) if sampling_steps else self.timesteps, self.timesteps))
        step_ids = torch.linspace(0, self.timesteps - 1, num_steps, device=batch_x.device).long().unique(sorted=True)
        prev_step_ids = torch.cat([torch.tensor([-1], device=batch_x.device), step_ids[:-1]])

        for step_id, prev_step_id in zip(reversed(step_ids.tolist()), reversed(prev_step_ids.tolist())):
            t = torch.full((batch_x.size(0),), step_id, device=batch_x.device, dtype=torch.long)
            prev_t = torch.full_like(t, max(prev_step_id, 0))
            with torch.enable_grad():
                seq_grad = seq.detach().requires_grad_(True)
                pred_noise = self.backbone(seq_grad, t, None)
                x0 = self._denoise(seq_grad, t, pred_noise)
                energy = F.mse_loss(x0[known_mask], target[known_mask], reduction='sum') / known_mask.sum().clamp_min(1)
                score = -torch.autograd.grad(energy, seq_grad)[0].clamp(-guidance_clip, guidance_clip)
            pred_noise = pred_noise.detach() - extract(self.sqrt_one_minus_alphas_cumprod, t, seq.shape) * guidance_scale * score.detach()
            alpha_t = extract(self.alphas_cumprod, t, seq.shape)
            if step_id > 0:
                alpha_prev = extract(self.alphas_cumprod, prev_t, seq.shape)
            else:
                alpha_prev = torch.ones_like(seq[:, :1])
            seq = alpha_prev.sqrt() * (seq - extract(self.sqrt_one_minus_alphas_cumprod, t, seq.shape) * pred_noise) / extract(self.sqrt_alphas_cumprod, t, seq.shape) + (1.0 - alpha_prev).clamp_min(0).sqrt() * pred_noise

        return seq[:, -self.pred_len:, :] * scale

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, sample_times=1, **kwargs):
        num_samples = max(1, int(sample_times))
        sample_list = [
            self._sample_once(
                x_enc,
                int(getattr(self.configs, 'tsdiff_sampling_steps', self.timesteps)),
                float(getattr(self.configs, 'tsdiff_guidance_scale', 1.0)),
                float(getattr(self.configs, 'tsdiff_guidance_clip', 10.0)),
            )
            for _ in range(num_samples)
        ]
        all_samples = torch.stack(sample_list, dim=1)
        outputs = all_samples.mean(dim=1)
        return outputs, all_samples

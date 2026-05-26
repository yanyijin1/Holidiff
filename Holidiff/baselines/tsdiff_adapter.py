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


def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.1, timesteps).float()


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if embeddings.size(-1) < self.dim:
            embeddings = F.pad(embeddings, (0, self.dim - embeddings.size(-1)))
        return embeddings


class TSDiffResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.time_linear = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, padding_mode='circular'),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.gate = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)

    def forward(self, x, t_emb):
        # x: (B, L, H). The original TSDiff uses S4 residual blocks;
        # this keeps the same residual/skip denoising interface without S4 deps.
        z = self.norm(x)
        z = z + self.time_linear(t_emb).unsqueeze(1)
        z = self.conv(z.transpose(1, 2))
        a, b = self.gate(z).chunk(2, dim=1)
        skip = torch.tanh(a) * torch.sigmoid(b)
        out = x + skip.transpose(1, 2)
        return out, skip.transpose(1, 2)


class TSDiffBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=None, step_emb=128, num_residual_blocks=4, dropout=0.0, init_skip=True):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        self.input_init = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.step_embedding = SinusoidalPositionEmbeddings(step_emb)
        self.time_init = nn.Sequential(
            nn.Linear(step_emb, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.residual_blocks = nn.ModuleList([
            TSDiffResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_residual_blocks)
        ])
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.init_skip = init_skip

    def forward(self, x, t, features=None):
        h = self.input_init(x)
        t_emb = self.time_init(self.step_embedding(t))
        skips = []
        for block in self.residual_blocks:
            h, skip = block(h, t_emb)
            skips.append(skip)
        h = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(len(skips))
        out = self.out_linear(h)
        if self.init_skip:
            out = out + x
        return out


class TSDiffCore(nn.Module):
    def __init__(self, seq_length, feature_size, timesteps=1000, beta_schedule='cosine', hidden_dim=128, step_emb=128, num_residual_blocks=4, dropout=0.0, init_skip=True):
        super().__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.timesteps = int(timesteps)
        self.backbone = TSDiffBackbone(
            input_dim=feature_size,
            hidden_dim=hidden_dim,
            output_dim=feature_size,
            step_emb=step_emb,
            num_residual_blocks=num_residual_blocks,
            dropout=dropout,
            init_skip=init_skip,
        )
        betas = linear_beta_schedule(self.timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register = lambda name, val: self.register_buffer(name, val.float())
        register('betas', betas)
        register('alphas', alphas)
        register('alphas_cumprod', alphas_cumprod)
        register('alphas_cumprod_prev', alphas_cumprod_prev)
        register('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.backbone(x_noisy, t, features=None)
        return F.mse_loss(predicted_noise, noise)

    def training_loss(self, x_start):
        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device).long()
        return self.p_losses(x_start, t)

    def fast_denoise(self, xt, t, noise=None):
        noise = self.backbone(xt, t, features=None) if noise is None else noise
        return (xt - extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape) * noise) / extract(self.sqrt_alphas_cumprod, t, xt.shape)

    @torch.no_grad()
    def p_sample(self, x, t, t_index, noise=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        predicted_noise = self.backbone(x, t, features=None) if noise is None else noise
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)

    def p_sample_genddim(self, x, t, t_index, t_prev, eta=0.0, noise=None):
        noise = self.backbone(x, t, features=None) if noise is None else noise
        alphas_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_prev_t = extract(self.alphas_cumprod, t_prev.clamp_min(0), x.shape) if t_index > 0 else torch.ones_like(alphas_cumprod_t)
        sqrt_alphas_cumprod_prev_t = alphas_cumprod_prev_t.sqrt()
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        x0pointer = sqrt_alphas_cumprod_prev_t * (x - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        c1 = eta * ((1 - alphas_cumprod_t / alphas_cumprod_prev_t) * (1 - alphas_cumprod_prev_t) / (1 - alphas_cumprod_t)).clamp_min(0).sqrt()
        c2 = ((1 - alphas_cumprod_prev_t) - c1 ** 2).clamp_min(0).sqrt()
        return x0pointer + c1 * torch.randn_like(x) + c2 * noise


class TSDiffAdapter(nn.Module):
    is_tsdiff = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.full_len = int(getattr(configs, 'tsdiff_seq_length', self.seq_len + self.pred_len))
        self.sampling_steps = int(getattr(configs, 'tsdiff_sampling_steps', 50))
        self.skip_type = str(getattr(configs, 'tsdiff_skip_type', 'uniform'))
        self.eta = float(getattr(configs, 'tsdiff_eta', 0.0))
        self.guidance_scale = float(getattr(configs, 'tsdiff_guidance_scale', 1.0))
        self.guidance_clip = float(getattr(configs, 'tsdiff_guidance_clip', 10.0))
        self.use_ema = bool(getattr(configs, 'tsdiff_use_ema', True))
        self.ema_decay = float(getattr(configs, 'tsdiff_ema_decay', 0.995))
        self.ema_update_every = int(getattr(configs, 'tsdiff_ema_update_every', 10))
        self.core = TSDiffCore(
            seq_length=self.full_len,
            feature_size=self.enc_in,
            timesteps=int(getattr(configs, 'tsdiff_timesteps', 1000)),
            beta_schedule=str(getattr(configs, 'tsdiff_beta_schedule', 'cosine')),
            hidden_dim=int(getattr(configs, 'tsdiff_hidden_dim', getattr(configs, 'd_model', 128))),
            step_emb=int(getattr(configs, 'tsdiff_step_emb', getattr(configs, 'd_model', 128))),
            num_residual_blocks=int(getattr(configs, 'tsdiff_num_residual_blocks', 4)),
            dropout=float(getattr(configs, 'tsdiff_dropout', 0.0)),
            init_skip=bool(getattr(configs, 'tsdiff_init_skip', True)),
        )
        self.ema_core = copy.deepcopy(self.core)
        for param in self.ema_core.parameters():
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

    def training_loss(self, x_enc, batch_y, batch_y_mask=None):
        return self.core.training_loss(self._build_full_series(x_enc, batch_y))

    @torch.no_grad()
    def after_optimizer_step(self):
        if not self.use_ema:
            return
        self.ema_steps += 1
        if int(self.ema_steps.item()) % max(self.ema_update_every, 1) != 0:
            return
        for ema_param, param in zip(self.ema_core.parameters(), self.core.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)
        for ema_buffer, buffer in zip(self.ema_core.buffers(), self.core.buffers()):
            ema_buffer.copy_(buffer)

    def _timesteps(self, core):
        steps = max(1, min(self.sampling_steps, core.timesteps))
        if self.skip_type == 'quadratic':
            vals = torch.linspace(0, math.sqrt(core.timesteps - 1), steps, device=core.betas.device).pow(2).long()
        else:
            vals = torch.linspace(0, core.timesteps - 1, steps, device=core.betas.device).long()
        vals = torch.unique(vals, sorted=True)
        prev = torch.cat([torch.tensor([-1], device=vals.device, dtype=torch.long), vals[:-1]])
        return vals, prev

    def _guided_sample(self, core, target, mask):
        batch_size = target.shape[0]
        seq = torch.randn_like(target)
        timesteps, prev_timesteps = self._timesteps(core)
        for i, j in zip(reversed(timesteps.tolist()), reversed(prev_timesteps.tolist())):
            t = torch.full((batch_size,), i, device=target.device, dtype=torch.long)
            t_prev = torch.full((batch_size,), j, device=target.device, dtype=torch.long)
            with torch.enable_grad():
                seq_for_grad = seq.detach().requires_grad_(True)
                noise = core.backbone(seq_for_grad, t, features=None)
                x0 = core.fast_denoise(seq_for_grad, t, noise=noise)
                energy = F.mse_loss(x0[mask], target[mask], reduction='sum') / mask.sum().clamp_min(1)
                score = -torch.autograd.grad(energy, seq_for_grad)[0]
                if self.guidance_clip > 0:
                    score = score.clamp(-self.guidance_clip, self.guidance_clip)
            scale = extract(core.sqrt_one_minus_alphas_cumprod, t, seq.shape) * self.guidance_scale
            guided_noise = noise.detach() - scale * score.detach()
            with torch.no_grad():
                seq = core.p_sample_genddim(seq, t, t_index=i, t_prev=t_prev, eta=self.eta, noise=guided_noise)
        return seq

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=1, holiday_flag=None):
        target = self._build_full_series(x_enc, None)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, :self.seq_len, :] = True
        core = self.ema_core if self.use_ema and not self.training else self.core
        samples = []
        for _ in range(max(1, int(sample_times))):
            generated = self._guided_sample(core, target, mask)
            samples.append(generated[:, -self.pred_len:, :])
        all_samples = torch.stack(samples, dim=1)
        return all_samples.mean(dim=1), all_samples

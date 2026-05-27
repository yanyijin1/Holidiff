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


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=1000):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(dim, max_steps), persistent=False)
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))

    @staticmethod
    def _build_embedding(dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(dim).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / dim)
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class TimeGradResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            padding_mode='circular',
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(1, 2 * residual_channels, kernel_size=1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size=1)

    def forward(self, x, conditioner, diffusion_step):
        y = x + self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = self.dilated_conv(y) + self.conditioner_projection(conditioner)
        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = F.leaky_relu(self.output_projection(y), 0.4)
        residual, skip = y.chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class TimeGradEpsilonTheta(nn.Module):
    def __init__(self, target_dim, cond_length, time_emb_dim=16, residual_layers=8, residual_channels=8, dilation_cycle_length=2, residual_hidden=64):
        super().__init__()
        self.target_dim = target_dim
        self.input_projection = nn.Conv1d(1, residual_channels, kernel_size=1)
        self.diffusion_embedding = DiffusionEmbedding(time_emb_dim, proj_dim=residual_hidden)
        self.cond_upsampler = nn.Sequential(
            nn.Linear(cond_length, max(target_dim // 2, 1)),
            nn.LeakyReLU(0.4),
            nn.Linear(max(target_dim // 2, 1), target_dim),
            nn.LeakyReLU(0.4),
        )
        self.residual_layers = nn.ModuleList([
            TimeGradResidualBlock(
                hidden_size=residual_hidden,
                residual_channels=residual_channels,
                dilation=2 ** (i % dilation_cycle_length),
            )
            for i in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.output_projection = nn.Conv1d(residual_channels, 1, kernel_size=3, padding=1, padding_mode='circular')
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        # inputs: (B, 1, target_dim), cond: (B, 1, cond_length)
        x = F.leaky_relu(self.input_projection(inputs), 0.4)
        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skips = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_up, diffusion_step)
            skips.append(skip)
        x = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(len(skips))
        x = F.leaky_relu(self.skip_projection(x), 0.4)
        return self.output_projection(x)


class TimeGradGaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, input_size, diff_steps=100, loss_type='l2', beta_end=0.1, beta_schedule='linear'):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.num_timesteps = int(diff_steps)
        self.loss_type = loss_type
        betas = make_beta_schedule(self.num_timesteps, beta_end=beta_end, schedule=beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register = lambda name, value: self.register_buffer(name, value.float())
        register('betas', betas)
        register('alphas_cumprod', alphas_cumprod)
        register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register('posterior_log_variance_clipped', torch.log(posterior_variance.clamp_min(1e-20)))
        register('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def p_losses(self, x_start, cond):
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = self.denoise_fn(x_noisy, t, cond=cond)
        if self.loss_type == 'l1':
            return F.l1_loss(pred_noise, noise)
        if self.loss_type == 'huber':
            return F.smooth_l1_loss(pred_noise, noise)
        return F.mse_loss(pred_noise, noise)

    def p_mean_variance(self, x, cond, t):
        noise = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t, noise).clamp(-5.0, 5.0)
        mean = extract(self.posterior_mean_coef1, t, x.shape) * x_recon + extract(self.posterior_mean_coef2, t, x.shape) * x
        log_var = extract(self.posterior_log_variance_clipped, t, x.shape)
        return mean, log_var

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        mean, log_var = self.p_mean_variance(x, cond, t)
        nonzero = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, cond, sampling_steps=None):
        batch_size = cond.shape[0]
        x = torch.randn(batch_size, 1, self.input_size, device=cond.device, dtype=cond.dtype)
        steps = self.num_timesteps if sampling_steps is None else min(int(sampling_steps), self.num_timesteps)
        timesteps = torch.linspace(0, self.num_timesteps - 1, steps, device=cond.device).long().unique(sorted=True)
        for i in reversed(timesteps.tolist()):
            t = torch.full((batch_size,), i, device=cond.device, dtype=torch.long)
            x = self.p_sample(x, cond, t)
        return x


class TimeGradCore(nn.Module):
    def __init__(self, target_dim, lags_seq, rnn_hidden=128, rnn_layers=1, dropout=0.0, cond_length=128, cell_type='GRU', diff_steps=50, loss_type='l2', beta_end=0.1, beta_schedule='linear', residual_layers=8, residual_channels=8, dilation_cycle_length=2, scaling=True, train_steps=None):
        super().__init__()
        self.target_dim = target_dim
        self.lags_seq = sorted(set(int(x) for x in lags_seq))
        self.max_lag = max(self.lags_seq)
        self.scaling = scaling
        self.train_steps = train_steps
        input_size = target_dim * len(self.lags_seq)
        rnn_cls = nn.LSTM if cell_type.upper() == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(input_size=input_size, hidden_size=rnn_hidden, num_layers=rnn_layers, dropout=dropout if rnn_layers > 1 else 0.0, batch_first=True)
        self.proj_cond = nn.Linear(rnn_hidden, cond_length)
        self.denoise_fn = TimeGradEpsilonTheta(
            target_dim=target_dim,
            cond_length=cond_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.diffusion = TimeGradGaussianDiffusion(
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

    def _scale(self, history):
        if not self.scaling:
            return torch.ones(history.size(0), 1, history.size(-1), device=history.device, dtype=history.dtype)
        return history.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)

    def _lag_features(self, sequence):
        features = []
        for lag in self.lags_seq:
            features.append(sequence[:, self.max_lag - lag:sequence.size(1) - lag, :])
        return torch.cat(features, dim=-1)

    def conditions(self, sequence):
        if sequence.size(1) <= self.max_lag:
            raise ValueError(f'TimeGrad requires sequence length > max lag, got length={sequence.size(1)} max_lag={self.max_lag}')
        lagged = self._lag_features(sequence)
        rnn_out, _ = self.rnn(lagged)
        return self.proj_cond(rnn_out)

    def training_loss(self, history, future):
        full = torch.cat([history, future], dim=1)
        scale = self._scale(history)
        full_scaled = full / scale
        cond = self.conditions(full_scaled)
        target = full_scaled[:, self.max_lag:, :]
        b, s, n = target.shape
        if self.train_steps and s > self.train_steps:
            idx = torch.randperm(s, device=target.device)[:self.train_steps]
            target = target[:, idx, :]
            cond = cond[:, idx, :]
            s = self.train_steps
        return self.diffusion.p_losses(target.reshape(b * s, 1, n), cond.reshape(b * s, 1, -1))

    @torch.no_grad()
    def sample(self, history, pred_len, sampling_steps=50):
        scale = self._scale(history)
        sequence = history / scale
        for _ in range(pred_len):
            cond = self.conditions(sequence)[:, -1:, :]
            new_value = self.diffusion.sample(cond, sampling_steps=sampling_steps).squeeze(1)
            sequence = torch.cat([sequence, new_value.unsqueeze(1)], dim=1)
        return sequence[:, -pred_len:, :] * scale


class TimeGradAdapter(nn.Module):
    is_timegrad = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        lags_seq = [int(x) for x in str(getattr(configs, 'timegrad_lags_seq', '1_2_4_8_16_24')).split('_')]
        self.sampling_steps = int(getattr(configs, 'timegrad_sampling_steps', 50))
        self.use_ema = bool(getattr(configs, 'timegrad_use_ema', True))
        self.ema_decay = float(getattr(configs, 'timegrad_ema_decay', 0.995))
        self.ema_update_every = int(getattr(configs, 'timegrad_ema_update_every', 10))
        self.core = TimeGradCore(
            target_dim=self.enc_in,
            lags_seq=lags_seq,
            rnn_hidden=int(getattr(configs, 'timegrad_rnn_hidden', getattr(configs, 'd_model', 128))),
            rnn_layers=int(getattr(configs, 'timegrad_rnn_layers', 1)),
            dropout=float(getattr(configs, 'timegrad_dropout', 0.0)),
            cond_length=int(getattr(configs, 'timegrad_cond_length', getattr(configs, 'd_model', 128))),
            cell_type=str(getattr(configs, 'timegrad_cell_type', 'GRU')),
            diff_steps=int(getattr(configs, 'timegrad_diff_steps', 50)),
            loss_type=str(getattr(configs, 'timegrad_loss_type', 'l2')),
            beta_end=float(getattr(configs, 'timegrad_beta_end', 0.1)),
            beta_schedule=str(getattr(configs, 'timegrad_beta_schedule', 'linear')),
            residual_layers=int(getattr(configs, 'timegrad_residual_layers', 8)),
            residual_channels=int(getattr(configs, 'timegrad_residual_channels', 8)),
            dilation_cycle_length=int(getattr(configs, 'timegrad_dilation_cycle_length', 2)),
            scaling=bool(getattr(configs, 'timegrad_scaling', True)),
            train_steps=int(getattr(configs, 'timegrad_train_steps', 0)) or None,
        )
        self.ema_core = copy.deepcopy(self.core)
        for param in self.ema_core.parameters():
            param.requires_grad_(False)
        self.register_buffer('ema_steps', torch.tensor(0, dtype=torch.long))

    def training_loss(self, x_enc, batch_y, batch_y_mask=None):
        return self.core.training_loss(x_enc, batch_y[:, -self.pred_len:, :])

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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=1, holiday_flag=None):
        sample_core = self.ema_core if self.use_ema and not self.training else self.core
        samples = []
        for _ in range(max(1, int(sample_times))):
            samples.append(sample_core.sample(x_enc, self.pred_len, sampling_steps=self.sampling_steps))
        all_samples = torch.stack(samples, dim=1)
        return all_samples.mean(dim=1), all_samples

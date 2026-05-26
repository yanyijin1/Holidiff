import math
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(x):
    return x


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    return torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float64)


class Transpose(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ConvMLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.net(x).transpose(1, 2)


class TrendBlock(nn.Module):
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super().__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose((1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1),
        )
        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.register_buffer('poly_space', torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0))

    def forward(self, x):
        x = self.trend(x).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space)
        return trend_vals.transpose(1, 2)


class FourierLayer(nn.Module):
    def __init__(self, low_freq=1, factor=1):
        super().__init__()
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t, device=x.device)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t, device=x.device)[self.low_freq:]
        if x_freq.size(1) == 0:
            return torch.zeros_like(x)
        top_k = max(1, int(self.factor * math.log(max(x_freq.size(1), 2))))
        top_k = min(top_k, x_freq.size(1))
        _, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_b = torch.arange(b, device=x.device).view(b, 1, 1).expand(-1, top_k, d)
        mesh_d = torch.arange(d, device=x.device).view(1, 1, d).expand(b, top_k, -1)
        x_freq = x_freq[mesh_b, indices, mesh_d]
        f = f[indices]
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        ts = torch.arange(t, dtype=torch.float, device=x.device).view(1, 1, t, 1)
        amp = x_freq.abs().unsqueeze(2)
        phase = x_freq.angle().unsqueeze(2)
        return (amp * torch.cos(2 * math.pi * f.unsqueeze(2) * ts + phase)).sum(dim=1)


class FullAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class CrossAttention(nn.Module):
    def __init__(self, n_embd, condition_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, enc):
        b, t, c = x.size()
        t_enc = enc.size(1)
        k = self.key(enc).view(b, t_enc, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(enc).view(b, t_enc, self.n_head, c // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(nn.Linear(n_embd, mlp_hidden_times * n_embd), nn.GELU(), nn.Linear(mlp_hidden_times * n_embd, n_embd), nn.Dropout(resid_pdrop))

    def forward(self, x, t):
        x = x + self.attn(self.ln1(x, t))
        return x + self.mlp(self.ln2(x))


class DecoderBlock(nn.Module):
    def __init__(self, n_channel, n_feat, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln1_1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn1 = FullAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn2 = CrossAttention(n_embd, n_embd, n_head, attn_pdrop, resid_pdrop)
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, nn.GELU())
        self.seasonal = FourierLayer()
        self.mlp = nn.Sequential(nn.Linear(n_embd, mlp_hidden_times * n_embd), nn.GELU(), nn.Linear(mlp_hidden_times * n_embd, n_embd), nn.Dropout(resid_pdrop))
        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, enc, t):
        x = x + self.attn1(self.ln1(x, t))
        x = x + self.attn2(self.ln1_1(x, t), enc)
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season


class DiffusionTSTransformer(nn.Module):
    def __init__(self, n_feat, n_channel, n_layer_enc=3, n_layer_dec=6, n_embd=128, n_heads=8, attn_pdrop=0.0, resid_pdrop=0.0, mlp_hidden_times=4, max_len=2048):
        super().__init__()
        self.emb = ConvMLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = ConvMLP(n_embd, n_feat, resid_pdrop=resid_pdrop)
        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=5, stride=1, padding=2, padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.encoder = nn.ModuleList([EncoderBlock(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times) for _ in range(n_layer_enc)])
        self.decoder = nn.ModuleList([DecoderBlock(n_channel, n_feat, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times) for _ in range(n_layer_dec)])
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

    def forward(self, x, t):
        emb = self.emb(x)
        enc = self.pos_enc(emb)
        for block in self.encoder:
            enc = block(enc, t)
        dec = self.pos_dec(emb)
        b, c, _ = dec.shape
        means = []
        season = torch.zeros_like(dec)
        trend = torch.zeros((b, c, x.size(-1)), device=x.device, dtype=x.dtype)
        for block in self.decoder:
            dec, residual_mean, residual_trend, residual_season = block(dec, enc, t)
            means.append(residual_mean)
            trend = trend + residual_trend
            season = season + residual_season
        res = self.inverse(dec)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        trend = self.combine_m(torch.cat(means, dim=1)) + res_m + trend
        return trend, season_error


class DiffusionTSCore(nn.Module):
    def __init__(self, seq_length, feature_size, n_layer_enc=3, n_layer_dec=6, d_model=128, timesteps=1000, sampling_timesteps=50, loss_type='l1', beta_schedule='cosine', n_heads=8, mlp_hidden_times=4, eta=0.0, attn_pd=0.0, resid_pd=0.0, use_ff=True, reg_weight=None):
        super().__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.eta = eta
        self.use_ff = use_ff
        self.ff_weight = default(reg_weight, math.sqrt(seq_length) / 5)
        self.model = DiffusionTSTransformer(feature_size, seq_length, n_layer_enc, n_layer_dec, d_model, n_heads, attn_pd, resid_pd, mlp_hidden_times, max_len=seq_length)
        betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.num_timesteps = int(betas.shape[0])
        self.sampling_timesteps = sampling_timesteps or self.num_timesteps
        self.fast_sampling = self.sampling_timesteps < self.num_timesteps
        self.loss_type = loss_type
        register = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register('betas', betas)
        register('alphas_cumprod', alphas_cumprod)
        register('alphas_cumprod_prev', alphas_cumprod_prev)
        register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register('posterior_variance', posterior_variance)
        register('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        register('loss_weight', torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100)

    @property
    def loss_fn(self):
        return F.l1_loss if self.loss_type == 'l1' else F.mse_loss

    def output(self, x, t):
        trend, season = self.model(x, t)
        return trend + season

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def model_predictions(self, x, t, clip_x_start=False):
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        x_start = maybe_clip(self.output(x, t))
        return self.predict_noise_from_start(x, t, x_start), x_start

    def q_posterior(self, x_start, x_t, t):
        model_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        return model_mean, extract(self.posterior_variance, t, x_t.shape), extract(self.posterior_log_variance_clipped, t, x_t.shape)

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def forward(self, x, target=None):
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        target = x if target is None else target
        noise = torch.randn_like(x)
        x_t = self.q_sample(x_start=x, t=t, noise=noise)
        model_out = self.output(x_t, t)
        train_loss = self.loss_fn(model_out, target, reduction='none')
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward').transpose(1, 2)
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward').transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss = train_loss + self.ff_weight * fourier_loss
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    @torch.no_grad()
    def fast_sample_infill(self, shape, target, partial_mask, sampling_timesteps=None, clip_denoised=True):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta
        sampling_timesteps = sampling_timesteps or self.sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1, device=device)
        time_pairs = list(zip(list(reversed(times.int().tolist()))[:-1], list(reversed(times.int().tolist()))[1:]))
        img = torch.randn(shape, device=device)
        target = target.to(device)
        partial_mask = partial_mask.to(device)
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * torch.randn_like(img)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]
        img[partial_mask] = target[partial_mask]
        return img


class DiffusionTSAdapter(nn.Module):
    is_diffusion_ts = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.full_len = int(getattr(configs, 'diffusion_ts_seq_length', self.seq_len + self.pred_len))
        self.sampling_timesteps = int(getattr(configs, 'diffusion_ts_sampling_timesteps', 50))
        self.use_ema = bool(getattr(configs, 'diffusion_ts_use_ema', True))
        self.ema_decay = float(getattr(configs, 'diffusion_ts_ema_decay', 0.995))
        self.ema_update_every = int(getattr(configs, 'diffusion_ts_ema_update_every', 10))
        self.core = DiffusionTSCore(
            seq_length=self.full_len,
            feature_size=int(getattr(configs, 'diffusion_ts_feature_size', self.enc_in)),
            n_layer_enc=int(getattr(configs, 'diffusion_ts_n_layer_enc', 3)),
            n_layer_dec=int(getattr(configs, 'diffusion_ts_n_layer_dec', 6)),
            d_model=int(getattr(configs, 'diffusion_ts_d_model', getattr(configs, 'd_model', 128))),
            timesteps=int(getattr(configs, 'diffusion_ts_timesteps', 1000)),
            sampling_timesteps=self.sampling_timesteps,
            loss_type=str(getattr(configs, 'diffusion_ts_loss_type', 'l1')),
            beta_schedule=str(getattr(configs, 'diffusion_ts_beta_schedule', 'cosine')),
            n_heads=int(getattr(configs, 'diffusion_ts_n_heads', getattr(configs, 'n_heads', 8))),
            mlp_hidden_times=int(getattr(configs, 'diffusion_ts_mlp_hidden_times', 4)),
            eta=float(getattr(configs, 'diffusion_ts_eta', 0.0)),
            attn_pd=float(getattr(configs, 'diffusion_ts_attn_pd', 0.0)),
            resid_pd=float(getattr(configs, 'diffusion_ts_resid_pd', 0.0)),
            use_ff=bool(getattr(configs, 'diffusion_ts_use_ff', True)),
            reg_weight=getattr(configs, 'diffusion_ts_reg_weight', None),
        )
        self.ema_core = copy.deepcopy(self.core)
        for param in self.ema_core.parameters():
            param.requires_grad_(False)
        self.register_buffer('ema_steps', torch.tensor(0, dtype=torch.long))

    def _build_full_series(self, x_enc, batch_y=None):
        if batch_y is None:
            future = torch.zeros(x_enc.size(0), self.pred_len, x_enc.size(-1), device=x_enc.device, dtype=x_enc.dtype)
        else:
            future = batch_y[:, -self.pred_len:, :]
        full = torch.cat([x_enc, future], dim=1)
        if full.size(1) > self.full_len:
            full = full[:, :self.full_len, :]
        elif full.size(1) < self.full_len:
            full = F.pad(full, (0, 0, 0, self.full_len - full.size(1)))
        return full

    def training_loss(self, x_enc, batch_y, batch_y_mask=None):
        full = self._build_full_series(x_enc, batch_y)
        return self.core(full, target=full)

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
        target = self._build_full_series(x_enc, None)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, :self.seq_len, :] = True
        samples = []
        times = max(1, int(sample_times))
        sampler = self.ema_core if self.use_ema and not self.training else self.core
        for _ in range(times):
            sample = sampler.fast_sample_infill(target.shape, target=target, partial_mask=mask, sampling_timesteps=self.sampling_timesteps)
            samples.append(sample[:, -self.pred_len:, :])
        all_samples = torch.stack(samples, dim=1)
        return all_samples.mean(dim=1), all_samples

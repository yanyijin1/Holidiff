import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _get_trans(heads, layers, channels):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels * 4, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class CSDIDiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x

    @staticmethod
    def _build_embedding(num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim, dtype=torch.float32) / max(dim - 1, 1) * 4.0).unsqueeze(0)
        table = steps.float() * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class CSDIResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = _conv1d_init(side_dim, 2 * channels, 1)
        self.mid_projection = _conv1d_init(channels, 2 * channels, 1)
        self.output_projection = _conv1d_init(channels, 2 * channels, 1)
        self.time_layer = _get_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = _get_trans(heads=nheads, layers=1, channels=channels)

    def _forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)

    def _forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        y = x.reshape(B, channel, K * L)
        y = y + self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = self._forward_time(y, base_shape)
        y = self._forward_feature(y, base_shape)
        y = self.mid_projection(y)
        cond_info = self.cond_projection(cond_info.reshape(B, -1, K * L))
        y = y + cond_info
        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)
        residual, skip = y.chunk(2, dim=1)
        return (x + residual.reshape(base_shape)) / math.sqrt(2.0), skip.reshape(base_shape)


class CSDIDiffModel(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = CSDIDiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = _conv1d_init(inputdim, self.channels, 1)
        self.output_projection1 = _conv1d_init(self.channels, self.channels, 1)
        self.output_projection2 = _conv1d_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList([
            CSDIResidualBlock(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"],
            )
            for _ in range(config["layers"])
        ])

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        x = F.relu(self.input_projection(x.reshape(B, inputdim, K * L)))
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skips = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_info, diffusion_emb)
            skips.append(skip)
        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.residual_layers))
        x = F.relu(self.output_projection1(x.reshape(B, self.channels, K * L)))
        return self.output_projection2(x).reshape(B, K, L)


class CSDICore(nn.Module):
    def __init__(self, target_dim, num_steps=50, layers=4, channels=64, nheads=8,
                 diffusion_embedding_dim=128, beta_start=0.0001, beta_end=0.5,
                 schedule="quad", timeemb=128, featureemb=16):
        super().__init__()
        self.target_dim = target_dim
        self.emb_time_dim = timeemb
        self.emb_feature_dim = featureemb
        emb_total_dim = timeemb + featureemb + 1
        self.embed_layer = nn.Embedding(num_embeddings=target_dim, embedding_dim=featureemb)
        config_diff = {
            "layers": layers,
            "channels": channels,
            "nheads": nheads,
            "diffusion_embedding_dim": diffusion_embedding_dim,
            "num_steps": num_steps,
            "side_dim": emb_total_dim,
        }
        self.diffmodel = CSDIDiffModel(config_diff, inputdim=2)
        self.num_steps = num_steps
        if schedule == "quad":
            self.beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        else:
            self.beta = np.linspace(beta_start, beta_end, num_steps)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer("alpha_torch", torch.tensor(self.alpha, dtype=torch.float32).unsqueeze(1).unsqueeze(1))

    def _time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = pos.unsqueeze(2)
        div_term = 1.0 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=pos.device, dtype=torch.float32) / d_model)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe

    def _get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self._time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(K, device=cond_mask.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        side_info = torch.cat([side_info, cond_mask.unsqueeze(1)], dim=1)
        return side_info

    def _calc_loss(self, observed_data, cond_mask, side_info, t):
        B = observed_data.shape[0]
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = current_alpha.sqrt() * observed_data + (1.0 - current_alpha).sqrt() * noise
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)
        predicted = self.diffmodel(total_input, side_info, t)
        target_mask = 1.0 - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum().clamp_min(1)
        return (residual ** 2).sum() / num_eval

    def training_loss(self, observed_data, cond_mask, observed_tp):
        side_info = self._get_side_info(observed_tp, cond_mask)
        t = torch.randint(0, self.num_steps, (observed_data.shape[0],), device=observed_data.device).long()
        return self._calc_loss(observed_data, cond_mask, side_info, t)

    @torch.no_grad()
    def impute(self, observed_data, cond_mask, observed_tp):
        B, K, L = observed_data.shape
        side_info = self._get_side_info(observed_tp, cond_mask)
        current_sample = torch.randn_like(observed_data)
        for t in range(self.num_steps - 1, -1, -1):
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
            diff_input = torch.cat([cond_obs, noisy_target], dim=1)
            t_tensor = torch.tensor([t], device=observed_data.device).long()
            predicted = self.diffmodel(diff_input, side_info, t_tensor)
            coeff1 = 1.0 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * predicted)
            if t > 0:
                sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                current_sample = current_sample + sigma * torch.randn_like(current_sample)
        return current_sample


class CSDIAdapter(nn.Module):
    is_csdi = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.full_len = self.seq_len + self.pred_len
        self.use_ema = bool(getattr(configs, 'csdi_use_ema', True))
        self.ema_decay = float(getattr(configs, 'csdi_ema_decay', 0.995))
        self.ema_update_every = int(getattr(configs, 'csdi_ema_update_every', 10))
        self.core = CSDICore(
            target_dim=self.enc_in,
            num_steps=int(getattr(configs, 'csdi_num_steps', 50)),
            layers=int(getattr(configs, 'csdi_layers', 4)),
            channels=int(getattr(configs, 'csdi_channels', 64)),
            nheads=int(getattr(configs, 'csdi_nheads', 8)),
            diffusion_embedding_dim=int(getattr(configs, 'csdi_diffusion_embedding_dim', 128)),
            beta_start=float(getattr(configs, 'csdi_beta_start', 0.0001)),
            beta_end=float(getattr(configs, 'csdi_beta_end', 0.5)),
            schedule=str(getattr(configs, 'csdi_schedule', 'quad')),
            timeemb=int(getattr(configs, 'csdi_timeemb', 128)),
            featureemb=int(getattr(configs, 'csdi_featureemb', 16)),
        )
        self.ema_core = copy.deepcopy(self.core)
        for p in self.ema_core.parameters():
            p.requires_grad_(False)
        self.register_buffer('ema_steps', torch.tensor(0, dtype=torch.long))

    def _prepare_batch(self, x_enc, batch_y):
        # x_enc: (B, seq_len, N), batch_y: (B, label+pred, N)
        # Build observed_data: (B, N, full_len)
        history = x_enc.permute(0, 2, 1)  # (B, N, seq_len)
        future = batch_y[:, -self.pred_len:, :].permute(0, 2, 1)  # (B, N, pred_len)
        observed_data = torch.cat([history, future], dim=2)  # (B, N, full_len)
        # cond_mask: history=1, future=0 → (B, N, full_len)
        B, N, L = observed_data.shape
        cond_mask = torch.zeros(B, N, L, device=observed_data.device)
        cond_mask[:, :, :self.seq_len] = 1.0
        # observed_tp: (B, L)
        observed_tp = torch.arange(L, device=observed_data.device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        return observed_data, cond_mask, observed_tp

    def training_loss(self, x_enc, batch_y, batch_y_mask=None):
        observed_data, cond_mask, observed_tp = self._prepare_batch(x_enc, batch_y)
        return self.core.training_loss(observed_data, cond_mask, observed_tp)

    @torch.no_grad()
    def after_optimizer_step(self):
        if not self.use_ema:
            return
        self.ema_steps += 1
        if int(self.ema_steps.item()) % max(self.ema_update_every, 1) != 0:
            return
        for ema_p, p in zip(self.ema_core.parameters(), self.core.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)
        for ema_b, b in zip(self.ema_core.buffers(), self.core.buffers()):
            ema_b.copy_(b)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None,
                dec_self_mask=None, dec_enc_mask=None, sample_times=1, holiday_flag=None):
        sample_core = self.ema_core if self.use_ema and not self.training else self.core
        observed_data, cond_mask, observed_tp = self._prepare_batch(x_enc, x_dec)
        samples = []
        for _ in range(max(1, int(sample_times))):
            imputed = sample_core.impute(observed_data, cond_mask, observed_tp)
            samples.append(imputed[:, :, self.seq_len:].permute(0, 2, 1))  # (B, pred_len, N)
        all_samples = torch.stack(samples, dim=1)  # (B, n_samples, pred_len, N)
        return all_samples.mean(dim=1), all_samples

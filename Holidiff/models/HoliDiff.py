
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
import torch.nn as nn
from Holidiff.micro.tek import TEK
from Holidiff.macro.dpm_sampler import DPMSolverSampler
from Holidiff.macro.agg import build_macro_aggregator
from Holidiff.models.simdiff.revin import RevIN
from Holidiff.utils.diffusion_utils import *
from Holidiff.frequency.build import build_frequency_patch_embed



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


class HATEK(nn.Module):

    def __init__(self, configs):
        super(HATEK, self).__init__()

        self.configs = configs
        self.device = torch.device(f'cuda:{configs.gpu}' if torch.cuda.is_available() and getattr(configs, 'use_gpu', False) else 'cpu')
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
        self.aggregation_mode = str(getattr(configs, 'aggregation_mode', 'mom')).lower()
        self.use_holidiff_lstde = bool(getattr(configs, 'use_holidiff_lstde', True))
        self.use_sfcn = bool(getattr(configs, 'use_sfcn', True))
        self.frequency_patch_embed = build_frequency_patch_embed(configs) if self.use_sfcn else None
        self.macro_aggregator = build_macro_aggregator(configs)
        self.tek = TEK(configs)
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.enc_in = configs.enc_in
        self.batch_size =configs.batch_size
        self.beta_start = 1e-4 # 1e4
        self.beta_end = 1e-1#2e-2
        self.beta_schedule = 'cosine'
        self.v_posterior = 0.0
        self.loss_type = "l1"
        self.set_micro_uncertainty_schedule(None, self.beta_schedule, self.diff_steps, self.beta_start, self.beta_end)
        self.total_N = len(self.alphas_cumprod)
        self.T = 1.
        self.eps = 1e-5
        self.nn = self.tek
        self.sampler = DPMSolverSampler(configs,self.nn, self.device,self.alphas_cumprod,self.betas.device)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5, holiday_flag=None, future_target=None,
                mask_band=None, preset_noises=None):
        if self.training:
            return self.forward_micro_generation_train(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                             enc_self_mask, dec_self_mask, dec_enc_mask, holiday_flag=holiday_flag, future_target=future_target,
                                             mask_band=mask_band)
        else:
            return self.forward_consensus_inference(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                            enc_self_mask, dec_self_mask, dec_enc_mask, sample_times, holiday_flag=holiday_flag,
                                            mask_band=mask_band, preset_noises=preset_noises)


    def forward_micro_generation_train(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, holiday_flag=None, future_target=None, mask_band=None):

        x = x_dec[:, -self.configs.pred_len:, :].permute(0, 2, 1)
        if x.shape[-1] < self.patch_len:
            pad_len = self.patch_len - x.shape[-1]
            x = F.pad(x, (0, pad_len), mode='replicate')
        f_dim = -1 if self.configs.features in ['MS'] else 0
        x = x[:, f_dim:, :]
        if self.use_holidiff_lstde:
            cond_ts = x_enc.permute(0, 2, 1)
            raw_history = cond_ts
            mean_ = torch.mean(cond_ts, dim=-1, keepdims=True)
            std_ = torch.std(cond_ts, dim=-1, keepdims=True)
            cond_ts = (cond_ts - mean_) / (std_ + 0.00001)
            x = (x - mean_) / (std_ + 0.00001)
            denorm_mean = mean_
            denorm_std = std_
            use_revin_denorm = False
        else:
            cond_ts = self.revin_layer(x_enc, 'norm').permute(0, 2, 1)
            raw_history = x_enc.permute(0, 2, 1)
            node_count = x.shape[1]
            mean_ = torch.mean(x[:, -node_count:, :], dim=1).unsqueeze(1)
            std_ = torch.ones_like(torch.std(x, dim=1).unsqueeze(1))
            x = (x - mean_.repeat(1, node_count, 1)) / (std_.repeat(1, node_count, 1) + 0.00001)
            denorm_mean = None
            denorm_std = None
            use_revin_denorm = True
        B = np.shape(x)[0]
        N = self.configs.enc_in
        L1 = np.shape(cond_ts)[2]
        L2 = np.shape(x)[2]
        target_len = self.configs.pred_len
        cond_ts = torch.reshape(cond_ts, (B * N, L1))
        x = torch.reshape(x, (B * N, L2))
        frequency_patch_embedding = None
        if self.frequency_patch_embed is not None:
            frequency_patch_embedding = self.frequency_patch_embed(raw_history=raw_history, enc_in=self.configs.enc_in)
        t = torch.randint(0, self.num_timesteps, size=[B * N // 2,]).long().to(self.device)
        t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)
        noise = torch.randn_like(x)
        x_k = self.generate_micro_realization(x_start=x, t=t, noise=noise)
        model_out = self.nn(x_k, t, cond_ts, x_mark_enc, raw_history=raw_history, frequency_patch_embedding=frequency_patch_embedding, mask_band=mask_band)
        model_out = torch.reshape(model_out, (B, N, target_len))
        if use_revin_denorm:
            model_out = self.revin_layer(model_out.permute(0, 2, 1), 'denorm').permute(0, 2, 1)
        else:
            model_out = model_out * (denorm_std + 0.00001) + denorm_mean
        weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(B, N, 1)
        model_out = model_out.permute(0, 2, 1)
        return model_out, weight_tmp

    def forward_consensus_inference(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5, holiday_flag=None, mask_band=None, preset_noises=None):

        history_for_trend = x_enc
        x_past = x_enc.permute(0, 2, 1)
        batchs, nF, nL = np.shape(x_past)[0], self.enc_in, self.pred_len
        batchs = batchs * self.enc_in
        if self.configs.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        all_outs = []
        B = np.shape(x_past)[0]
        N = self.configs.enc_in
        if self.use_holidiff_lstde:
            mean_ = torch.mean(x_past, dim=-1, keepdims=True)
            std_ = torch.std(x_past, dim=-1, keepdims=True)
            raw_history = x_past.clone()
            mean_ = mean_.to(self.betas.device)
            std_ = std_.to(self.betas.device)
            x_past = (x_past - mean_) / (std_ + 0.00001)
            use_revin_denorm = False
        else:
            x_past = self.revin_layer(x_enc, 'norm').permute(0, 2, 1)
            raw_history = x_enc.permute(0, 2, 1).to(self.betas.device)
            x_past = x_past.to(self.betas.device)
            mean_ = None
            std_ = None
            use_revin_denorm = True
        x_past = torch.reshape(x_past, (B * N, -1)).to(self.betas.device)
        frequency_patch_embedding = None
        if self.frequency_patch_embed is not None:
            frequency_patch_embedding = self.frequency_patch_embed(raw_history=raw_history, enc_in=self.configs.enc_in)
        for i in range(sample_times):
            if preset_noises is not None:
                start_code = preset_noises[i].to(self.betas.device)
            else:
                start_code = torch.randn((batchs, nL), device=self.betas.device)
            diff_samples, _ = self.sampler.sample(
                S=self.configs.s_steps,
                conditioning=x_past,
                x_mark_enc=x_mark_enc,
                batch_size=batchs,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None,
                eta=0.,
                x_T=start_code,
                raw_history=raw_history,
                frequency_patch_embedding=frequency_patch_embedding,
                mask_band=mask_band,
            )
            diff_samples = diff_samples.to(self.betas.device)
            diff_samples = torch.reshape(diff_samples, (B, N, -1))
            if use_revin_denorm:
                diff_samples = self.revin_layer(diff_samples.permute(0, 2, 1), 'denorm')
            else:
                diff_samples = diff_samples * (std_ + 0.00001) + mean_
                diff_samples = diff_samples.permute(0, 2, 1)
            all_outs.append(diff_samples)
        all_outs = torch.stack(all_outs, dim=0)
        outs = self._aggregate_samples(all_outs, history_context=history_for_trend)

        return outs,all_outs.permute(1,0,2,3)

    def set_micro_uncertainty_schedule(self, given_betas=None, beta_schedule="linear", diff_steps=1000, beta_start=1e-4, beta_end=2e-2
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

    def generate_micro_realization(self, x_start, t, noise=None):

        noise = default(noise, lambda: self.scaling_noise * torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def _consensus_mean(self, seq):
        return torch.sum(seq, dim=0) / seq.size(0)

    def _aggregate_samples(self, all_outs: torch.Tensor, history_context: torch.Tensor | None = None) -> torch.Tensor:
        if all_outs.size(0) <= 1:
            return all_outs.mean(0)

        mode = self.aggregation_mode
        if mode in {'simple', 'single'}:
            return all_outs.mean(0)
        if mode == 'median':
            return torch.median(all_outs, dim=0)[0]
        if mode == 'mom':
            return self.extract_consensus(all_outs)
        if self.macro_aggregator is not None and hasattr(self.macro_aggregator, 'aggregate_batch'):
            return self.macro_aggregator.aggregate_batch(all_outs, history_context=history_context)

        return all_outs.mean(0)

    def _consensus_reduce(self, tensor):
        if self.n_blocks > tensor.size(0):
            self.n_blocks = int(torch.ceil(tensor.size(0) / 2))

        indic = torch.randperm(tensor.size(0))
        tensor = tensor[indic]
        block_size = tensor.size(0) // self.n_blocks

        means = []
        for i in range(self.n_blocks):
            start_index = i * block_size
            end_index = start_index + block_size if (i + 1) < self.n_blocks else tensor.size(0)
            block = tensor[start_index:end_index]
            block_mean = self._consensus_mean(block)
            means.append(block_mean)

        means = torch.stack(means)
        return torch.median(means, dim=0)[0]

    def extract_consensus(self, outputs):
        results = []
        for _ in range(self.rmom_n):
            shuffled_outputs = outputs[torch.randperm(outputs.size(0))]
            result = self._consensus_reduce(shuffled_outputs)
            results.append(result)
        results = torch.stack(results)

        return torch.median(results, dim=0)[0]

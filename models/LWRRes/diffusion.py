"""
扩散调度模块 - 管理噪声调度和采样参数
"""
import torch
import numpy as np
from functools import partial


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


def setup_noise_schedule(model, configs, beta_schedule="cosine", diff_steps=1000, beta_start=1e-4, beta_end=2e-2):
    betas = cosine_beta_schedule(diff_steps, configs.coss)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    timesteps, = betas.shape
    model.num_timesteps = int(timesteps)
    model.linear_start = beta_start
    model.linear_end = beta_end

    to_torch = partial(torch.tensor, dtype=torch.float32)

    model.register_buffer('betas', to_torch(betas))
    model.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
    model.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

    model.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
    model.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
    model.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
    model.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
    model.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    v_posterior = getattr(model, 'v_posterior', 0.0)
    posterior_variance = (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + v_posterior * betas
    model.register_buffer('posterior_variance', to_torch(posterior_variance))
    model.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
    model.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
    model.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    lvlb_weights = 0.8 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
    lvlb_weights[0] = lvlb_weights[1]
    model.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
    assert not torch.isnan(model.lvlb_weights).all()


class DiffusionAggregator:
    """扩散采样聚合器 - 提供样本聚合方法"""

    def __init__(self, model):
        self.model = model
        self.rmom_n = model.rmom_n
        self.n_blocks = model.n_blocks

    def _emp_mean(self, seq):
        return torch.sum(seq, dim=0) / seq.size(0)

    def _median_of_means(self, tensor):
        if self.n_blocks > tensor.size(0):
            self.n_blocks = int(torch.ceil(torch.tensor(tensor.size(0) / 2, device=tensor.device)).item())

        indic = torch.randperm(tensor.size(0))
        tensor = tensor[indic]
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

    def aggregate(self, all_outs):
        m = all_outs.size(0)
        use_mom = bool(getattr(self.model.configs, 'use_mom', 1))
        mode = str(getattr(self.model.configs, 'mom_mode', 'mom')).lower()

        if m <= 1:
            return all_outs.mean(0)

        if mode == 'mean':
            return all_outs.mean(0)

        if use_mom:
            return self._rob_median_of_means(all_outs)

        return all_outs.mean(0)

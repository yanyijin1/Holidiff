# Version: v0.1-flow-matching
# Date: 2026-04-07
# Description: Flow Matching 扩散核心实现 + MoM聚合

import torch
import torch.nn as nn
import numpy as np
from functools import partial


def cosine_beta_schedule(timesteps, s=5):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract_into_tensor(a, t, x_shape):
    """Extract values from tensor at indices and reshape"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class FlowMatchingScheduler:
    """Flow Matching / Diffusion Scheduler"""
    
    def __init__(self, configs, beta_start=1e-4, beta_end=1e-1, v_posterior=0.0):
        self.configs = configs
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.v_posterior = v_posterior
        
    def setup_schedule(self, diff_steps, coss=0.008):
        """Setup noise schedule based on cosine beta schedule"""
        betas = cosine_beta_schedule(diff_steps, s=coss)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        
        timesteps = betas.shape[0]
        to_torch = partial(torch.tensor, dtype=torch.float32)
        
        buffers = {
            'betas': to_torch(betas),
            'alphas_cumprod': to_torch(alphas_cumprod),
            'alphas_cumprod_prev': to_torch(alphas_cumprod_prev),
            'sqrt_alphas_cumprod': to_torch(np.sqrt(alphas_cumprod)),
            'sqrt_one_minus_alphas_cumprod': to_torch(np.sqrt(1. - alphas_cumprod)),
            'log_one_minus_alphas_cumprod': to_torch(np.log(1. - alphas_cumprod)),
            'sqrt_recip_alphas_cumprod': to_torch(np.sqrt(1. / alphas_cumprod)),
            'sqrt_recipm1_alphas_cumprod': to_torch(np.sqrt(1. / alphas_cumprod - 1)),
        }
        
        # Posterior variance calculation
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        buffers['posterior_variance'] = to_torch(posterior_variance)
        buffers['posterior_log_variance_clipped'] = to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        buffers['posterior_mean_coef1'] = to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        buffers['posterior_mean_coef2'] = to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        return buffers
        
    def noise_ts(self, x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
        """Add noise to x_start at timestep t"""
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


class MoMAggregator:
    """
    Mean of Means Aggregation for Multiple Samples
    
    Provides multiple aggregation strategies:
    - mean: simple average
    - mom: median of means
    - cmom: clustered mean of means  
    - mmmom: mini-batch MoM
    """
    
    def __init__(self, configs):
        self.configs = configs
        self.n_blocks = configs.n_b
        
    def _emp_mean(self, seq):
        """Empirical mean across first dimension"""
        return torch.sum(seq, dim=0) / seq.size(0)
    
    def _median_of_means(self, tensor):
        """Median of Means aggregation"""
        n_blocks = self.n_blocks
        if n_blocks > tensor.size(0):
            n_blocks = int(torch.ceil(tensor.size(0) / 2))
            
        indic = torch.randperm(tensor.size(0))
        tensor = tensor[indic]
        block_size = tensor.size(0) // n_blocks
        
        means = []
        for i in range(n_blocks):
            start_index = i * block_size
            end_index = start_index + block_size if (i + 1) < n_blocks else tensor.size(0)
            block = tensor[start_index:end_index]
            block_mean = self._emp_mean(block)
            means.append(block_mean)
            
        means = torch.stack(means)
        return torch.median(means, dim=0)[0]
    
    def _rob_median_of_means(self, outputs):
        """Robust Median of Means (multiple shuffles)"""
        rmom_n = self.configs.rmom
        results = []
        for _ in range(rmom_n):
            shuffled_outputs = outputs[torch.randperm(outputs.size(0))]
            result = self._median_of_means(shuffled_outputs)
            results.append(result)
        results = torch.stack(results)
        return self._emp_mean(results)
    
    def aggregate(self, all_outs):
        """
        Aggregate multiple samples using configured method.
        
        Args:
            all_outs: tensor of shape (M, B, T, N) where M is number of samples
        """
        m = all_outs.size(0)
        use_mom = bool(getattr(self.configs, 'use_mom', 1))
        mode = str(getattr(self.configs, 'mom_mode', 'mom')).lower()
        
        if m <= 1:
            return all_outs.mean(0)
            
        if mode == 'mean':
            return all_outs.mean(0)
            
        if use_mom:
            return self._rob_median_of_means(all_outs)
        return all_outs.mean(0)

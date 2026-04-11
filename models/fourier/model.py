"""
主模型模块 - LWRDiff Model

包含训练和采样推理的前向传播
"""
import torch
import torch.nn as nn
import numpy as np
from functools import partial

from layers.RevIN import RevIN
from layers.samplers.dpm_sampler import DPMSolverSampler
from .diffusion import cosine_beta_schedule, DiffusionAggregator
from .unet_bone import PatchUVIT


class Model(nn.Module):
    """
    LWRDiff 主模型
    
    支持训练模式和采样推理模式
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.diff_steps = configs.diff_steps
        self.stride = configs.stride
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.num_heads = configs.num_heads
        self.rmom_n = configs.rmom
        self.n_blocks = configs.n_b
        self.enc_in = configs.enc_in
        self.batch_size = configs.batch_size

        # 初始化 U-Net 主干
        self.nn = PatchUVIT(configs)
        
        # 扩散参数
        self.beta_start = 1e-4
        self.beta_end = 1e-1
        self.beta_schedule = 'cosine'
        self.v_posterior = 0.0
        self.loss_type = "l1"
        
        # 设置噪声调度
        self._setup_diffusion_schedule()
        
        # 初始化采样器
        self.sampler = DPMSolverSampler(
            configs, 
            self.nn, 
            self.device, 
            self.alphas_cumprod, 
            self.betas.device
        )
        
        # RevIN 归一化层
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)
        
        # 聚合器
        self.aggregator = DiffusionAggregator(self)

    def _setup_diffusion_schedule(self):
        """设置扩散过程的噪声调度表"""
        betas = cosine_beta_schedule(self.diff_steps, self.configs.coss)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = self.beta_start
        self.linear_end = self.beta_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
        lvlb_weights = 0.8 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):
        """前向传播，根据模式选择训练或推理"""
        if self.training:
            return self.forward_train(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                             enc_self_mask, dec_self_mask, dec_enc_mask)
        else:
            return self.forward_val_test(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                            enc_self_mask, dec_self_mask, dec_enc_mask, sample_times)

    def forward_train(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        训练模式：添加噪声并预测
        
        Args:
            x_enc: (B, seq_len, N) - 编码输入
            x_dec: (B, label_len+pred_len, N) - 解码输入
            ...
        
        Returns:
            model_out: (B, pred_len, N) - 预测输出
            weight_tmp: (B*N, 1, 1) - 噪声权重
        """
        # 取 target（只取 pred_len）
        x_target = x_dec[:, -self.configs.pred_len:, :]   # (B, 12, N)

        # 归一化
        x = x_target.permute(0, 2, 1)                    # (B, N, 12)
        cond_ts = x_enc.permute(0, 2, 1)                 # (B, N, 96)

        mean_ = cond_ts.mean(dim=(1, 2), keepdim=True)
        std_  = cond_ts.std(dim=(1, 2), keepdim=True) + 1e-5
        x_norm = (x - mean_) / std_                      # (B, N, 12)
        cond_norm = (cond_ts - mean_) / std_            # (B, N, 96)

        # 加噪
        B, N, dec_L = x_norm.shape
        t = torch.randint(0, self.num_timesteps,
                          size=[B * N // 2]).long().to(self.device)
        t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)  # (B*N,)
        noise = torch.randn_like(x_norm)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(B, N, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].reshape(B, N, 1)
        x_k = sqrt_alpha * x_norm + sqrt_one_minus * noise

        # FormerBone 前向
        x_k_flat = x_k.reshape(B * N, dec_L)
        cond_flat = cond_norm.reshape(B * N, -1)
        model_out_flat = self.nn(x_k_flat, t, cond_flat)

        # 反归一化
        model_out = model_out_flat.reshape(B, N, dec_L)
        model_out = model_out * std_ + mean_

        weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(B * N, 1, 1)

        # trainer 期望 (B, pred_len, N)
        model_out = model_out.permute(0, 2, 1)

        return model_out, weight_tmp

    def forward_val_test(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):
        """
        推理模式：多步采样
        
        Args:
            x_enc: (B, seq_len, N) - 编码输入
            x_dec: (B, label_len+pred_len, N) - 解码输入（取前label_len作为上下文）
            sample_times: 采样次数
        
        Returns:
            outs: (B, pred_len, N) - 聚合后的预测
            all_outs: (B, M, pred_len, N) - 所有采样结果
        """
        x_future = x_dec[:, -self.configs.pred_len:, :].permute(0, 2, 1)
        x_past = x_enc.permute(0, 2, 1)
        
        batchs, nF, nL = np.shape(x_past)[0], self.enc_in, self.pred_len
        batchs = batchs * self.enc_in
        if self.configs.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        
        all_outs = []
        B = np.shape(x_past)[0]
        N = self.configs.enc_in
        
        # 归一化
        if self.configs.new_norm:
            x_past = x_past.permute(0, 2, 1)
            x_past = self.revin_layer(x_past, 'norm')
            x_past = x_past.permute(0, 2, 1)
        else:
            mean_ = torch.mean(x_past, dim=-1, keepdims=True)
            std_ = torch.std(x_past, dim=-1, keepdims=True)
            x_past = (x_past - mean_) / (std_ + 0.00001)
        
        x_past = torch.reshape(x_past, (B * N, -1))
        x_past = x_past.to(device=self.device)
        
        for i in range(sample_times):
            start_code = torch.randn((batchs, nL), device=self.device)
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
                x_T=start_code
            )
            
            diff_samples = torch.reshape(diff_samples, (B, N, -1))
            
            # 反归一化
            if self.configs.new_norm:
                diff_samples = diff_samples.permute(0, 2, 1)
                diff_samples = self.revin_layer(diff_samples, 'denorm')
            else:
                diff_samples = diff_samples * (std_ + 0.00001) + mean_
                diff_samples = diff_samples.permute(0, 2, 1)
            
            # 只取最后 pred_len 步
            diff_samples = diff_samples[:, -self.configs.pred_len:, :]
            all_outs.append(diff_samples)
        
        all_outs = torch.stack(all_outs, dim=0)  # (M, B, pred_len, N)
        outs = self.aggregator.aggregate(all_outs)

        return outs, all_outs.permute(1, 0, 2, 3)

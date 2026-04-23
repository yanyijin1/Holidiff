"""
LWRDiff Model - 物理约束残差扩散模型（修正版）

核心修正：
1. forward_for_diffusion 接口与 DPM-Solver 严格对齐
2. forward_val_test 保持双分支（扩散采样 / 直接推理）
3. 清理数值不稳定点（clamp、eps、reshape 一致性）
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial

from layers.RevIN import RevIN
from layers.samplers.dpm_sampler import DPMSolverSampler
from .diffusion import cosine_beta_schedule, DiffusionAggregator
from .FormerBone import PatchUVIT_STFormer


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.diff_steps = configs.diff_steps
        self.enc_in = configs.enc_in
        self.rmom_n = configs.rmom
        self.n_blocks = configs.n_b

        self.nn = PatchUVIT_STFormer(configs)

        # 打印模型基本信息
        print("\n" + "="*60)
        print("[LWRRes Model] Initializing (DEBUG MODE)")
        print(f"  seq_len={self.seq_len}, pred_len={self.pred_len}")
        print(f"  d_model={configs.d_model}, enc_in={self.enc_in}")
        print(f"  diffusion steps={self.diff_steps}")
        print(f"  is_diff={configs.is_diff}, new_norm={configs.new_norm}")
        print(f"  use_lwrres={getattr(configs, 'use_lwrres', False)}")

        # 扩散参数
        self.beta_start = 1e-4
        self.beta_end = 1e-1
        self.v_posterior = 0.0

        self._setup_diffusion_schedule()

        # DPM-Solver 采样器
        self.sampler = DPMSolverSampler(
            configs, self.nn, self.device,
            self.alphas_cumprod, self.betas
        )

        # RevIN 归一化
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)

        # 聚合器
        self.aggregator = DiffusionAggregator(self)

        self._pcgk_cache = None

        # 打印物理参数
        try:
            for i, layer in enumerate(self.nn.model.st_layers):
                pcgk = layer.pcgk
                print(f"  [Layer {i}] v_critical_init={pcgk.v_critical.item():.4f}, "
                      f"temperature_init={pcgk.temperature.item():.4f}, "
                      f"alpha_init={pcgk.alpha.item():.4f}")
        except Exception as e:
            print(f"  [Phys Params] Error reading: {e}")
        print("="*60)

    def _setup_diffusion_schedule(self):
        """Cosine beta schedule + 完整 posterior 统计量"""
        betas = cosine_beta_schedule(
            self.diff_steps, getattr(self.configs, 'coss', 0.008)
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped',
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def noise_ts(self, x_start, t, noise=None):
        """标准前向加噪：x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε"""
        if noise is None:
            noise = torch.randn_like(x_start)

        target_device = x_start.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(target_device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(target_device)
        t = t.to(target_device)

        alpha_t = sqrt_alpha.gather(-1, t).reshape(*x_start.shape[:-1], 1)
        one_minus_t = sqrt_one_minus.gather(-1, t).reshape(*x_start.shape[:-1], 1)

        return alpha_t * x_start + one_minus_t * noise

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                sample_times=5, flow_x=None, flow_y=None):
        if self.training:
            return self.forward_train(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                     flow_x=flow_x, flow_y=flow_y)
        else:
            return self.forward_val_test(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                        sample_times=sample_times)

    def forward_for_diffusion(self, x, t, cond, x_mark_enc=None, **kwargs):
        """
        DPM-Solver 回调接口。
        必须严格返回 (B*N, pred_len) 的 x0 预测。
        """
        B_N = x.shape[0]
        N = self.enc_in
        B = B_N // N

        # reshape: (B*N, pred_len) -> (B, pred_len, N)
        x_perm = x.reshape(B, N, -1).permute(0, 2, 1)
        cond_perm = cond.reshape(B, N, -1).permute(0, 2, 1)

        # 归一化条件
        if self.configs.new_norm:
            cond_normed = self.revin_layer(cond_perm, 'norm')
        else:
            mean_ = cond_perm.mean(dim=1, keepdim=True)
            std_ = cond_perm.std(dim=1, keepdim=True) + 1e-5
            cond_normed = (cond_perm - mean_) / std_

        cond_flat = cond_normed.permute(0, 2, 1).reshape(B * N, -1)

        # 时间步
        t_int = t.long().clamp(0, self.diff_steps - 1)

        # 主干网络前向（使用 PCGK 缓存）
        pcgk_cache = getattr(self, '_pcgk_cache', None)
        model_out_flat = self.nn(x, t_int, cond_flat, x_mark_enc, pcgk_cache=pcgk_cache)

        # reshape 并反归一化
        model_out = model_out_flat.reshape(B, N, -1).permute(0, 2, 1)

        if self.configs.new_norm:
            model_out = self.revin_layer(model_out, 'denorm')
        else:
            model_out = model_out * std_ + mean_

        return model_out.permute(0, 2, 1).reshape(B * N, -1)

    def forward_train(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                     enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                     flow_x=None, flow_y=None):
        """
        训练模式：加噪 -> 预测 x0 -> 反归一化 -> 返回加权 loss 系数
        """
        x_target = x_dec[:, -self.configs.pred_len:, :]
        B, _, N = x_enc.shape
        pred_len = self.pred_len

        # 准备物理观测
        # q_obs: 流量 → 用于 PCGK 强度门控
        # v_obs: 速度（真实速度数据 x_enc）→ 用于相态检测
        if flow_x is not None:
            q_obs = flow_x.permute(0, 2, 1)   # (B, N, L) 流量
        else:
            q_obs = x_enc.permute(0, 2, 1)

        v_obs = x_enc.permute(0, 2, 1)         # (B, N, L) 速度 ← 真实速度数据！

        # Debug print once per training run
        if not hasattr(self, '_train_print_done'):
            self._train_print_done = True
            print(f"[forward_train] B={B}, N={N}, pred_len={pred_len}, "
                  f"x_enc={x_enc.shape}, x_target={x_target.shape}, "
                  f"flow_x={flow_x.shape if flow_x is not None else None}")

        if self.configs.new_norm:
            # RevIN 归一化（条件与目标独立统计量）
            cond_ts = self.revin_layer(x_enc, 'norm').permute(0, 2, 1)

            x = x_target.permute(0, 2, 1)
            mean_ = x.mean(dim=1, keepdim=True)
            std_ = torch.ones_like(x.std(dim=1, keepdim=True))
            x_norm = (x - mean_) / (std_ + 1e-5)

            B, N, L1 = cond_ts.shape
            _, _, L2 = x_norm.shape

            cond_flat = cond_ts.reshape(B * N, L1)
            x_flat = x_norm.reshape(B * N, L2)

            # 物理观测归一化
            q_obs_norm = self.revin_layer(q_obs.permute(0, 2, 1), 'norm').permute(0, 2, 1)
            q_obs_flat = q_obs_norm.reshape(B * N, -1)
            # v_obs: 真实速度，用 RevIN 归一化
            v_obs_norm = self.revin_layer(v_obs.permute(0, 2, 1), 'norm').permute(0, 2, 1)
            v_obs_flat = v_obs_norm.reshape(B * N, -1)

            # 时间步对称采样（SimDiff 技巧）
            t = torch.randint(0, self.diff_steps, size=[B * N // 2]).long().to(x_enc.device)
            t = torch.cat([t, self.diff_steps - 1 - t], dim=0)
            noise = torch.randn_like(x_flat)
            x_k = self.noise_ts(x_start=x_flat, t=t, noise=noise)

            # 模型预测 x0
            model_out_flat = self.nn(
                x_k, t, cond_flat, x_mark_enc,
                q_obs=q_obs_flat, v_obs=v_obs_flat
            )

            # 反归一化
            model_out = model_out_flat.reshape(B, N, L2).permute(0, 2, 1)
            model_out = self.revin_layer(model_out, 'denorm')

            # 加权系数（SimDiff 加权 MAE）
            weight_tmp = self.sqrt_one_minus_alphas_cumprod.to(x_flat.device)[t]\
                            .reshape(B * N, 1, 1)

        else:
            # 全局标准化
            x = x_target.permute(0, 2, 1)
            cond_ts = x_enc.permute(0, 2, 1)

            mean_ = cond_ts.mean(dim=(1, 2), keepdim=True)
            std_ = cond_ts.std(dim=(1, 2), keepdim=True) + 1e-5
            x_norm = (x - mean_) / std_
            cond_norm = (cond_ts - mean_) / std_

            B, N, dec_L = x_norm.shape
            t = torch.randint(0, self.diff_steps, size=[B * N // 2]).long().to(x_enc.device)
            t = torch.cat([t, self.diff_steps - 1 - t], dim=0)
            noise = torch.randn_like(x_norm)

            sqrt_alpha = self.sqrt_alphas_cumprod.to(x_norm.device)[t].reshape(B, N, 1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x_norm.device)[t].reshape(B, N, 1)
            x_k = sqrt_alpha * x_norm + sqrt_one_minus * noise

            x_k_flat = x_k.reshape(B * N, dec_L)
            cond_flat = cond_norm.reshape(B * N, -1)

            q_obs_norm = (q_obs - mean_) / std_
            q_obs_flat = q_obs_norm.reshape(B * N, -1)
            # v_obs: 真实速度，用全局标准化
            v_obs_norm = (v_obs - mean_) / std_
            v_obs_flat = v_obs_norm.reshape(B * N, -1)

            model_out_flat = self.nn(x_k_flat, t, cond_flat, q_obs=q_obs_flat, v_obs=v_obs_flat)

            model_out = model_out_flat.reshape(B, N, dec_L)
            model_out = model_out * std_ + mean_
            model_out = model_out.permute(0, 2, 1)

            weight_tmp = self.sqrt_one_minus_alphas_cumprod.to(x_k_flat.device)[t]\
                            .reshape(B * N, 1, 1)

        return model_out, weight_tmp

    def forward_val_test(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                        sample_times=5):
        """
        推理模式：
        - is_diff=1: 多步扩散采样（DPM-Solver）+ MoM 聚合
        - is_diff=0: 直接确定性推理（t=0，无采样）
        """
        x_past = x_enc.permute(0, 2, 1)
        B, N, _ = x_past.shape
        batchs = B * N
        nL = self.pred_len

        if not hasattr(self, '_valtest_print_done'):
            self._valtest_print_done = True
            print(f"[forward_val_test] B={B}, N={N}, batchs={batchs}, "
                  f"x_enc={x_enc.shape}, is_diff={self.configs.is_diff}")

        if self.configs.features in ['MS']:
            nF = 1
        else:
            nF = self.enc_in
        shape = [nF, nL]

        # 归一化条件历史
        if self.configs.new_norm:
            x_past_orig = x_past.permute(0, 2, 1)
            x_past_normed = self.revin_layer(x_past_orig, 'norm')
            x_past_for_forward = x_past_normed.permute(0, 2, 1)
            std_ = None  # new_norm 下在反归一化时内部处理
            mean_ = None
        else:
            mean_ = x_past.mean(dim=-1, keepdim=True)
            std_ = x_past.std(dim=-1, keepdim=True)
            x_past_for_forward = (x_past - mean_) / (std_ + 1e-5)

        x_past_flat = x_past_for_forward.reshape(B * N, -1).to(self.device)

        if self.configs.is_diff:
            # ========== 扩散采样模式 ==========
            self._pcgk_cache = self.nn.precompute_pcgk(x_past_flat)

            all_outs = []
            for i in range(sample_times):
                start_code = torch.randn((batchs, nL), device=self.device)
                diff_samples, _ = self.sampler.sample(
                    S=self.configs.s_steps,
                    conditioning=x_past_flat,
                    x_mark_enc=x_mark_enc,
                    batch_size=batchs,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=1.0,
                    unconditional_conditioning=None,
                    eta=0.0,
                    x_T=start_code
                )

                diff_samples = diff_samples.reshape(B, N, -1)

                if self.configs.new_norm:
                    diff_samples = diff_samples.permute(0, 2, 1)
                    diff_samples = self.revin_layer(diff_samples, 'denorm')
                else:
                    diff_samples = diff_samples * (std_ + 1e-5) + mean_
                    diff_samples = diff_samples.permute(0, 2, 1)

                all_outs.append(diff_samples)

            all_outs = torch.stack(all_outs, dim=0)          # (M, B, pred_len, N)
            outs = self.aggregator.aggregate(all_outs)       # (B, pred_len, N)
            self._pcgk_cache = None

            return outs, all_outs.permute(1, 0, 2, 3)

        else:
            # ========== 直接推理模式（确定性，无扩散）==========
            t = torch.zeros(B * N, dtype=torch.long, device=x_enc.device)

            # 物理观测复用条件历史
            q_obs_flat = x_past_flat
            v_obs_flat = x_past_flat

            model_out_flat = self.nn(
                x_past_flat, t, x_past_flat, x_mark_enc,
                q_obs=q_obs_flat, v_obs=v_obs_flat
            )

            model_out = model_out_flat.reshape(B, N, -1).permute(0, 2, 1)

            if self.configs.new_norm:
                model_out = self.revin_layer(model_out, 'denorm')
            else:
                model_out = model_out * (std_ + 1e-5) + mean_

            weight_tmp = torch.ones(B * N, 1, 1, device=x_enc.device)
            return model_out, weight_tmp

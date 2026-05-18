import math

import torch
import torch.nn as nn


class PhysicalInjectionModule(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.enable = bool(getattr(configs, 'physical_injection_enable', False))
        self.attention_enable = bool(getattr(configs, 'physical_attention_enable', False))
        self.residual_enable = bool(getattr(configs, 'physical_residual_enable', False))
        self.residual_in_train = bool(getattr(configs, 'physical_residual_in_train', False))
        self.attention_lambda = float(getattr(configs, 'physical_attention_lambda', 0.0))
        self.residual_eta = float(getattr(configs, 'physical_residual_eta', 0.0))
        self.residual_mode = str(getattr(configs, 'physical_residual_mode', 'fixed'))
        self.residual_eta_init = float(getattr(configs, 'physical_residual_eta_init', self.residual_eta if self.residual_eta != 0.0 else 1.0))
        self.residual_eta_max = float(getattr(configs, 'physical_residual_eta_max', 2.0))
        self.residual_hidden_dim = int(getattr(configs, 'physical_residual_hidden_dim', 16))
        self.free_flow_epsilon = float(getattr(configs, 'physical_free_flow_epsilon', 0.0))

        self.global_eta = None
        self.eta_mlp = None
        self._last_eta = None
        self._eta_history = []
        if self.residual_mode == 'global_learnable':
            self.global_eta = nn.Parameter(torch.tensor(self.residual_eta_init, dtype=torch.float32))
        elif self.residual_mode == 'adaptive':
            self.eta_mlp = nn.Sequential(
                nn.Linear(3, self.residual_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.residual_hidden_dim, 1),
                nn.Sigmoid(),
            )
            self._init_adaptive_eta_head()

    def _init_adaptive_eta_head(self):
        if self.eta_mlp is None:
            return
        init_ratio = min(max(self.residual_eta_init / max(self.residual_eta_max, 1e-6), 1e-4), 1 - 1e-4)
        init_logit = math.log(init_ratio / (1.0 - init_ratio))
        final_linear = self.eta_mlp[2]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, init_logit)

    def reset_eta_diagnostics(self):
        self._last_eta = None
        self._eta_history = []

    def compute_history_slope(self, raw_history):
        if raw_history is None or raw_history.size(1) <= 1:
            return None
        slope = (raw_history[:, -1, :] - raw_history[:, 0, :]) / max(raw_history.size(1) - 1, 1)
        if self.free_flow_epsilon > 0:
            slope = torch.where(slope.abs() < self.free_flow_epsilon, torch.zeros_like(slope), slope)
        return slope

    def compute_history_std(self, raw_history):
        if raw_history is None or raw_history.size(1) <= 1:
            return None
        return raw_history.std(dim=1, unbiased=False)

    def compute_history_level(self, raw_history):
        if raw_history is None or raw_history.size(1) <= 0:
            return None
        return raw_history[:, -1, :]

    def build_attention_bias(self, raw_history, token_count, device, dtype, state_start_idx=1):
        if not (self.enable and self.attention_enable and self.attention_lambda != 0.0):
            return None

        slope = self.compute_history_slope(raw_history)
        if slope is None:
            return None

        seq_tokens = token_count - state_start_idx
        if seq_tokens <= 1:
            return None

        pos = torch.arange(seq_tokens, device=device, dtype=dtype)
        trend_mask = (pos[None, :] - pos[:, None]).clamp(min=0)
        trend_mask = trend_mask / max(seq_tokens - 1, 1)

        bias_state = self.attention_lambda * slope.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1) * trend_mask.view(1, 1, seq_tokens, seq_tokens)
        bias = torch.zeros(raw_history.size(0), raw_history.size(2), token_count, token_count, device=device, dtype=dtype)
        bias[:, :, state_start_idx:, state_start_idx:] = bias_state
        return bias

    def _batch_normalize_feature(self, feature):
        mean = feature.mean(dim=0, keepdim=True)
        std = feature.std(dim=0, keepdim=True, unbiased=False)
        return (feature - mean) / (std + 1e-6)

    def resolve_residual_eta(self, raw_history, pred=None, x_mark_enc=None):
        slope = self.compute_history_slope(raw_history)
        if slope is None:
            return None, None

        if self.residual_mode == 'fixed':
            eta = torch.full_like(slope, self.residual_eta)
        elif self.residual_mode == 'global_learnable':
            eta_value = torch.clamp(self.global_eta, min=0.0, max=self.residual_eta_max)
            eta = torch.ones_like(slope) * eta_value
        elif self.residual_mode == 'adaptive':
            hist_std = self.compute_history_std(raw_history)
            hist_level = self.compute_history_level(raw_history)
            feats = torch.stack([
                self._batch_normalize_feature(slope.abs()),
                self._batch_normalize_feature(hist_std),
                self._batch_normalize_feature(hist_level),
            ], dim=-1)
            eta = self.eta_mlp(feats).squeeze(-1) * self.residual_eta_max
        else:
            eta = torch.full_like(slope, self.residual_eta)

        if pred is not None:
            eta = eta.to(device=pred.device, dtype=pred.dtype)
            slope = slope.to(device=pred.device, dtype=pred.dtype)
        self._last_eta = eta.detach().cpu() if eta is not None else None
        if eta is not None:
            self._eta_history.append(eta.detach().cpu())
        return eta, slope

    def apply_output_residual(self, pred, raw_history, is_training=False, x_mark_enc=None):
        if not self.enable or not self.residual_enable:
            return pred
        if is_training and not self.residual_in_train:
            return pred
        if self.residual_mode == 'fixed' and self.residual_eta == 0.0:
            return pred

        eta, slope = self.resolve_residual_eta(raw_history, pred=pred, x_mark_enc=x_mark_enc)
        if eta is None or slope is None:
            return pred

        horizon = pred.size(1)
        steps = torch.arange(1, horizon + 1, device=pred.device, dtype=pred.dtype).view(1, horizon, 1)
        correction = eta.unsqueeze(1) * slope.unsqueeze(1) * steps
        return pred + correction

    def attention_kwargs(self, raw_history, token_count, device, dtype, state_start_idx=1):
        return {
            'attn_bias': self.build_attention_bias(raw_history, token_count, device, dtype, state_start_idx=state_start_idx)
        }

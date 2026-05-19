import math

import torch
import torch.nn as nn

from Holidiff.frequency.factory import build_frequency_residual


class PhysicalInjectionModule(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.enable = bool(getattr(configs, 'physical_injection_enable', True))
        self.residual_enable = bool(getattr(configs, 'physical_residual_enable', True))
        self.residual_eta = float(getattr(configs, 'physical_residual_eta', 0.5))
        self.free_flow_epsilon = float(getattr(configs, 'physical_free_flow_epsilon', 0.0))
        self.residual_module = build_frequency_residual(configs)

    def reset_eta_diagnostics(self):
        return

    def record_eta_gradients(self):
        return

    def compute_history_slope(self, raw_history):
        if raw_history is None or raw_history.size(1) <= 1:
            return None
        slope = (raw_history[:, -1, :] - raw_history[:, 0, :]) / max(raw_history.size(1) - 1, 1)
        if self.free_flow_epsilon > 0:
            slope = torch.where(slope.abs() < self.free_flow_epsilon, torch.zeros_like(slope), slope)
        return slope

    def apply_output_residual(self, pred, raw_history, is_training=False, x_mark_enc=None):
        if not self.enable or not self.residual_enable:
            return pred
        return self.residual_module.apply_output_residual(
            pred,
            raw_history,
            is_training=is_training,
            x_mark_enc=x_mark_enc,
        )

    def attention_kwargs(self, raw_history, token_count, device, dtype, state_start_idx=1):
        return {'attn_bias': None}

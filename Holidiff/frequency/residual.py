from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_horizon_steps, compute_time_slope


class TimeResidualModule(nn.Module):
    def __init__(self, eta: float = 0.5, free_flow_epsilon: float = 0.0, enc_in: Optional[int] = None):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(float(eta), dtype=torch.float32))
        self.free_flow_epsilon = free_flow_epsilon
        self.enc_in = enc_in

    def apply_output_residual(self, pred, raw_history, is_training=False, x_mark_enc=None):
        slope = compute_time_slope(raw_history, free_flow_epsilon=self.free_flow_epsilon, enc_in=self.enc_in)
        if slope is None:
            return pred
        slope = slope.to(device=pred.device, dtype=pred.dtype)
        eta = self.eta.to(device=pred.device, dtype=pred.dtype).view(1, 1, 1)
        steps = build_horizon_steps(pred.size(1), pred.device, pred.dtype)
        correction = eta * slope.unsqueeze(1) * steps
        return pred + correction


class FrequencyResidualModule(nn.Module):
    def __init__(
        self,
        decomposer: nn.Module,
        eta_init: Optional[List[float]] = None,
        beta_init: Optional[List[float]] = None,
        free_flow_epsilon: float = 0.0,
        enc_in: Optional[int] = None,
    ):
        super().__init__()
        self.decomposer = decomposer
        eta_init = eta_init or [0.2, 0.5, 1.0, 1.2]
        beta_init = beta_init or [0.15, 0.2, 0.3, 0.35]
        self.free_flow_epsilon = free_flow_epsilon
        self.enc_in = enc_in
        eta_tensor = torch.log(torch.expm1(torch.tensor(eta_init, dtype=torch.float32)).clamp_min(1e-6))
        self.eta = nn.Parameter(eta_tensor)
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def _band_slope(self, band_history: torch.Tensor) -> Optional[torch.Tensor]:
        if band_history is None or band_history.size(-1) <= 1:
            return None
        slope = (band_history[:, :, -1] - band_history[:, :, 0]) / max(band_history.size(-1) - 1, 1)
        if self.free_flow_epsilon > 0:
            slope = torch.where(slope.abs() < self.free_flow_epsilon, torch.zeros_like(slope), slope)
        return slope

    def apply_output_residual(self, pred, raw_history, is_training=False, x_mark_enc=None):
        bands = self.decomposer(raw_history, enc_in=self.enc_in)
        if not bands:
            return pred

        steps = build_horizon_steps(pred.size(1), pred.device, pred.dtype)
        eta = F.softplus(self.eta)
        beta = self.beta / self.beta.sum().clamp_min(1e-8)

        correction = torch.zeros_like(pred)
        for idx, band in enumerate(bands):
            if idx >= eta.numel() or idx >= beta.numel():
                break
            slope = self._band_slope(band)
            if slope is None:
                continue
            slope = slope.to(device=pred.device, dtype=pred.dtype)
            correction = correction + beta[idx] * eta[idx] * slope.unsqueeze(1) * steps
        return pred + correction


class HybridResidualModule(nn.Module):
    def __init__(self, time_module: nn.Module, frequency_module: nn.Module, mix_alpha: float = 0.5):
        super().__init__()
        self.time_module = time_module
        self.frequency_module = frequency_module
        self.mix_alpha = mix_alpha

    def apply_output_residual(self, pred, raw_history, is_training=False, x_mark_enc=None):
        time_pred = self.time_module.apply_output_residual(pred, raw_history, is_training=is_training, x_mark_enc=x_mark_enc)
        freq_pred = self.frequency_module.apply_output_residual(pred, raw_history, is_training=is_training, x_mark_enc=x_mark_enc)
        return pred + self.mix_alpha * (time_pred - pred) + (1.0 - self.mix_alpha) * (freq_pred - pred)

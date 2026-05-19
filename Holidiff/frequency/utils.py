from __future__ import annotations

from typing import Optional

import torch


def infer_history_layout(raw_history: torch.Tensor, enc_in: Optional[int] = None) -> str:
    if raw_history.dim() != 3:
        raise ValueError(f'Expected 3D history tensor, got shape {tuple(raw_history.shape)}')
    if enc_in is not None:
        if raw_history.shape[-1] == enc_in:
            return 'BTN'
        if raw_history.shape[1] == enc_in:
            return 'BNT'
    return 'BTN'


def to_bnt(raw_history: torch.Tensor, enc_in: Optional[int] = None) -> torch.Tensor:
    layout = infer_history_layout(raw_history, enc_in=enc_in)
    if layout == 'BTN':
        return raw_history.permute(0, 2, 1)
    return raw_history


def to_btn(x: torch.Tensor, enc_in: Optional[int] = None) -> torch.Tensor:
    layout = infer_history_layout(x, enc_in=enc_in)
    if layout == 'BNT':
        return x.permute(0, 2, 1)
    return x


def build_horizon_steps(horizon: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(1, horizon + 1, device=device, dtype=dtype).view(1, horizon, 1)


def compute_time_slope(raw_history: torch.Tensor, free_flow_epsilon: float = 0.0, enc_in: Optional[int] = None) -> Optional[torch.Tensor]:
    if raw_history is None or raw_history.size(1) <= 1:
        return None
    history_btn = to_btn(raw_history, enc_in=enc_in)
    slope = (history_btn[:, -1, :] - history_btn[:, 0, :]) / max(history_btn.size(1) - 1, 1)
    if free_flow_epsilon > 0:
        slope = torch.where(slope.abs() < free_flow_epsilon, torch.zeros_like(slope), slope)
    return slope

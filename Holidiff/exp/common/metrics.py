"""Common metrics for complex traffic-state evaluation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

EPS = 1e-8


def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, eps: float = EPS) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def compute_high_state_thresholds(train_future: Any, quantile: float = 0.9) -> torch.Tensor:
    train_future = _to_tensor(train_future)
    if train_future.ndim != 3:
        raise ValueError(f'train_future must have shape [B, H, N], got {tuple(train_future.shape)}')
    flat = train_future.reshape(-1, train_future.shape[-1])
    return torch.quantile(flat, q=float(quantile), dim=0)


def compute_complex_state_metrics(
    y_pred: Any,
    y_true: Any,
    histories: Any,
    adj: Any,
    train_future: Any,
    high_state_quantile: float = 0.9,
    peak_radius: int = 1,
    delta_t: float = 1.0,
    valid_mask: Any | None = None,
    eps: float = EPS,
) -> dict[str, float]:
    y_pred = _to_tensor(y_pred)
    y_true = _to_tensor(y_true)
    histories = _to_tensor(histories)
    train_future = _to_tensor(train_future)
    adj = _to_tensor(adj)
    mask = _to_tensor(valid_mask) if valid_mask is not None else None

    if y_pred.shape != y_true.shape:
        raise ValueError(f'y_pred and y_true must match, got {tuple(y_pred.shape)} vs {tuple(y_true.shape)}')
    if histories.ndim != 3:
        raise ValueError(f'histories must have shape [B, L, N], got {tuple(histories.shape)}')
    if train_future.ndim != 3:
        raise ValueError(f'train_future must have shape [B, H, N], got {tuple(train_future.shape)}')

    bsz, horizon, n_nodes = y_true.shape
    if histories.shape[0] != bsz or histories.shape[2] != n_nodes:
        raise ValueError('histories shape is incompatible with y_true')

    theta = compute_high_state_thresholds(train_future, quantile=high_state_quantile)
    high_mask = y_true > theta.view(1, 1, -1)
    if mask is not None:
        high_eval_mask = high_mask & (mask > 0)
    else:
        high_eval_mask = high_mask
    hs_mae = _masked_mean(torch.abs(y_pred - y_true), high_eval_mask, eps=eps)

    h_true = y_true.argmax(dim=1)
    h_pred = y_pred.argmax(dim=1)
    pte = torch.abs(h_true - h_pred).to(y_true.dtype).mean() * float(delta_t)

    h_index = torch.arange(horizon, device=y_true.device).view(1, horizon, 1)
    peak_mask = torch.abs(h_index - h_true.unsqueeze(1)) <= int(peak_radius)
    if mask is not None:
        peak_eval_mask = peak_mask & (mask > 0)
    else:
        peak_eval_mask = peak_mask
    peak_mae = _masked_mean(torch.abs(y_pred - y_true), peak_eval_mask, eps=eps)

    edge_index = (adj > 0).nonzero(as_tuple=False)
    if edge_index.numel() == 0:
        fce = torch.tensor(float('nan'), dtype=y_true.dtype)
    else:
        src = edge_index[:, 0].long().to(y_true.device)
        dst = edge_index[:, 1].long().to(y_true.device)
        pred_diff = y_pred[:, :, dst] - y_pred[:, :, src]
        true_diff = y_true[:, :, dst] - y_true[:, :, src]
        hist_std = (histories[:, :, dst] - histories[:, :, src]).std(dim=1).clamp_min(eps)
        norm_err = torch.abs((pred_diff - true_diff) / (hist_std[:, None, :] + eps))
        if mask is not None:
            edge_mask = (mask[:, :, dst] > 0) & (mask[:, :, src] > 0)
        else:
            edge_mask = None
        fce = _masked_mean(norm_err, edge_mask, eps=eps)

    return {
        'HS-MAE': _to_float(hs_mae),
        'Peak-MAE': _to_float(peak_mae),
        'PTE': _to_float(pte),
        'FCE': _to_float(fce),
        'high_state_quantile': float(high_state_quantile),
        'peak_radius': int(peak_radius),
        'delta_t': float(delta_t),
        'num_edges': int(edge_index.shape[0]) if edge_index.numel() > 0 else 0,
    }


def dumps_pretty(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)

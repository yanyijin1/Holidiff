from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def to_numpy_history(history_context: torch.Tensor | None) -> np.ndarray | None:
    if history_context is None:
        return None
    if isinstance(history_context, torch.Tensor):
        return history_context.detach().float().cpu().numpy()
    return np.asarray(history_context, dtype=np.float32)


def split_history_regimes(hist_values: np.ndarray | None) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if hist_values is None:
        return None, None, None
    hist_values = np.asarray(hist_values, dtype=np.float32).reshape(-1)
    hist_values = hist_values[np.isfinite(hist_values)]
    if hist_values.size == 0:
        return None, None, None
    if hist_values.size == 1:
        return hist_values, hist_values, hist_values

    threshold = float(np.median(hist_values))
    free = hist_values[hist_values <= threshold]
    cong = hist_values[hist_values > threshold]
    if free.size == 0 or cong.size == 0:
        sorted_vals = np.sort(hist_values)
        split = max(1, sorted_vals.size // 2)
        free = sorted_vals[:split]
        cong = sorted_vals[split:]
        if cong.size == 0:
            cong = free.copy()
    return hist_values, free.astype(np.float32), cong.astype(np.float32)

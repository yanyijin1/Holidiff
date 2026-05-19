from __future__ import annotations

import numpy as np
import torch

from .aggregation_utils import split_history_regimes, to_numpy_history


class DensityCentroidAggregator:
    def __init__(self, bandwidth_kde: float = 15.0, bandwidth_hist: float = 20.0, mode: str = 'joint', eps: float = 1e-8):
        self.h_kde = float(bandwidth_kde)
        self.h_hist = float(bandwidth_hist)
        self.mode = str(mode).lower()
        self.eps = float(eps)

    def _kde_weights(self, samples_1d: np.ndarray) -> np.ndarray:
        diff = samples_1d[:, None] - samples_1d[None, :]
        return np.exp(-0.5 * (diff / max(self.h_kde, self.eps)) ** 2).sum(axis=1)

    def _hist_weights(self, samples_1d: np.ndarray, hist_free: np.ndarray | None, hist_cong: np.ndarray | None) -> np.ndarray:
        if hist_free is None and hist_cong is None:
            return np.ones(len(samples_1d), dtype=np.float32)
        like_free = np.zeros(len(samples_1d), dtype=np.float32)
        like_cong = np.zeros(len(samples_1d), dtype=np.float32)
        h = max(self.h_hist, self.eps)
        if hist_free is not None and len(hist_free) > 0:
            diff_free = samples_1d[:, None] - hist_free[None, :]
            like_free = np.exp(-0.5 * (diff_free / h) ** 2).sum(axis=1)
        if hist_cong is not None and len(hist_cong) > 0:
            diff_cong = samples_1d[:, None] - hist_cong[None, :]
            like_cong = np.exp(-0.5 * (diff_cong / h) ** 2).sum(axis=1)
        if self.mode == 'competitive':
            return np.maximum(like_free, like_cong)
        if self.mode == 'hist_only':
            return like_free + like_cong
        return like_free + like_cong

    def aggregate(self, samples_1d: np.ndarray, hist_all: np.ndarray | None = None, hist_free: np.ndarray | None = None, hist_cong: np.ndarray | None = None) -> float:
        samples_1d = np.asarray(samples_1d, dtype=np.float32).reshape(-1)
        if samples_1d.size == 0:
            return 0.0
        w_kde = self._kde_weights(samples_1d)
        w_hist = self._hist_weights(samples_1d, hist_free, hist_cong)
        if self.mode == 'kde_only' or (hist_free is None and hist_cong is None):
            w = w_kde
        elif self.mode == 'hist_only':
            w = w_hist
        else:
            w = w_kde * np.maximum(w_hist, self.eps)
        denom = float(np.sum(w))
        if denom < self.eps:
            return float(samples_1d.mean())
        return float(np.sum(w * samples_1d) / denom)

    def aggregate_batch(self, samples: torch.Tensor, history_context: torch.Tensor | None = None) -> torch.Tensor:
        s, b, h, n = samples.shape
        preds = np.zeros((b, h, n), dtype=np.float32)
        history_np = to_numpy_history(history_context)
        samples_np = samples.detach().float().cpu().numpy()
        for bi in range(b):
            for ni in range(n):
                hist_all, hist_free, hist_cong = split_history_regimes(history_np[bi, :, ni] if history_np is not None else None)
                for hi in range(h):
                    preds[bi, hi, ni] = self.aggregate(samples_np[:, bi, hi, ni], hist_all=hist_all, hist_free=hist_free, hist_cong=hist_cong)
        return torch.from_numpy(preds).to(samples.device)

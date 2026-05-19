from __future__ import annotations

import numpy as np
import torch


class SkewnessCorrectedPeakAggregator:
    def __init__(self, lambda_skew: float = 0.5):
        self.lambda_skew = float(lambda_skew)

    def aggregate(self, samples_1d: np.ndarray) -> float:
        samples_1d = np.asarray(samples_1d, dtype=np.float32).reshape(-1)
        if samples_1d.size == 0:
            return 0.0
        y_bar = float(np.mean(samples_1d))
        sigma = float(np.std(samples_1d))
        if sigma < 1e-6:
            return y_bar
        gamma1 = float(np.mean((samples_1d - y_bar) ** 3) / (sigma ** 3 + 1e-8))
        pred = y_bar - self.lambda_skew * gamma1 * sigma
        return float(pred)

    def aggregate_batch(self, samples: torch.Tensor, history_context: torch.Tensor | None = None) -> torch.Tensor:
        s, b, h, n = samples.shape
        preds = np.zeros((b, h, n), dtype=np.float32)
        samples_np = samples.detach().float().cpu().numpy()
        for bi in range(b):
            for hi in range(h):
                for ni in range(n):
                    preds[bi, hi, ni] = self.aggregate(samples_np[:, bi, hi, ni])
        return torch.from_numpy(preds).to(samples.device)

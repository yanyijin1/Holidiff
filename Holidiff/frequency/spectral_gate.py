from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .utils import to_bnt


class LearnableSpectralGate(nn.Module):
    def __init__(self, seq_len: int, num_bands: int = 3, init_std: float = 0.02):
        super().__init__()
        self.seq_len = seq_len
        self.num_bands = num_bands
        self.freq_len = seq_len // 2 + 1
        self.gate_logits = nn.Parameter(torch.randn(self.freq_len, num_bands) * init_std)

    def forward(self, raw_history: torch.Tensor, enc_in: Optional[int] = None) -> List[torch.Tensor]:
        x = to_bnt(raw_history, enc_in=enc_in)
        seq_len = x.size(-1)
        x_fft = torch.fft.rfft(x, dim=-1)
        if seq_len != self.seq_len:
            raise ValueError(f'LearnableSpectralGate expected seq_len={self.seq_len}, got {seq_len}')
        weights = torch.softmax(self.gate_logits, dim=-1).to(device=x.device, dtype=x_fft.real.dtype)
        bands = []
        for k in range(self.num_bands):
            wk = weights[:, k].view(1, 1, -1)
            band_fft = x_fft * wk
            band = torch.fft.irfft(band_fft, n=self.seq_len, dim=-1)
            bands.append(band)
        return bands

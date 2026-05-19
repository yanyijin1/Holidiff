from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .utils import to_bnt


class IdentityDecomposer(nn.Module):
    def forward(self, raw_history: torch.Tensor, enc_in: Optional[int] = None) -> List[torch.Tensor]:
        return [to_bnt(raw_history, enc_in=enc_in)]


class FixedBandDecomposer(nn.Module):
    def __init__(self, seq_len: int, num_bands: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.num_bands = num_bands
        self.freq_len = seq_len // 2 + 1

    def forward(self, raw_history: torch.Tensor, enc_in: Optional[int] = None) -> List[torch.Tensor]:
        x = to_bnt(raw_history, enc_in=enc_in)
        seq_len = x.size(-1)
        x_fft = torch.fft.rfft(x, dim=-1)
        freq_len = x_fft.size(-1)
        edges = torch.linspace(0, freq_len, self.num_bands + 1, device=x.device)
        bands = []
        for i in range(self.num_bands):
            start = int(edges[i].item())
            end = int(edges[i + 1].item())
            if i == self.num_bands - 1:
                end = freq_len
            mask = torch.zeros(freq_len, device=x.device, dtype=x_fft.real.dtype)
            mask[start:end] = 1.0
            band = torch.fft.irfft(x_fft * mask.view(1, 1, -1), n=seq_len, dim=-1)
            bands.append(band)
        return bands

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .decompose import FixedBandDecomposer
from .utils import to_bnt


class IdentityPatchEmbed(nn.Module):
    def forward(self, patch_tokens: Optional[torch.Tensor] = None, raw_history: Optional[torch.Tensor] = None, enc_in: Optional[int] = None):
        return None


class BandSpecificTrendAwarePatchEmbed(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        d_model: int,
        num_bands: int = 4,
        decomp_type: str = 'fixed_fft',
        fusion_type: str = 'concat_proj',
        enc_in: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_bands = num_bands
        self.fusion_type = fusion_type
        self.enc_in = enc_in
        self.band_dim = max(1, d_model // num_bands)
        self.decomposer = FixedBandDecomposer(seq_len=seq_len, num_bands=num_bands)
        self.band_projs = nn.ModuleList([nn.Linear(patch_len + 1, self.band_dim) for _ in range(num_bands)])
        self.fuse_proj = nn.Linear(self.band_dim * num_bands, d_model)

    def _compute_patch_slope(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        denom = max(self.patch_len - 1, 1)
        slope = (patch_tokens[..., -1] - patch_tokens[..., 0]) / denom
        return slope.unsqueeze(-1)

    def _tokens_from_band(self, band_history: torch.Tensor, target_patch_count: int) -> torch.Tensor:
        tokens = band_history.unfold(dimension=-1, size=self.patch_len, step=1)
        if tokens.size(-2) > target_patch_count:
            tokens = tokens[..., :target_patch_count, :]
        elif tokens.size(-2) < target_patch_count:
            pad_count = target_patch_count - tokens.size(-2)
            pad_tokens = tokens[..., -1:, :].expand(*tokens.shape[:-2], pad_count, tokens.size(-1))
            tokens = torch.cat([tokens, pad_tokens], dim=-2)
        return tokens

    def forward(self, patch_tokens: Optional[torch.Tensor] = None, raw_history: Optional[torch.Tensor] = None, enc_in: Optional[int] = None):
        if raw_history is None:
            return None
        history = to_bnt(raw_history, enc_in=enc_in or self.enc_in)
        bands = self.decomposer(history, enc_in=enc_in or self.enc_in)
        if patch_tokens is not None:
            target_patch_count = patch_tokens.size(-2)
        else:
            target_patch_count = max(1, history.size(-1) - self.patch_len + 1)
        band_embeddings: List[torch.Tensor] = []
        for idx, band in enumerate(bands[: self.num_bands]):
            band_tokens = self._tokens_from_band(band, target_patch_count)
            slope = self._compute_patch_slope(band_tokens)
            features = torch.cat([band_tokens, slope], dim=-1)
            band_embeddings.append(self.band_projs[idx](features))
        if not band_embeddings:
            return None
        if len(band_embeddings) < self.num_bands:
            filler = torch.zeros_like(band_embeddings[0])
            while len(band_embeddings) < self.num_bands:
                band_embeddings.append(filler)
        fused = torch.cat(band_embeddings, dim=-1)
        return self.fuse_proj(fused)

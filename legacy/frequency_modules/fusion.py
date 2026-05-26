from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class IdentityFusion(nn.Module):
    def forward(self, xs: Iterable[torch.Tensor]):
        return xs


class ConcatFusion(nn.Module):
    def forward(self, xs: Iterable[torch.Tensor]):
        xs = list(xs)
        if not xs:
            return None
        return torch.cat(xs, dim=-1)

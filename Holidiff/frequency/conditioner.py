from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class IdentityConditioner(nn.Module):
    def forward(self, raw_history: torch.Tensor, enc_in: Optional[int] = None):
        return None

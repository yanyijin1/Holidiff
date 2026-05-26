from __future__ import annotations

import torch
import torch.nn as nn


class IdentityModulator(nn.Module):
    def forward(self, x):
        return x


class BandTrendModulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

import torch.nn as nn
from einops import rearrange

from Holidiff.micro.stek_backbone import STEKBackbone


class TEK(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = STEKBackbone(configs)
        self.enc_in = configs.enc_in

    def forward(self, micro_realization, timesteps, hist_macro_state, x_mark_enc=None, *configs, **kwargs):
        micro_realization = rearrange(micro_realization, '(b n) h -> b n h', n=self.enc_in)
        hist_macro_state = rearrange(hist_macro_state, '(b n) h -> b n h', n=self.enc_in)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in).unsqueeze(-1).unsqueeze(-1)
        micro_estimate = self.model(micro_realization, timesteps, hist_macro_state, x_mark_enc)
        return micro_estimate

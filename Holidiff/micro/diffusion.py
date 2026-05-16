import torch.nn as nn
from einops import rearrange

from Holidiff.micro.FormerBone import FormerBone


class PatchUVIT(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = FormerBone(configs)
        self.enc_in = configs.enc_in

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):
        x = rearrange(x, '(b n) h -> b n h', n=self.enc_in)
        cond_ts = rearrange(cond_ts, '(b n) h -> b n h', n=self.enc_in)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in).unsqueeze(-1).unsqueeze(-1)
        x = self.model(x, timesteps, cond_ts, x_mark_enc)
        return x

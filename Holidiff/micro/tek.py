import torch
import torch.nn as nn
from einops import rearrange

from Holidiff.frequency.factory import build_frequency_patch_embed
from Holidiff.micro.stek_backbone import STEKBackbone


class TEK(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = STEKBackbone(configs)
        self.enc_in = configs.enc_in
        self.frequency_patch_embed = build_frequency_patch_embed(configs)

    def forward(self, micro_realization, timesteps, hist_macro_state, x_mark_enc=None, *configs, **kwargs):
        raw_history = kwargs.get('raw_history')
        physical_injection = kwargs.get('physical_injection')
        frequency_patch_embedding = None
        micro_realization = rearrange(micro_realization, '(b n) h -> b n h', n=self.enc_in)
        hist_macro_state = rearrange(hist_macro_state, '(b n) h -> b n h', n=self.enc_in)
        if raw_history is not None and self.frequency_patch_embed is not None:
            hist_preview_src = hist_macro_state
            future_preview_src = micro_realization
            if hist_preview_src.shape[-1] < self.model.patch_len:
                hist_preview_src = torch.nn.functional.pad(
                    hist_preview_src,
                    (0, self.model.patch_len - hist_preview_src.shape[-1]),
                    mode='replicate',
                )
            if future_preview_src.shape[-1] < self.model.patch_len:
                future_preview_src = torch.nn.functional.pad(
                    future_preview_src,
                    (0, self.model.patch_len - future_preview_src.shape[-1]),
                    mode='replicate',
                )
            hist_preview = hist_preview_src.unfold(dimension=-1, size=self.model.patch_len, step=self.model.stride)
            future_preview = future_preview_src.unfold(dimension=-1, size=self.model.patch_len, step=self.model.stride)
            state_preview = torch.cat([hist_preview, future_preview], dim=-2)
            frequency_patch_embedding = self.frequency_patch_embed(state_preview, raw_history=raw_history, enc_in=self.enc_in)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in).unsqueeze(-1).unsqueeze(-1)
        micro_estimate = self.model(
            micro_realization,
            timesteps,
            hist_macro_state,
            x_mark_enc,
            raw_history=raw_history,
            physical_injection=physical_injection,
            frequency_patch_embedding=frequency_patch_embedding,
        )
        return micro_estimate

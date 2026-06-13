import torch.nn as nn

from Holidiff.frequency.build import build_htrc_residual


class HybridTrendResidualCorrection(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.enabled = bool(getattr(configs, 'use_htrc', True))
        self.residual_module = build_htrc_residual(configs) if self.enabled else None

    def apply_output_residual(self, pred, raw_history, is_training=False, x_mark_enc=None):
        if not self.enabled or self.residual_module is None:
            return pred
        return self.residual_module.apply_output_residual(
            pred,
            raw_history,
            is_training=is_training,
            x_mark_enc=x_mark_enc,
        )

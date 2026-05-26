import torch.nn as nn

from Holidiff.frequency.factory import build_frequency_residual


class PhysicalInjectionModule(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.enable = bool(getattr(configs, 'physical_injection_enable', True))
        self.residual_enable = bool(getattr(configs, 'physical_residual_enable', True))
        self.residual_module = build_frequency_residual(configs)

    def apply_output_residual(self, pred, raw_history, is_training=False, x_mark_enc=None):
        if not self.enable or not self.residual_enable:
            return pred
        return self.residual_module.apply_output_residual(
            pred,
            raw_history,
            is_training=is_training,
            x_mark_enc=x_mark_enc,
        )

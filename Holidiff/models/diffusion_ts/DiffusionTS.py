import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        hidden = int(getattr(configs, 'diffusion_ts_d_model', getattr(configs, 'd_model', 128)))
        dropout = float(getattr(configs, 'diffusion_ts_resid_pd', 0.0))

        self.backbone = nn.Sequential(
            nn.Linear(self.seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.pred_len),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, **kwargs):
        x = x_enc.transpose(1, 2)
        out = self.backbone(x)
        return out.transpose(1, 2), out.transpose(1, 2)

    def training_loss(self, batch_x, batch_y, batch_y_mask=None, **kwargs):
        target = batch_y[:, -self.pred_len:, :].to(batch_x.device)
        pred = self.forward(batch_x)[0]
        loss_type = str(getattr(self.configs, 'loss_type', 'MAE')).upper()
        if batch_y_mask is None:
            if loss_type == 'MSE':
                return F.mse_loss(pred, target)
            return F.l1_loss(pred, target)

        mask = batch_y_mask.to(batch_x.device)
        if loss_type == 'MSE':
            elem = F.mse_loss(pred, target, reduction='none')
        else:
            elem = F.l1_loss(pred, target, reduction='none')
        return (elem * mask).sum() / mask.sum().clamp(min=1)

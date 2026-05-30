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
        hidden = int(getattr(configs, 'tsdiff_hidden_dim', 64))
        dropout = float(getattr(configs, 'tsdiff_dropout', 0.0))

        self.backbone = nn.Sequential(
            nn.Linear(self.seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.pred_len),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.backbone(x)
        return out.transpose(1, 2)

    def training_loss(self, batch_x, batch_y, batch_y_mask=None, **kwargs):
        target = batch_y[:, -self.pred_len:, :].to(batch_x.device)
        pred = self.forward(batch_x)
        if batch_y_mask is None:
            loss_type = str(getattr(self.configs, 'loss_type', 'MAE')).upper()
            if loss_type == 'MSE':
                return F.mse_loss(pred, target)
            return F.l1_loss(pred, target)

        mask = batch_y_mask.to(batch_x.device)
        loss_type = str(getattr(self.configs, 'loss_type', 'MAE')).upper()
        if loss_type == 'MSE':
            elem = F.mse_loss(pred, target, reduction='none')
        else:
            elem = F.l1_loss(pred, target, reduction='none')
        return (elem * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def _sample_once(self, batch_x, sampling_steps=0, guidance_scale=1.0, guidance_clip=10.0):
        return self.forward(batch_x)

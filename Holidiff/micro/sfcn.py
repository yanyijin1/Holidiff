from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from Holidiff.layers.RevIN import RevIN


class BaseTargetSpaceAdapter(nn.Module):
    def encode_training_future(
        self,
        x_future: torch.Tensor,
        x_history: torch.Tensor,
        use_local_scaling: bool,
        holiday_flag: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def encode_inference_history(
        self,
        x_history: torch.Tensor,
        use_local_scaling: bool,
        holiday_flag: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def decode_prediction(
        self,
        pred: torch.Tensor,
        stats: Dict[str, torch.Tensor],
        use_local_scaling: bool,
        holiday_flag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class VanillaNIAdapter(BaseTargetSpaceAdapter):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def _compute_node_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_mean = torch.mean(x, dim=-1, keepdim=True)
        node_scale = torch.std(x, dim=-1, keepdim=True)
        return node_mean, node_scale

    def encode_training_future(self, x_future: torch.Tensor, x_history: torch.Tensor, use_local_scaling: bool, holiday_flag: torch.Tensor | None = None):
        if use_local_scaling:
            node_count = x_future.shape[1]
            node_mean = torch.mean(x_future[:, -node_count:, :], dim=1, keepdim=True)
            node_scale = torch.ones_like(torch.std(x_future, dim=1, keepdim=True))
        else:
            node_mean, node_scale = self._compute_node_stats(x_history)
        x_norm = (x_future - node_mean) / (node_scale + self.eps)
        return x_norm, {'node_mean': node_mean, 'node_scale': node_scale}

    def encode_inference_history(self, x_history: torch.Tensor, use_local_scaling: bool, holiday_flag: torch.Tensor | None = None):
        if use_local_scaling:
            return x_history, {}
        node_mean, node_scale = self._compute_node_stats(x_history)
        x_norm = (x_history - node_mean) / (node_scale + self.eps)
        return x_norm, {'node_mean': node_mean, 'node_scale': node_scale}

    def decode_prediction(self, pred: torch.Tensor, stats: Dict[str, torch.Tensor], use_local_scaling: bool, holiday_flag: torch.Tensor | None = None):
        if use_local_scaling:
            return pred
        node_mean = stats.get('node_mean')
        node_scale = stats.get('node_scale')
        if node_mean is None or node_scale is None:
            return pred
        return pred * (node_scale + self.eps) + node_mean


class SFCN(VanillaNIAdapter):
    def __init__(
        self,
        adj: torch.Tensor,
        coupling_init: float = 0.05,
        eps: float = 1e-5,
        edge_var_window: int = 720,
        field_stats_source: str = 'future',
        holiday_enable: bool = False,
        holiday_mode: str = 'none',
        holiday_alpha_delta: float = 0.05,
        holiday_hard_alpha: float = 0.10,
        holiday_dropout_prob: float = 0.0,
        holiday_dual_bank_enable: bool = False,
    ):
        super().__init__(eps=eps)
        adj = adj.float()
        self.register_buffer('adj', adj)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer('field_mask', adj / degree)
        self.coupling = nn.Parameter(torch.tensor(float(coupling_init)), requires_grad=False)
        self.edge_var_window = int(edge_var_window)
        self.field_stats_source = field_stats_source
        self.holiday_enable = holiday_enable
        self.holiday_mode = holiday_mode
        self.holiday_dropout_prob = float(holiday_dropout_prob)
        self.holiday_dual_bank_enable = bool(holiday_dual_bank_enable)
        if holiday_mode == 'learned_delta':
            self.delta_alpha = nn.Parameter(torch.tensor(float(holiday_alpha_delta)), requires_grad=True)
        else:
            self.register_buffer('delta_alpha', torch.tensor(float(holiday_alpha_delta)))
        self.register_buffer('holiday_hard_alpha', torch.tensor(float(holiday_hard_alpha)))
        self.register_buffer('diag_last_holiday_ratio', torch.tensor(0.0))
        self.register_buffer('diag_last_holiday_count', torch.tensor(0.0))
        self.register_buffer('diag_last_batch_size', torch.tensor(0.0))
        self.register_buffer('diag_last_injection_delta', torch.tensor(0.0))
        self.register_buffer('diag_last_decode_delta', torch.tensor(0.0))
        self.register_buffer('diag_last_alpha_mean', torch.tensor(0.0))
        self.register_buffer('diag_last_holiday_alpha', torch.tensor(0.0))

    def get_coupling_value(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.coupling).detach()

    def get_delta_alpha_value(self) -> float:
        return float(self.delta_alpha.detach().float().cpu().item())

    def _apply_edge_window(self, x: torch.Tensor) -> torch.Tensor:
        if self.edge_var_window <= 0 or x.size(-1) <= self.edge_var_window:
            return x
        return x[..., -self.edge_var_window:]

    def _normalize_holiday_flag(self, holiday_flag: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if holiday_flag is None:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)
        return (holiday_flag.float().to(device).view(batch_size) > 0.5).float()

    def _apply_holiday_dropout(self, holiday_flag: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.holiday_dropout_prob <= 0:
            return holiday_flag
        keep_mask = (torch.rand_like(holiday_flag) > self.holiday_dropout_prob).float()
        return holiday_flag * keep_mask

    def _select_field_stats_source(self, x_future: torch.Tensor, x_history: torch.Tensor) -> torch.Tensor:
        if self.field_stats_source == 'future':
            return x_future
        return x_history

    def _compute_effective_alpha(self, holiday_flag: torch.Tensor) -> torch.Tensor:
        base_alpha = torch.nn.functional.softplus(self.coupling)
        self.diag_last_batch_size.fill_(float(holiday_flag.numel()))
        self.diag_last_holiday_count.fill_(float(holiday_flag.sum().detach().item()))
        if not self.holiday_enable or self.holiday_mode == 'none':
            alpha = torch.full_like(holiday_flag, float(base_alpha.detach().item()))
            self.diag_last_alpha_mean.fill_(float(alpha.mean().detach().item()))
            self.diag_last_holiday_alpha.fill_(float(base_alpha.detach().item()))
            return alpha.view(-1, 1, 1)
        holiday_flag_train = self._apply_holiday_dropout(holiday_flag)
        if self.holiday_mode == 'hard':
            holiday_alpha = float(self.holiday_hard_alpha.detach().item())
            alpha = torch.where(
                holiday_flag_train > 0.5,
                torch.full_like(holiday_flag_train, holiday_alpha),
                torch.full_like(holiday_flag_train, float(base_alpha.detach().item())),
            )
            self.diag_last_holiday_alpha.fill_(holiday_alpha)
        else:
            delta = torch.nn.functional.softplus(self.delta_alpha) if self.delta_alpha.requires_grad else self.delta_alpha
            alpha = torch.full_like(holiday_flag_train, float(base_alpha.detach().item())) + delta * holiday_flag_train
            self.diag_last_holiday_alpha.fill_(float((float(base_alpha.detach().item()) + float(delta.detach().item()))))
        self.diag_last_alpha_mean.fill_(float(alpha.mean().detach().item()))
        return alpha.view(-1, 1, 1)

    def compute_field_stats(self, x: torch.Tensor):
        x = self._apply_edge_window(x)
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        edge_flow = x_j - x_i
        edge_mean = edge_flow.mean(dim=-1)
        edge_var = edge_flow.std(dim=-1) + self.eps
        return edge_mean, edge_var

    def _compute_dual_bank_field_stats(self, field_stats_input: torch.Tensor, holiday_flag: torch.Tensor):
        edge_mean_all, edge_var_all = self.compute_field_stats(field_stats_input)
        if (not self.holiday_dual_bank_enable) or field_stats_input.size(0) == 0:
            return edge_mean_all, edge_var_all
        holiday_mask = holiday_flag > 0.5
        reg_mask = ~holiday_mask
        edge_mean = edge_mean_all.clone()
        edge_var = edge_var_all.clone()
        if holiday_mask.any() and reg_mask.any():
            hol_mean = edge_mean_all[holiday_mask].mean(dim=0, keepdim=True)
            hol_var = edge_var_all[holiday_mask].mean(dim=0, keepdim=True)
            reg_mean = edge_mean_all[reg_mask].mean(dim=0, keepdim=True)
            reg_var = edge_var_all[reg_mask].mean(dim=0, keepdim=True)
            edge_mean[holiday_mask] = hol_mean.expand(holiday_mask.sum(), -1, -1)
            edge_var[holiday_mask] = hol_var.expand(holiday_mask.sum(), -1, -1)
            edge_mean[reg_mask] = reg_mean.expand(reg_mask.sum(), -1, -1)
            edge_var[reg_mask] = reg_var.expand(reg_mask.sum(), -1, -1)
        return edge_mean, edge_var

    def _compute_field_residual(self, x: torch.Tensor, edge_mean: torch.Tensor, edge_var: torch.Tensor):
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        edge_flow = x_j - x_i
        edge_residual = (edge_flow - edge_mean.unsqueeze(-1)) / edge_var.unsqueeze(-1)
        weighted = self.field_mask.unsqueeze(0).unsqueeze(-1) * edge_residual
        return weighted.sum(dim=2)

    def encode_training_future(self, x_future: torch.Tensor, x_history: torch.Tensor, use_local_scaling: bool, holiday_flag: torch.Tensor | None = None):
        batch_size = x_future.size(0)
        holiday_flag_norm = self._normalize_holiday_flag(holiday_flag, batch_size, x_future.device)
        node_mean, node_scale = self._compute_node_stats(x_future)
        field_stats_input = self._select_field_stats_source(x_future, x_history)
        edge_mean, edge_var = self._compute_dual_bank_field_stats(field_stats_input, holiday_flag_norm)
        node_res = (x_future - node_mean) / (node_scale + self.eps)
        field_res = self._compute_field_residual(x_future, edge_mean, edge_var)
        alpha = self._compute_effective_alpha(holiday_flag_norm)
        base_alpha = torch.nn.functional.softplus(self.coupling)
        base_target = node_res + base_alpha * field_res
        target = node_res + alpha * field_res
        self.diag_last_holiday_ratio.fill_(float(holiday_flag_norm.mean().detach().item()))
        self.diag_last_injection_delta.fill_(float((target - base_target).abs().mean().detach().item()))
        return target, {
            'node_mean': node_mean,
            'node_scale': node_scale,
            'edge_mean': edge_mean,
            'edge_var': edge_var,
            'holiday_flag': holiday_flag_norm,
            'alpha': alpha,
        }

    def encode_inference_history(self, x_history: torch.Tensor, use_local_scaling: bool, holiday_flag: torch.Tensor | None = None):
        batch_size = x_history.size(0)
        holiday_flag_norm = self._normalize_holiday_flag(holiday_flag, batch_size, x_history.device)
        node_mean, node_scale = self._compute_node_stats(x_history)
        edge_mean, edge_var = self._compute_dual_bank_field_stats(x_history, holiday_flag_norm)
        alpha = self._compute_effective_alpha(holiday_flag_norm)
        return x_history, {
            'node_mean': node_mean,
            'node_scale': node_scale,
            'edge_mean': edge_mean,
            'edge_var': edge_var,
            'holiday_flag': holiday_flag_norm,
            'alpha': alpha,
        }

    def decode_prediction(self, pred: torch.Tensor, stats: Dict[str, torch.Tensor], use_local_scaling: bool, holiday_flag: torch.Tensor | None = None):
        node_mean = stats.get('node_mean')
        node_scale = stats.get('node_scale')
        edge_var = stats.get('edge_var')
        alpha = stats.get('alpha')
        if node_mean is None or node_scale is None:
            return pred
        node_pred = pred * (node_scale + self.eps) + node_mean
        if edge_var is None:
            return node_pred
        z_i = pred.unsqueeze(2)
        z_j = pred.unsqueeze(1)
        z_diff = z_j - z_i
        field_pred = (self.field_mask.unsqueeze(0).unsqueeze(-1) * edge_var.unsqueeze(-1) * z_diff).sum(dim=2)
        base_alpha = torch.nn.functional.softplus(self.coupling).view(1, 1, 1)
        base_pred = node_pred + base_alpha * field_pred
        if alpha is None:
            alpha = base_alpha
        out = node_pred + alpha * field_pred
        self.diag_last_decode_delta.fill_(float((out - base_pred).abs().mean().detach().item()))
        return out


class IdentityLocalScaling(nn.Module):
    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LocalAdaptiveScaling(nn.Module):
    def __init__(self, num_features: int, enabled: bool = True, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.enabled = enabled
        self.revin = RevIN(num_features, affine=affine, subtract_last=subtract_last) if enabled else None

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.revin is None:
            return x
        return self.revin(x, 'norm')

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.revin is None:
            return x
        return self.revin(x, 'denorm')


class LocalNorm(LocalAdaptiveScaling):
    def __init__(self, adj: torch.Tensor, num_features: int, enabled: bool = True, affine: bool = True, subtract_last: bool = False, eta_init: float = 0.1):
        super().__init__(num_features=num_features, enabled=enabled, affine=affine, subtract_last=subtract_last)
        adj = adj.float()
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer('field_mask', adj / degree)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)), requires_grad=False)

    def _field_smooth(self, stat: torch.Tensor) -> torch.Tensor:
        stat_bn = stat.squeeze(1)
        neighbor_mean = torch.matmul(stat_bn, self.field_mask.T)
        out = stat_bn + self.eta * (neighbor_mean - stat_bn)
        return out.unsqueeze(1)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.revin is None:
            return x
        self.revin._get_statistics(x)
        if not self.revin.subtract_last:
            self.revin.mean = self._field_smooth(self.revin.mean)
        self.revin.stdev = self._field_smooth(self.revin.stdev).clamp_min(self.revin.eps)
        return self.revin._normalize(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.revin is None:
            return x
        return self.revin._denormalize(x)

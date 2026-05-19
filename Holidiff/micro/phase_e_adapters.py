from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from Holidiff.layers.RevIN import RevIN


class BaseTargetSpaceAdapter(nn.Module):
    def normalize_training_future(
        self,
        x_future: torch.Tensor,
        x_history: torch.Tensor,
        use_revin_norm: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def normalize_inference_history(
        self,
        x_history: torch.Tensor,
        use_revin_norm: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def denormalize_prediction(
        self,
        pred: torch.Tensor,
        stats: Dict[str, torch.Tensor],
        use_revin_norm: bool,
    ) -> torch.Tensor:
        raise NotImplementedError


class VanillaNIAdapter(BaseTargetSpaceAdapter):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def _node_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_ = torch.mean(x, dim=-1, keepdim=True)
        std_ = torch.std(x, dim=-1, keepdim=True)
        return mean_, std_

    def normalize_training_future(self, x_future: torch.Tensor, x_history: torch.Tensor, use_revin_norm: bool):
        if use_revin_norm:
            node_count = x_future.shape[1]
            mean_ = torch.mean(x_future[:, -node_count:, :], dim=1, keepdim=True)
            std_ = torch.ones_like(torch.std(x_future, dim=1, keepdim=True))
        else:
            mean_, std_ = self._node_stats(x_history)
        x_norm = (x_future - mean_) / (std_ + self.eps)
        return x_norm, {'mean': mean_, 'std': std_}

    def normalize_inference_history(self, x_history: torch.Tensor, use_revin_norm: bool):
        if use_revin_norm:
            return x_history, {}
        mean_, std_ = self._node_stats(x_history)
        x_norm = (x_history - mean_) / (std_ + self.eps)
        return x_norm, {'mean': mean_, 'std': std_}

    def denormalize_prediction(self, pred: torch.Tensor, stats: Dict[str, torch.Tensor], use_revin_norm: bool):
        if use_revin_norm:
            return pred
        mean_ = stats.get('mean')
        std_ = stats.get('std')
        if mean_ is None or std_ is None:
            return pred
        return pred * (std_ + self.eps) + mean_


class MatrixNIAdapter(VanillaNIAdapter):
    def __init__(self, adj: torch.Tensor, zeta_init: float = 0.1, learnable_zeta: bool = False, eps: float = 1e-5):
        super().__init__(eps=eps)
        adj = adj.float()
        self.register_buffer('adj', adj)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer('adj_row_norm', adj / degree)
        self.zeta = nn.Parameter(torch.tensor(float(zeta_init)), requires_grad=learnable_zeta)

    def _edge_stats(self, x: torch.Tensor):
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        diff = x_j - x_i
        delta = diff.mean(dim=-1)
        sigma = diff.std(dim=-1) + self.eps
        return delta, sigma

    def _edge_term(self, x: torch.Tensor, delta: torch.Tensor, sigma: torch.Tensor):
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        diff = x_j - x_i
        edge_norm = (diff - delta.unsqueeze(-1)) / sigma.unsqueeze(-1)
        weighted = self.adj_row_norm.unsqueeze(0).unsqueeze(-1) * edge_norm
        return weighted.sum(dim=2)

    def _zeta_value(self):
        if self.zeta.requires_grad:
            return torch.nn.functional.softplus(self.zeta)
        return self.zeta

    def normalize_training_future(self, x_future: torch.Tensor, x_history: torch.Tensor, use_revin_norm: bool):
        mu_node, sigma_node = self._node_stats(x_future)
        delta_edge, sigma_edge = self._edge_stats(x_future)
        node_term = (x_future - mu_node) / (sigma_node + self.eps)
        edge_term = self._edge_term(x_future, delta_edge, sigma_edge)
        target = node_term + self._zeta_value() * edge_term
        return target, {'mean': mu_node, 'std': sigma_node, 'delta_edge': delta_edge, 'sigma_edge': sigma_edge}

    def normalize_inference_history(self, x_history: torch.Tensor, use_revin_norm: bool):
        mu_node, sigma_node = self._node_stats(x_history)
        delta_edge, sigma_edge = self._edge_stats(x_history)
        return x_history, {'mean': mu_node, 'std': sigma_node, 'delta_edge': delta_edge, 'sigma_edge': sigma_edge}

    def denormalize_prediction(self, pred: torch.Tensor, stats: Dict[str, torch.Tensor], use_revin_norm: bool):
        mu_node = stats.get('mean')
        sigma_node = stats.get('std')
        sigma_edge = stats.get('sigma_edge')
        if mu_node is None or sigma_node is None:
            return pred
        node_part = pred * (sigma_node + self.eps) + mu_node
        if sigma_edge is None:
            return node_part
        z_i = pred.unsqueeze(2)
        z_j = pred.unsqueeze(1)
        z_diff = z_j - z_i
        edge_part = (self.adj_row_norm.unsqueeze(0).unsqueeze(-1) * sigma_edge.unsqueeze(-1) * z_diff).sum(dim=2)
        return node_part + self._zeta_value() * edge_part


class IdentityRevINAdapter(nn.Module):
    def forward_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward_denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x


class VanillaRevINAdapter(nn.Module):
    def __init__(self, num_features: int, enabled: bool = True, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.enabled = enabled
        self.revin = RevIN(num_features, affine=affine, subtract_last=subtract_last) if enabled else None

    def forward_norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.revin is None:
            return x
        return self.revin(x, 'norm')

    def forward_denorm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.revin is None:
            return x
        return self.revin(x, 'denorm')


class FieldRevINAdapter(VanillaRevINAdapter):
    def __init__(self, adj: torch.Tensor, num_features: int, enabled: bool = True, affine: bool = True, subtract_last: bool = False, eta_init: float = 0.1):
        super().__init__(num_features=num_features, enabled=enabled, affine=affine, subtract_last=subtract_last)
        adj = adj.float()
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer('adj_row_norm', adj / degree)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)), requires_grad=False)

    def _field_smooth(self, stat: torch.Tensor) -> torch.Tensor:
        stat_bn = stat.squeeze(1)
        neighbor_mean = torch.matmul(stat_bn, self.adj_row_norm.T)
        out = stat_bn + self.eta * (neighbor_mean - stat_bn)
        return out.unsqueeze(1)

    def forward_norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.revin is None:
            return x
        self.revin._get_statistics(x)
        if not self.revin.subtract_last:
            self.revin.mean = self._field_smooth(self.revin.mean)
        self.revin.stdev = self._field_smooth(self.revin.stdev).clamp_min(self.revin.eps)
        return self.revin._normalize(x)

    def forward_denorm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.revin is None:
            return x
        return self.revin._denormalize(x)

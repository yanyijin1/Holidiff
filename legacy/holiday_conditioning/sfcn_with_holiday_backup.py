"""
Spatial Field Coupled Normalization (SFCN)

Paper final version - no holiday conditioning.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from Holidiff.layers.RevIN import RevIN


class BaseTargetSpaceAdapter(nn.Module):
    """Base class for target space adapters."""
    
    def encode_training_future(
        self,
        x_future: torch.Tensor,
        x_history: torch.Tensor,
        use_local_scaling: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def encode_inference_history(
        self,
        x_history: torch.Tensor,
        use_local_scaling: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def decode_prediction(
        self,
        pred: torch.Tensor,
        stats: Dict[str, torch.Tensor],
        use_local_scaling: bool,
    ) -> torch.Tensor:
        raise NotImplementedError


class VanillaNIAdapter(BaseTargetSpaceAdapter):
    """Vanilla node-level instance normalization."""
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def _compute_node_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_mean = torch.mean(x, dim=-1, keepdim=True)
        node_scale = torch.std(x, dim=-1, keepdim=True)
        return node_mean, node_scale

    def encode_training_future(self, x_future: torch.Tensor, x_history: torch.Tensor, use_local_scaling: bool):
        if use_local_scaling:
            node_count = x_future.shape[1]
            node_mean = torch.mean(x_future[:, -node_count:, :], dim=1, keepdim=True)
            node_scale = torch.ones_like(torch.std(x_future, dim=1, keepdim=True))
        else:
            node_mean, node_scale = self._compute_node_stats(x_history)
        x_norm = (x_future - node_mean) / (node_scale + self.eps)
        return x_norm, {'node_mean': node_mean, 'node_scale': node_scale}

    def encode_inference_history(self, x_history: torch.Tensor, use_local_scaling: bool):
        if use_local_scaling:
            return x_history, {}
        node_mean, node_scale = self._compute_node_stats(x_history)
        x_norm = (x_history - node_mean) / (node_scale + self.eps)
        return x_norm, {'node_mean': node_mean, 'node_scale': node_scale}

    def decode_prediction(self, pred: torch.Tensor, stats: Dict[str, torch.Tensor], use_local_scaling: bool):
        if use_local_scaling:
            return pred
        node_mean = stats.get('node_mean')
        node_scale = stats.get('node_scale')
        if node_mean is None or node_scale is None:
            return pred
        return pred * (node_scale + self.eps) + node_mean


class SFCN(VanillaNIAdapter):
    """
    Spatial Field Coupled Normalization (paper final version).
    
    Encodes spatial field structure into the target space by coupling
    node-level residuals with 1-hop neighbor field residuals.
    
    Args:
        adj: Adjacency matrix [N, N]
        coupling_init: Initial coupling strength α (learnable)
        eps: Numerical stability constant
        edge_var_window: Window size for computing edge variance
        field_stats_source: 'future' or 'history' for field statistics
    """
    
    def __init__(
        self,
        adj: torch.Tensor,
        coupling_init: float = 0.05,
        eps: float = 1e-5,
        edge_var_window: int = 720,
        field_stats_source: str = 'future',
    ):
        super().__init__(eps=eps)
        adj = adj.float()
        self.register_buffer('adj', adj)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer('field_mask', adj / degree)
        self.coupling = nn.Parameter(torch.tensor(float(coupling_init)), requires_grad=True)
        self.edge_var_window = int(edge_var_window)
        self.field_stats_source = field_stats_source

    def get_coupling_value(self) -> torch.Tensor:
        """Get current coupling strength α."""
        return torch.nn.functional.softplus(self.coupling).detach()

    def _apply_edge_window(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal window for edge variance computation."""
        if self.edge_var_window <= 0 or x.size(-1) <= self.edge_var_window:
            return x
        return x[..., -self.edge_var_window:]

    def _select_field_stats_source(self, x_future: torch.Tensor, x_history: torch.Tensor) -> torch.Tensor:
        """Select source for field statistics computation."""
        if self.field_stats_source == 'future':
            return x_future
        return x_history

    def compute_field_stats(self, x: torch.Tensor):
        """
        Compute field statistics: edge mean and variance.
        
        Args:
            x: Traffic flow [B, N, T]
        
        Returns:
            edge_mean: Mean flow difference [B, N, N]
            edge_var: Std of flow difference [B, N, N]
        """
        x = self._apply_edge_window(x)
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        edge_flow = x_j - x_i
        edge_mean = edge_flow.mean(dim=-1)
        edge_var = edge_flow.std(dim=-1) + self.eps
        return edge_mean, edge_var

    def _compute_field_residual(self, x: torch.Tensor, edge_mean: torch.Tensor, edge_var: torch.Tensor):
        """
        Compute field-coupled residual.
        
        Args:
            x: Current traffic flow [B, N, T]
            edge_mean: Edge mean [B, N, N]
            edge_var: Edge variance [B, N, N]
        
        Returns:
            field_residual: Weighted field residual [B, N, T]
        """
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        edge_flow = x_j - x_i
        edge_residual = (edge_flow - edge_mean.unsqueeze(-1)) / edge_var.unsqueeze(-1)
        weighted = self.field_mask.unsqueeze(0).unsqueeze(-1) * edge_residual
        return weighted.sum(dim=2)

    def encode_training_future(self, x_future: torch.Tensor, x_history: torch.Tensor, use_local_scaling: bool):
        """
        Encode future traffic flow into SFCN target space (training).
        
        Target = node_residual + α * field_residual
        
        Args:
            x_future: Future traffic flow [B, N, H]
            x_history: Historical traffic flow [B, N, L]
            use_local_scaling: Whether to use local scaling
        
        Returns:
            target: SFCN-encoded target [B, N, H]
            stats: Statistics for decoding
        """
        node_mean, node_scale = self._compute_node_stats(x_future)
        field_stats_input = self._select_field_stats_source(x_future, x_history)
        edge_mean, edge_var = self.compute_field_stats(field_stats_input)
        
        node_res = (x_future - node_mean) / (node_scale + self.eps)
        field_res = self._compute_field_residual(x_future, edge_mean, edge_var)
        
        alpha = torch.nn.functional.softplus(self.coupling)
        target = node_res + alpha * field_res
        
        return target, {
            'node_mean': node_mean,
            'node_scale': node_scale,
            'edge_mean': edge_mean,
            'edge_var': edge_var,
            'alpha': alpha,
        }

    def encode_inference_history(self, x_history: torch.Tensor, use_local_scaling: bool):
        """
        Encode historical traffic flow (inference).
        
        Args:
            x_history: Historical traffic flow [B, N, L]
            use_local_scaling: Whether to use local scaling
        
        Returns:
            x_history: Unchanged (encoding happens in denoiser)
            stats: Statistics for decoding
        """
        node_mean, node_scale = self._compute_node_stats(x_history)
        edge_mean, edge_var = self.compute_field_stats(x_history)
        alpha = torch.nn.functional.softplus(self.coupling)
        
        return x_history, {
            'node_mean': node_mean,
            'node_scale': node_scale,
            'edge_mean': edge_mean,
            'edge_var': edge_var,
            'alpha': alpha,
        }

    def decode_prediction(self, pred: torch.Tensor, stats: Dict[str, torch.Tensor], use_local_scaling: bool):
        """
        Decode SFCN target space back to traffic flow.
        
        Args:
            pred: Prediction in SFCN space [B, N, H]
            stats: Statistics from encoding
            use_local_scaling: Whether to use local scaling
        
        Returns:
            output: Decoded traffic flow [B, N, H]
        """
        node_mean = stats.get('node_mean')
        node_scale = stats.get('node_scale')
        edge_var = stats.get('edge_var')
        alpha = stats.get('alpha')
        
        if node_mean is None or node_scale is None:
            return pred
        
        node_pred = pred * (node_scale + self.eps) + node_mean
        
        if edge_var is None or alpha is None:
            return node_pred
        
        z_i = pred.unsqueeze(2)
        z_j = pred.unsqueeze(1)
        z_diff = z_j - z_i
        field_pred = (self.field_mask.unsqueeze(0).unsqueeze(-1) * edge_var.unsqueeze(-1) * z_diff).sum(dim=2)
        
        output = node_pred + alpha * field_pred
        return output


class IdentityLocalScaling(nn.Module):
    """Identity scaling (no-op)."""
    
    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LocalAdaptiveScaling(nn.Module):
    """Local adaptive scaling using RevIN."""
    
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
    """Local normalization with field smoothing."""
    
    def __init__(self, adj: torch.Tensor, num_features: int, enabled: bool = True, affine: bool = True, subtract_last: bool = False, eta_init: float = 0.1):
        super().__init__(num_features=num_features, enabled=enabled, affine=affine, subtract_last=subtract_last)
        adj = adj.float()
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer('field_mask', adj / degree)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)), requires_grad=False)

    def _field_smooth(self, stat: torch.Tensor) -> torch.Tensor:
        """Smooth statistics using spatial field."""
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

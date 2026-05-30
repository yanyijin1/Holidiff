from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn

from Holidiff.layers.RevIN import RevIN


class FieldStatsProvider:
    def __init__(self, root_path: str, enc_in: int, adj_file: str):
        self.root_path = Path(root_path)
        self.enc_in = enc_in
        self.adj_file = adj_file

    def load_graph(self) -> torch.Tensor:
        adj_path = self.root_path / self.adj_file
        if not adj_path.exists():
            return torch.eye(self.enc_in, dtype=torch.float32)
        adj_df = pd.read_csv(adj_path)
        if {'src_FID', 'nbr_FID'}.issubset(adj_df.columns):
            nodes = sorted(set(adj_df['src_FID'].astype(int)).union(set(adj_df['nbr_FID'].astype(int))))
            node_to_idx = {nid: i for i, nid in enumerate(nodes)}
            adj = torch.zeros((len(nodes), len(nodes)), dtype=torch.float32)
            for _, row in adj_df.iterrows():
                adj[node_to_idx[int(row['src_FID'])], node_to_idx[int(row['nbr_FID'])]] = 1.0
            return adj
        return torch.tensor(adj_df.values, dtype=torch.float32)

    def compute_node_stats(self, train_series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return train_series.mean(dim=-1, keepdim=True), train_series.std(dim=-1, keepdim=True)

    def compute_field_stats(self, train_series: torch.Tensor, graph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_i = train_series.unsqueeze(2)
        x_j = train_series.unsqueeze(1)
        edge_flow = x_j - x_i
        edge_mean = edge_flow.mean(dim=-1)
        edge_var = edge_flow.std(dim=-1) + 1e-5
        return edge_mean * graph.unsqueeze(0), edge_var * graph.unsqueeze(0)


class BaseTargetSpaceAdapter(nn.Module):
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
        self.coupling = nn.Parameter(torch.tensor(float(coupling_init)), requires_grad=False)
        self.edge_var_window = int(edge_var_window)
        self.field_stats_source = field_stats_source

    def get_coupling_value(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.coupling).detach()

    def _apply_edge_window(self, x: torch.Tensor) -> torch.Tensor:
        if self.edge_var_window <= 0 or x.size(-1) <= self.edge_var_window:
            return x
        return x[..., -self.edge_var_window:]

    def _select_field_stats_source(self, x_future: torch.Tensor, x_history: torch.Tensor) -> torch.Tensor:
        if self.field_stats_source == 'future':
            return x_future
        return x_history

    def compute_field_stats(self, x: torch.Tensor):
        x = self._apply_edge_window(x)
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        edge_flow = x_j - x_i
        edge_mean = edge_flow.mean(dim=-1)
        edge_var = edge_flow.std(dim=-1) + self.eps
        return edge_mean, edge_var

    def _compute_field_residual(self, x: torch.Tensor, edge_mean: torch.Tensor, edge_var: torch.Tensor):
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        edge_flow = x_j - x_i
        edge_residual = (edge_flow - edge_mean.unsqueeze(-1)) / edge_var.unsqueeze(-1)
        weighted = self.field_mask.unsqueeze(0).unsqueeze(-1) * edge_residual
        return weighted.sum(dim=2)

    def encode_training_future(self, x_future: torch.Tensor, x_history: torch.Tensor, use_local_scaling: bool):
        node_mean, node_scale = self._compute_node_stats(x_future)
        field_stats_input = self._select_field_stats_source(x_future, x_history)
        edge_mean, edge_var = self.compute_field_stats(field_stats_input)
        node_res = (x_future - node_mean) / (node_scale + self.eps)
        field_res = self._compute_field_residual(x_future, edge_mean, edge_var)
        alpha = torch.nn.functional.softplus(self.coupling).view(1, 1, 1)
        target = node_res + alpha * field_res
        return target, {
            'node_mean': node_mean,
            'node_scale': node_scale,
            'edge_mean': edge_mean,
            'edge_var': edge_var,
            'alpha': alpha,
        }

    def encode_inference_history(self, x_history: torch.Tensor, use_local_scaling: bool):
        node_mean, node_scale = self._compute_node_stats(x_history)
        edge_mean, edge_var = self.compute_field_stats(x_history)
        alpha = torch.nn.functional.softplus(self.coupling).view(1, 1, 1)
        return x_history, {
            'node_mean': node_mean,
            'node_scale': node_scale,
            'edge_mean': edge_mean,
            'edge_var': edge_var,
            'alpha': alpha,
        }

    def decode_prediction(self, pred: torch.Tensor, stats: Dict[str, torch.Tensor], use_local_scaling: bool):
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
        if alpha is None:
            alpha = torch.nn.functional.softplus(self.coupling).view(1, 1, 1)
        return node_pred + alpha * field_pred


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


def _build_field_stats_provider(configs) -> FieldStatsProvider:
    return FieldStatsProvider(
        root_path=getattr(configs, 'root_path', ''),
        enc_in=int(getattr(configs, 'enc_in', 30)),
        adj_file=getattr(configs, 'spatial_graph_file', 'adjacent_gantry.csv'),
    )


def build_target_adapter(configs):
    spatial_field_enable = bool(getattr(configs, 'spatial_field_enable', False))
    if spatial_field_enable:
        provider = _build_field_stats_provider(configs)
        return SFCN(
            adj=provider.load_graph(),
            coupling_init=float(getattr(configs, 'spatial_field_coupling_init', 0.05)),
            edge_var_window=int(getattr(configs, 'spatial_field_variance_window', 720)),
            field_stats_source=getattr(configs, 'spatial_field_stats_source', 'future'),
        )
    return VanillaNIAdapter()


def build_revin_adapter(configs):
    enabled = bool(getattr(configs, 'local_scaling_enable', 0))
    num_features = int(getattr(configs, 'enc_in', 1))
    if not enabled:
        return IdentityLocalScaling()
    return LocalAdaptiveScaling(num_features=num_features, enabled=True, affine=True, subtract_last=False)

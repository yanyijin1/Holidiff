from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch


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

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch


class GraphStatsProvider:
    def __init__(self, root_path: str, enc_in: int):
        self.root_path = Path(root_path)
        self.enc_in = enc_in

    def load_graph(self) -> torch.Tensor:
        adj_path = self.root_path / 'adjacent_gantry.csv'
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

    def compute_edge_stats(self, train_series: torch.Tensor, graph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_i = train_series.unsqueeze(2)
        x_j = train_series.unsqueeze(1)
        diff = x_j - x_i
        delta = diff.mean(dim=-1)
        sigma = diff.std(dim=-1) + 1e-5
        return delta * graph.unsqueeze(0), sigma * graph.unsqueeze(0)

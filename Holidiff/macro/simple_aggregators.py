"""
Simple aggregators for ablation experiments.

These aggregators are NOT part of the main HoliDiff model,
but are used for comparison in ablation studies.
"""

import torch
import torch.nn as nn


class MeanAggregator:
    """Simple mean aggregation baseline."""
    
    def aggregate_batch(self, all_outs: torch.Tensor, history_context: torch.Tensor = None) -> torch.Tensor:
        return all_outs.mean(0)


class MedianAggregator:
    """Simple median aggregation baseline."""
    
    def aggregate_batch(self, all_outs: torch.Tensor, history_context: torch.Tensor = None) -> torch.Tensor:
        return torch.median(all_outs, dim=0)[0]


class MoMAggregator:
    """
    Median-of-Means (MoM) aggregator.
    
    Used as baseline in ablation experiments.
    NOT part of the main HoliDiff model.
    """
    
    def __init__(self, n_blocks: int = 5, rmom_n: int = 3):
        self.n_blocks = n_blocks
        self.rmom_n = rmom_n
        self._reset_diagnostics()
    
    def _reset_diagnostics(self):
        self._diag_block_var = []
        self._diag_inter_dev = []
        self._diag_median_bias = []
        self._diag_block_var_map = []
        self._diag_inter_dev_map = []
        self._diag_mean_pred = []
        self._diag_median_pred = []
    
    def _consensus_mean(self, seq):
        return torch.sum(seq, dim=0) / seq.size(0)
    
    def _consensus_reduce(self, tensor):
        n_blocks = self.n_blocks
        if n_blocks > tensor.size(0):
            n_blocks = int(torch.ceil(tensor.size(0) / 2))
        
        indic = torch.randperm(tensor.size(0))
        tensor = tensor[indic]
        block_size = tensor.size(0) // n_blocks
        
        means = []
        block_var_maps = []
        block_var_scalars = []
        for i in range(n_blocks):
            start_index = i * block_size
            end_index = start_index + block_size if (i + 1) < n_blocks else tensor.size(0)
            block = tensor[start_index:end_index]
            block_mean = self._consensus_mean(block)
            means.append(block_mean)
            block_var = torch.var(block, dim=0, unbiased=False)
            block_var_maps.append(block_var)
            block_var_scalars.append(block_var.mean(dim=[1, 2]))
        
        means = torch.stack(means)
        means_mean = means.mean(dim=0)
        inter_dev_map = torch.mean((means - means_mean.unsqueeze(0)) ** 2, dim=0)
        median_pred = torch.median(means, dim=0)[0]
        median_bias = torch.abs(median_pred - means_mean).mean(dim=[1, 2])
        
        block_var_map_mean = torch.stack(block_var_maps, dim=0).mean(dim=0)
        block_var_scalar_mean = torch.stack(block_var_scalars, dim=0).mean(dim=0)
        inter_dev_scalar = inter_dev_map.mean(dim=[1, 2])
        
        self._diag_block_var.append(block_var_scalar_mean.detach().cpu())
        self._diag_inter_dev.append(inter_dev_scalar.detach().cpu())
        self._diag_median_bias.append(median_bias.detach().cpu())
        self._diag_block_var_map.append(block_var_map_mean.detach().cpu())
        self._diag_inter_dev_map.append(inter_dev_map.detach().cpu())
        self._diag_mean_pred.append(means_mean.detach().cpu())
        self._diag_median_pred.append(median_pred.detach().cpu())
        
        return median_pred
    
    def aggregate_batch(self, outputs: torch.Tensor, history_context: torch.Tensor = None) -> torch.Tensor:
        results = []
        start_idx = len(self._diag_block_var)
        for _ in range(self.rmom_n):
            shuffled_outputs = outputs[torch.randperm(outputs.size(0))]
            result = self._consensus_reduce(shuffled_outputs)
            results.append(result)
        results = torch.stack(results)
        
        self._diag_block_var = self._diag_block_var[:start_idx]
        self._diag_inter_dev = self._diag_inter_dev[:start_idx]
        self._diag_median_bias = self._diag_median_bias[:start_idx]
        self._diag_block_var_map = self._diag_block_var_map[:start_idx]
        self._diag_inter_dev_map = self._diag_inter_dev_map[:start_idx]
        self._diag_mean_pred = self._diag_mean_pred[:start_idx]
        self._diag_median_pred = self._diag_median_pred[:start_idx]
        
        return torch.median(results, dim=0)[0]

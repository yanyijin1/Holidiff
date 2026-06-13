from __future__ import annotations

from .dca_aggregator import DensityCentroidAggregator


def build_macro_aggregators(configs):
    common_kwargs = {
        'bandwidth_kde': float(getattr(configs, 'dca_bandwidth_kde', 15.0)),
        'bandwidth_hist': float(getattr(configs, 'dca_bandwidth_hist', 20.0)),
        'eps': float(getattr(configs, 'aggregation_eps', 1e-8)),
    }
    return {
        'kde_mode': DensityCentroidAggregator(mode='kde_only', **common_kwargs),
        'dca': DensityCentroidAggregator(
            mode=str(getattr(configs, 'dca_mode', 'joint')).lower(),
            **common_kwargs,
        ),
    }

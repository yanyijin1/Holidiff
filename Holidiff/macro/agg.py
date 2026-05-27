from __future__ import annotations

from .dca_aggregator import DensityCentroidAggregator


def build_macro_aggregator(configs):
    mode = str(getattr(configs, 'aggregation_mode', 'simple')).lower()
    if mode == 'dca':
        return DensityCentroidAggregator(
            bandwidth_kde=float(getattr(configs, 'dca_bandwidth_kde', 15.0)),
            bandwidth_hist=float(getattr(configs, 'dca_bandwidth_hist', 20.0)),
            mode=str(getattr(configs, 'dca_mode', 'joint')).lower(),
            eps=float(getattr(configs, 'aggregation_eps', 1e-8)),
        )
    return None

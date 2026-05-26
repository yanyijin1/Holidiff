from __future__ import annotations

from .dca_aggregator import DensityCentroidAggregator
from .scp_aggregator import SkewnessCorrectedPeakAggregator


def build_macro_aggregator(configs):
    mode = str(getattr(configs, 'aggregation_mode', 'simple')).lower()
    if mode == 'dca':
        return DensityCentroidAggregator(
            bandwidth_kde=float(getattr(configs, 'dca_bandwidth_kde', getattr(configs, 'density_bandwidth', 15.0))),
            bandwidth_hist=float(getattr(configs, 'dca_bandwidth_hist', 20.0)),
            mode=str(getattr(configs, 'dca_mode', 'joint')).lower(),
            eps=float(getattr(configs, 'aggregation_eps', 1e-8)),
        )
    if mode == 'scp':
        return SkewnessCorrectedPeakAggregator(
            lambda_skew=float(getattr(configs, 'scp_lambda_skew', 0.5)),
        )
    return None

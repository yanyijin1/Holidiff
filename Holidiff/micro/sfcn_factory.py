from __future__ import annotations

from Holidiff.micro.field_stats import FieldStatsProvider
from Holidiff.micro.sfcn import (
    IdentityLocalScaling,
    LocalAdaptiveScaling,
    SFCN,
    VanillaNIAdapter,
)


def _build_field_stats_provider(configs) -> FieldStatsProvider:
    return FieldStatsProvider(
        root_path=getattr(configs, 'root_path', ''),
        enc_in=int(getattr(configs, 'enc_in', 30)),
        adj_file=getattr(configs, 'phase_e_adj_file', 'adjacent_gantry.csv'),
    )


def build_target_adapter(configs):
    phase_e_enable = bool(getattr(configs, 'phase_e_enable', False))
    if phase_e_enable:
        provider = _build_field_stats_provider(configs)
        return SFCN(
            adj=provider.load_graph(),
            coupling_init=float(getattr(configs, 'phase_e_zeta_init', 0.05)),
            edge_var_window=int(getattr(configs, 'phase_e_edge_var_window', 720)),
            field_stats_source=getattr(configs, 'phase_e_field_stats_source', 'future'),
        )
    return VanillaNIAdapter()


def build_revin_adapter(configs):
    enabled = bool(getattr(configs, 'new_norm', 0))
    num_features = int(getattr(configs, 'enc_in', 1))
    if not enabled:
        return IdentityLocalScaling()
    return LocalAdaptiveScaling(num_features=num_features, enabled=True, affine=True, subtract_last=False)

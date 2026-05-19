from __future__ import annotations

from Holidiff.micro.field_stats import FieldStatsProvider
from Holidiff.micro.sfcn import (
    IdentityLocalScaling,
    LocalAdaptiveScaling,
    LocalNorm,
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
    adapter_name = getattr(configs, 'phase_e_target_adapter', 'vanilla_ni')
    if phase_e_enable and adapter_name == 'matrix_ni':
        provider = _build_field_stats_provider(configs)
        return SFCN(
            adj=provider.load_graph(),
            coupling_init=float(getattr(configs, 'phase_e_zeta_init', 0.05)),
            edge_var_window=int(getattr(configs, 'phase_e_edge_var_window', 720)),
            field_stats_source=getattr(configs, 'phase_e_field_stats_source', 'future'),
            holiday_enable=bool(getattr(configs, 'holiday_enable', False)),
            holiday_mode=getattr(configs, 'holiday_mode', 'none'),
            holiday_alpha_delta=float(getattr(configs, 'holiday_alpha_delta', 0.05)),
            holiday_hard_alpha=float(getattr(configs, 'holiday_hard_alpha', 0.10)),
            holiday_dropout_prob=float(getattr(configs, 'holiday_dropout_prob', 0.0)),
            holiday_dual_bank_enable=bool(getattr(configs, 'holiday_dual_bank_enable', False)),
        )
    return VanillaNIAdapter()


def build_revin_adapter(configs):
    enabled = bool(getattr(configs, 'new_norm', 0))
    adapter_name = getattr(configs, 'phase_e_revin_adapter', 'vanilla_revin')
    num_features = int(getattr(configs, 'enc_in', 1))
    if not enabled:
        return IdentityLocalScaling()
    if adapter_name == 'field_revin' and bool(getattr(configs, 'phase_e_enable', False)):
        provider = _build_field_stats_provider(configs)
        return LocalNorm(
            adj=provider.load_graph(),
            num_features=num_features,
            enabled=True,
            affine=True,
            subtract_last=False,
            eta_init=float(getattr(configs, 'phase_e_field_eta', 0.1)),
        )
    return LocalAdaptiveScaling(num_features=num_features, enabled=True, affine=True, subtract_last=False)

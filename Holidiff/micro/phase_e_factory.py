from __future__ import annotations

from Holidiff.micro.phase_e_adapters import (
    FieldRevINAdapter,
    IdentityRevINAdapter,
    MatrixNIAdapter,
    VanillaNIAdapter,
    VanillaRevINAdapter,
)
from Holidiff.micro.phase_e_graph_stats import GraphStatsProvider


def _build_graph_provider(configs) -> GraphStatsProvider:
    return GraphStatsProvider(
        root_path=getattr(configs, 'root_path', ''),
        enc_in=int(getattr(configs, 'enc_in', 30)),
    )


def build_target_adapter(configs):
    phase_e_enable = bool(getattr(configs, 'phase_e_enable', False))
    adapter_name = getattr(configs, 'phase_e_target_adapter', 'vanilla_ni')
    if phase_e_enable and adapter_name == 'matrix_ni':
        provider = _build_graph_provider(configs)
        return MatrixNIAdapter(
            adj=provider.load_graph(),
            zeta_init=float(getattr(configs, 'phase_e_zeta_init', 0.1)),
            learnable_zeta=True,
        )
    return VanillaNIAdapter()


def build_revin_adapter(configs):
    enabled = bool(getattr(configs, 'new_norm', 0))
    adapter_name = getattr(configs, 'phase_e_revin_adapter', 'vanilla_revin')
    num_features = int(getattr(configs, 'enc_in', 1))
    if not enabled:
        return IdentityRevINAdapter()
    if adapter_name == 'field_revin' and bool(getattr(configs, 'phase_e_enable', False)):
        provider = _build_graph_provider(configs)
        return FieldRevINAdapter(
            adj=provider.load_graph(),
            num_features=num_features,
            enabled=True,
            affine=True,
            subtract_last=False,
            eta_init=float(getattr(configs, 'phase_e_field_eta', 0.1)),
        )
    return VanillaRevINAdapter(num_features=num_features, enabled=True, affine=True, subtract_last=False)

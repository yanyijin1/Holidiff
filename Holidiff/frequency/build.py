from __future__ import annotations

from .residual import (
    FrequencyResidualModule,
    HybridResidualModule,
    TimeResidualModule,
)
from .spec import BandSpecificTrendAwarePatchEmbed, FixedBandDecomposer


def build_frequency_decomposer(configs):
    seq_len = int(getattr(configs, 'seq_len', 0))
    num_bands = int(getattr(configs, 'frequency_num_bands', 4))
    return FixedBandDecomposer(seq_len=seq_len, num_bands=num_bands)


def build_frequency_residual(configs):
    frequency_enable = bool(getattr(configs, 'frequency_enable', False))
    enc_in = getattr(configs, 'enc_in', None)
    free_flow_epsilon = float(getattr(configs, 'physical_free_flow_epsilon', 0.0))
    time_module = TimeResidualModule(
        eta=float(getattr(configs, 'physical_residual_eta', 0.5)),
        free_flow_epsilon=free_flow_epsilon,
        enc_in=enc_in,
    )
    if not frequency_enable:
        return time_module
    decomposer = build_frequency_decomposer(configs)
    freq_module = FrequencyResidualModule(
        decomposer=decomposer,
        eta_init=list(getattr(configs, 'frequency_eta_init', [0.2, 0.5, 1.0])),
        beta_init=list(getattr(configs, 'frequency_beta_init', [0.2, 0.3, 0.5])),
        use_learnable_beta=bool(getattr(configs, 'frequency_use_learnable_beta', False)),
        use_softplus_eta=bool(getattr(configs, 'frequency_use_softplus_eta', True)),
        free_flow_epsilon=free_flow_epsilon,
        enc_in=enc_in,
    )
    return HybridResidualModule(
        time_module=time_module,
        frequency_module=freq_module,
        mix_alpha=float(getattr(configs, 'frequency_hybrid_mix_alpha', 0.5)),
    )


def build_frequency_patch_embed(configs):
    return BandSpecificTrendAwarePatchEmbed(
        seq_len=int(getattr(configs, 'seq_len', 0)),
        patch_len=int(getattr(configs, 'patch_len', 16)),
        d_model=int(getattr(configs, 'd_model', 128)),
        num_bands=int(getattr(configs, 'frequency_patch_embed_num_bands', getattr(configs, 'frequency_num_bands', 3))),
        fusion_type=getattr(configs, 'frequency_patch_embed_fusion', 'concat_proj'),
        enc_in=getattr(configs, 'enc_in', None),
    )

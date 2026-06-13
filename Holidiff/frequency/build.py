from __future__ import annotations

from .residual import (
    FrequencyResidualModule,
    HybridResidualModule,
    TimeResidualModule,
)
from .spec import BandSpecificTrendAwarePatchEmbed, FixedBandDecomposer


def build_band_decomposer(configs):
    seq_len = int(getattr(configs, 'seq_len', 0))
    num_bands = int(getattr(configs, 'num_bands', 4))
    return FixedBandDecomposer(seq_len=seq_len, num_bands=num_bands)


def build_htrc_residual(configs):
    enc_in = getattr(configs, 'enc_in', None)
    free_flow_epsilon = float(getattr(configs, 'htrc_free_flow_epsilon', 0.0))
    time_module = TimeResidualModule(
        eta=float(getattr(configs, 'htrc_time_eta', 0.5)),
        free_flow_epsilon=free_flow_epsilon,
        enc_in=enc_in,
    )
    decomposer = build_band_decomposer(configs)
    freq_module = FrequencyResidualModule(
        decomposer=decomposer,
        eta_init=list(getattr(configs, 'htrc_band_eta_init', [0.2, 0.5, 1.0, 1.2])),
        beta_init=list(getattr(configs, 'htrc_band_beta_init', [0.15, 0.2, 0.3, 0.35])),
        free_flow_epsilon=free_flow_epsilon,
        enc_in=enc_in,
    )
    return HybridResidualModule(
        time_module=time_module,
        frequency_module=freq_module,
        mix_alpha=float(getattr(configs, 'htrc_mix_alpha', 0.5)),
    )


def build_lstde_patch_embed(configs):
    return BandSpecificTrendAwarePatchEmbed(
        seq_len=int(getattr(configs, 'seq_len', 0)),
        patch_len=int(getattr(configs, 'patch_len', 16)),
        d_model=int(getattr(configs, 'd_model', 128)),
        num_bands=int(getattr(configs, 'num_bands', 4)),
        fusion_type=getattr(configs, 'lstde_patch_fusion', 'concat_proj'),
        enc_in=getattr(configs, 'enc_in', None),
    )

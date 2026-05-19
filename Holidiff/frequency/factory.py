from __future__ import annotations

from .conditioner import IdentityConditioner
from .decompose import FixedBandDecomposer, IdentityDecomposer
from .patch_embed import BandSpecificTrendAwarePatchEmbed, IdentityPatchEmbed
from .residual import (
    FrequencyResidualModule,
    HybridResidualModule,
    IdentityResidualModule,
    TimeResidualModule,
)


def build_frequency_decomposer(configs):
    mode = getattr(configs, 'frequency_decomp_type', 'fixed_fft')
    seq_len = int(getattr(configs, 'seq_len', 0))
    num_bands = int(getattr(configs, 'frequency_num_bands', 4))
    if mode == 'fixed_fft':
        return FixedBandDecomposer(seq_len=seq_len, num_bands=num_bands)
    return IdentityDecomposer()


def build_frequency_residual(configs):
    frequency_enable = bool(getattr(configs, 'frequency_enable', False))
    residual_mode = getattr(configs, 'frequency_residual_type', 'time_residual')
    enc_in = getattr(configs, 'enc_in', None)
    free_flow_epsilon = float(getattr(configs, 'physical_free_flow_epsilon', 0.0))
    time_module = TimeResidualModule(
        eta=float(getattr(configs, 'physical_residual_eta', 0.5)),
        free_flow_epsilon=free_flow_epsilon,
        enc_in=enc_in,
    )
    if not frequency_enable:
        return time_module
    if residual_mode == 'none':
        return IdentityResidualModule()
    if residual_mode == 'time_residual':
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
    if residual_mode == 'frequency_residual':
        return freq_module
    if residual_mode == 'hybrid_residual':
        return HybridResidualModule(
            time_module=time_module,
            frequency_module=freq_module,
            mix_alpha=float(getattr(configs, 'frequency_hybrid_mix_alpha', 0.5)),
        )
    return time_module


def build_frequency_conditioner(configs):
    return IdentityConditioner()


def build_frequency_patch_embed(configs):
    frequency_enable = bool(getattr(configs, 'frequency_enable', False))
    mode = getattr(configs, 'frequency_injection_mode', 'none')
    patch_mode = getattr(configs, 'frequency_patch_embed_mode', 'identity')
    if not frequency_enable or mode != 'embed_replace' or patch_mode != 'band_trend':
        return IdentityPatchEmbed()
    return BandSpecificTrendAwarePatchEmbed(
        seq_len=int(getattr(configs, 'seq_len', 0)),
        patch_len=int(getattr(configs, 'patch_len', 16)),
        d_model=int(getattr(configs, 'd_model', 128)),
        num_bands=int(getattr(configs, 'frequency_patch_embed_num_bands', getattr(configs, 'frequency_num_bands', 3))),
        decomp_type=getattr(configs, 'frequency_decomp_type', 'fixed_fft'),
        fusion_type=getattr(configs, 'frequency_patch_embed_fusion', 'concat_proj'),
        enc_in=getattr(configs, 'enc_in', None),
    )

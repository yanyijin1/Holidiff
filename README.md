# HoliDiff

HoliDiff is a traffic forecasting project built around a diffusion-based micro-to-macro prediction pipeline for holiday traffic scenarios.

The current repository has been cleaned and consolidated around the paper-facing pipeline:

- one active task surface: `long_term_forecast`
- one active experiment class: `Exp_Long_Term_Forecast`
- one main model entry: `HATEK`
- one main aggregation reference: `DCA`
- retained ablations:
  - `simple / median / mom / dca` aggregation
  - frequency on/off ablation
  - non-graph RevIN ablation

---

## 1. Repository status

This repository is now aligned to the current paper-oriented structure.

What has already been cleaned:

- removed unused experiment task files for classification / imputation / anomaly detection / short-term forecasting
- removed obsolete holiday conditioning path from the model call chain
- merged and shortened module structure in `micro/`, `frequency/`, and `macro/`
- removed temporary debug logs and transient checkpoints
- promoted DCA to the main reference configuration

---

## 2. Method overview

The current HoliDiff pipeline can be read as:

1. **History input**
   - use historical traffic flow as conditional context

2. **Micro realization generation**
   - sample future micro realizations with diffusion

3. **Traffic evolution kernel estimation**
   - use `TEK` and its spatiotemporal backbone to estimate future states

4. **Residual correction**
   - optionally inject time/frequency residual structure

5. **Macro aggregation**
   - aggregate multiple sampled futures into final macro prediction
   - current supported modes:
     - `simple`
     - `median`
     - `mom`
     - `dca`

`dca` is the current main paper-line aggregation reference.

---

## 3. Current project structure

```text
STdiff/
├── Holidiff/
│   ├── HoliDiff.py
│   ├── train.py
│   ├── configs/
│   │   ├── main_fujian30_dca.yaml
│   │   └── quickrun_fujian30_1epoch.yaml
│   ├── data_provider/
│   │   ├── data_factory.py
│   │   ├── data_loader.py
│   │   └── fujian30_loader.py
│   ├── exp/
│   │   ├── exp_basic.py
│   │   └── exp_long_term_forecasting.py
│   ├── frequency/
│   │   ├── __init__.py
│   │   ├── build.py
│   │   ├── residual.py
│   │   ├── spec.py
│   │   ├── spectral_gate.py
│   │   └── utils.py
│   ├── layers/
│   │   ├── RevIN.py
│   │   └── rotaryembedding.py
│   ├── macro/
│   │   ├── __init__.py
│   │   ├── agg.py
│   │   ├── aggregation_utils.py
│   │   ├── dca_aggregator.py
│   │   ├── dpm_sampler.py
│   │   └── dpm_solver.py
│   ├── micro/
│   │   ├── __init__.py
│   │   ├── adapt.py
│   │   ├── inject.py
│   │   ├── stek_backbone.py
│   │   └── tek.py
│   ├── utils/
│   │   ├── diffusion_utils.py
│   │   ├── metrics.py
│   │   ├── print_args.py
│   │   └── tools.py
│   └── plot/
│       └── overview/
│           ├── flow.py
│           └── holidiff_framework.svg
├── docs/
└── README.md
```

---

## 4. Important modules

### 4.1 Top-level model

#### `Holidiff/HoliDiff.py`
Defines the top-level `HATEK` model.

Main responsibilities:
- diffusion schedule setup
- training-time micro generation
- inference-time repeated sampling
- macro aggregation dispatch

---

### 4.2 Training entry

#### `Holidiff/train.py`
The unified YAML-based training entry.

Current behavior:
- reads YAML config
- builds `Exp_Long_Term_Forecast`
- runs train / validation / test

This entry is now intentionally narrowed to the long-term forecasting path.

---

### 4.3 Micro modules

#### `Holidiff/micro/adapt.py`
Unified micro adaptation module.

Contains:
- `FieldStatsProvider`
- `BaseTargetSpaceAdapter`
- `VanillaNIAdapter`
- `SFCN`
- `IdentityLocalScaling`
- `LocalAdaptiveScaling`
- `build_target_adapter`
- `build_revin_adapter`

This file is the merged result of the old target-adapter / graph-stat / adapter-factory pieces.

#### `Holidiff/micro/inject.py`
Residual injection wrapper.

Main role:
- connect physical/time/frequency residual correction into the model output path

#### `Holidiff/micro/tek.py`
Defines `TEK`.

Main role:
- wrap the spatiotemporal backbone
- reshape inputs and call the backbone

#### `Holidiff/micro/stek_backbone.py`
Defines the main spatiotemporal backbone.

Main role:
- patchify history and future tokens
- inject time token
- run attention blocks
- output future estimate

---

### 4.4 Frequency modules

#### `Holidiff/frequency/spec.py`
Merged spectral representation module.

Contains:
- `FixedBandDecomposer`
- `BandSpecificTrendAwarePatchEmbed`

#### `Holidiff/frequency/build.py`
Frequency builder entry.

Contains:
- `build_frequency_decomposer`
- `build_frequency_patch_embed`
- `build_frequency_residual`

#### `Holidiff/frequency/residual.py`
Residual definitions for:
- time residual
- frequency residual
- hybrid residual

Ablation note:
- if frequency residual is disabled, the retained fallback path is `time residual`

---

### 4.5 Macro modules

#### `Holidiff/macro/agg.py`
Short aggregation entry.

Contains:
- `build_macro_aggregator`

This is the active macro aggregation dispatch file.

#### `Holidiff/macro/dca_aggregator.py`
Implements the DCA aggregation logic.

`dca` is retained as the current main paper innovation path.

#### `Holidiff/macro/dpm_sampler.py`
DPM-Solver sampler wrapper used during inference-time diffusion sampling.

---

### 4.6 Experiment modules

#### `Holidiff/exp/exp_basic.py`
Base experiment class.

#### `Holidiff/exp/exp_long_term_forecasting.py`
The only active experiment implementation in the current repository state.

Main responsibilities:
- build model
- train / validate / test
- compute masked loss
- report metrics
- load checkpoints compatibly

---

### 4.7 Data modules

#### `Holidiff/data_provider/fujian30_loader.py`
Dedicated loader for the Fujian-30 traffic dataset.

#### `Holidiff/data/fujian-30/adjacent_gantry.csv`
Adjacency file used by the graph-aware spatial field path.

---

## 5. Config files

### `Holidiff/configs/main_fujian30_dca.yaml`
Main paper-line reference config.

Use this when you want:
- the current main Fujian-30 setup
- DCA as the aggregation reference
- the cleaned repository structure

### `Holidiff/configs/quickrun_fujian30_1epoch.yaml`
Quick regression config.

Use this when you want:
- a 1-epoch smoke test
- fast structural verification after refactors

---

## 6. How to run

Recommended environment:

```bash
conda activate holiday
cd /root/yanyijin/STdiff
```

### Main reference run

```bash
python Holidiff/train.py --config Holidiff/configs/main_fujian30_dca.yaml
```

### Quick 1-epoch verification

```bash
python Holidiff/train.py --config Holidiff/configs/quickrun_fujian30_1epoch.yaml
```

### Optional custom run tag

```bash
python Holidiff/train.py --config Holidiff/configs/main_fujian30_dca.yaml --version my_run
```

---

## 7. Active experimental surface

The current repository intentionally focuses on the following experiment surface.

### Kept
- `long_term_forecast`
- `HATEK`
- `DCA` main line
- aggregation ablations:
  - `simple`
  - `median`
  - `mom`
  - `dca`
- frequency ablation via `frequency_enable`
- non-graph RevIN ablation via `new_norm`

### Removed from active code surface
- classification task entry
- imputation task entry
- anomaly detection task entry
- short-term forecasting task entry
- holiday-conditioned model call path
- older fragmented module names replaced by merged short-name files

---

## 8. Notes on naming

Current shortened module naming follows this pattern:

- `micro/adapt.py` for target-space / graph / RevIN adaptation
- `micro/inject.py` for residual injection
- `frequency/spec.py` for spectral representation
- `frequency/build.py` for frequency module builders
- `macro/agg.py` for macro aggregation dispatch

The goal is to keep names short while preserving paper-facing semantics.

---

## 9. Output artifacts

Training uses:
- checkpoint directory from YAML: `Holidiff/checkpoints/`
- standard test/result folders created by the experiment code when needed

Temporary debug logs are not intended to be kept in the repository.

---

## 10. Related files

- `Holidiff/utils/print_args.py` for formatted run summaries
- `Holidiff/utils/diffusion_utils.py` for diffusion helpers
- `Holidiff/layers/RevIN.py` for RevIN implementation
- `Holidiff/layers/rotaryembedding.py` for rotary attention embedding
- `Holidiff/plot/overview/holidiff_framework.svg` for overview figure output

---

## 11. Current recommended usage

If you only need one stable starting point, use:

```bash
python Holidiff/train.py --config Holidiff/configs/main_fujian30_dca.yaml
```

If you are refactoring structure and want a quick safety check, use:

```bash
python Holidiff/train.py --config Holidiff/configs/quickrun_fujian30_1epoch.yaml
```

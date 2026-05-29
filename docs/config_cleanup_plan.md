# 配置文件清理规划

> **目标**: 配置文件只保留论文终版需要的参数，其他回归默认值

---

## 📊 当前配置分析

### 当前配置文件问题

**compare_fujian30_standard_1epoch.yaml** 包含 **156 行配置**，其中：

1. **废弃参数**（已不使用）：约 40+ 个
2. **实验性参数**（非论文终版）：约 30+ 个
3. **默认值参数**（可省略）：约 50+ 个
4. **论文终版必需参数**：约 30 个

---

## 🎯 论文终版配置规划

### 核心原则

1. **只保留论文终版使用的参数**
2. **其他参数回归代码默认值**
3. **配置文件清晰对应论文方法**
4. **便于复现和理解**

### 论文终版主路径

```
LSTDE (Fixed FFT, K=4, patch_len=12)
→ Trend-Aware PatchEmbed (concat)
→ SFCN (learnable α=0.05)
→ Diffusion Denoiser (100 steps)
→ Hybrid Residual (η=0.5)
→ DCA Aggregator
```

---

## 📋 配置参数分类

### 类别 1: 必需保留（论文终版）

#### 基础配置
```yaml
task_name: long_term_forecast
model: HATEK
is_diff: true
```

#### 数据配置
```yaml
data: fujian30
root_path: ./Holidiff/data/fujian-30
data_path: fujian30_clean.csv
features: M
freq: 15min
enc_in: 30
```

#### 序列长度
```yaml
seq_len: 96
label_len: 48
pred_len: 12
```

#### 模型架构
```yaml
d_model: 128
n_heads: 8
e_layers: 1
d_layers: 1
d_ff: 256
```

#### LSTDE (Fixed FFT, K=4)
```yaml
frequency_enable: true
frequency_decomp_type: fixed_fft
frequency_num_bands: 4
```

#### Trend-Aware PatchEmbed
```yaml
patch_len: 12
stride: 1
frequency_patch_embed_mode: band_trend
frequency_patch_embed_fusion: concat_proj
frequency_injection_mode: embed_replace
```

#### SFCN (learnable α)
```yaml
phase_e_enable: true
phase_e_target_adapter: matrix_ni
phase_e_zeta_init: 0.05
phase_e_edge_var_window: 720
phase_e_field_stats_source: future
phase_e_adj_file: adjacent_gantry.csv
```

#### Hybrid Residual (η=0.5)
```yaml
frequency_residual_type: hybrid_residual
physical_residual_eta: 0.5
```

#### DCA Aggregator
```yaml
aggregation_mode: dca
dca_bandwidth_kde: 15.0
dca_bandwidth_hist: 20.0
dca_mode: joint
```

#### Diffusion
```yaml
diff_steps: 100
s_steps: 10
sample_times: 10
```

#### 训练配置
```yaml
train_epochs: 30
batch_size: 32
learning_rate: 0.0001
loss_type: MAE
```

#### 硬件配置
```yaml
use_gpu: true
gpu: 0
num_workers: 4
```

**小计**: 约 **45 个必需参数**

---

### 类别 2: 废弃参数（需要删除）

#### 已废弃的聚合器参数
```yaml
use_mom: 1                    # ❌ 已移除 MoM
rmom: 5                       # ❌ MoM 参数
n_b: 5                        # ❌ MoM 参数
abms_m_neighbor: 3            # ❌ ABMS 已废弃
abms_max_iter: 10             # ❌ ABMS 已废弃
abms_tol: 0.001               # ❌ ABMS 已废弃
rcl_bandwidth_hist: 20.0      # ❌ RCL 已废弃
rcl_mode: soft                # ❌ RCL 已废弃
scp_lambda_skew: 0.5          # ❌ SCP 非主方法
mom_version: "baseline"       # ❌ MoM 已废弃
mom_alpha: 0.4                # ❌ MoM 已废弃
mom_topk: 3                   # ❌ MoM 已废弃
mom_trend_temp: 5.0           # ❌ MoM 已废弃
mom_lambda: 0.7               # ❌ MoM 已废弃
```

#### Holiday 条件化参数（已移除）
```yaml
holiday_enable: false         # ❌ 已移除
holiday_mode: none            # ❌ 已移除
holiday_alpha_delta: 0.05     # ❌ 已移除
holiday_hard_alpha: 0.10      # ❌ 已移除
holiday_dropout_prob: 0.0     # ❌ 已移除
holiday_dual_bank_enable: false # ❌ 已移除
```

#### 未使用的模型参数
```yaml
expand: 2                     # ❌ 未使用
d_conv: 4                     # ❌ 未使用
top_k: 5                      # ❌ 未使用
num_kernels: 6                # ❌ 未使用
num_heads: 8                  # ❌ 重复（已有 n_heads）
dec_in: 30                    # ❌ 未使用
c_out: 30                     # ❌ 未使用
moving_avg: 25                # ❌ 未使用
factor: 1                     # ❌ 未使用
distil: true                  # ❌ 未使用
skip_dropout: 0.4             # ❌ 未使用
channel_independence: 1       # ❌ 未使用
decomp_method: moving_avg     # ❌ 未使用
down_sampling_layers: 0       # ❌ 未使用
down_sampling_window: 1       # ❌ 未使用
down_sampling_method: null    # ❌ 未使用
seg_len: 48                   # ❌ 未使用
```

#### 未使用的训练参数
```yaml
seasonal_patterns: Monthly    # ❌ 未使用
inverse: false                # ❌ 未使用
vs_times: 10                  # ❌ 未使用
skip_type: time_quadratic     # ❌ 未使用
method: multistep             # ❌ 未使用
lower_order_final: true       # ❌ 未使用
order: 2                      # ❌ 未使用
des: Exp                      # ❌ 未使用
loss: MSE                     # ❌ 重复（已有 loss_type）
lradj: type1                  # ❌ 未使用
use_amp: false                # ❌ 未使用
use_multi_gpu: false          # ❌ 未使用
devices: '0'                  # ❌ 重复（已有 gpu）
```

#### 数据增强参数（未使用）
```yaml
use_dtw: false
augmentation_ratio: 0
jitter: false
scaling: false
permutation: false
randompermutation: false
magwarp: false
timewarp: false
windowslice: false
windowwarp: false
rotation: false
spawner: false
dtwwarp: false
shapedtwwarp: false
wdba: false
discdtw: false
discsdtw: false
```

#### 其他未使用参数
```yaml
p_hidden_dims: [128, 128]     # ❌ 未使用
p_hidden_layers: 2            # ❌ 未使用
use_shuffle: 1                # ❌ 未使用
use_first: 0                  # ❌ 未使用
extra_tag: ''                 # ❌ 未使用
mode: standard                # ❌ 未使用
```

**小计**: 约 **70+ 个废弃参数**

---

### 类别 3: 可选参数（回归默认值）

这些参数在代码中有默认值，可以从配置文件中删除：

```yaml
dropout: 0.0                  # 默认值
activation: gelu              # 默认值
embed: timeF                  # 默认值
use_norm: 1                   # 默认值
scale: true                   # 默认值
seed: 2021                    # 默认值
patience: 10                  # 默认值
itr: 1                        # 默认值
checkpoints: ...              # 可自动生成
model_id: ...                 # 可自动生成
target: traffic_flow          # 可推断
coss: 5.0                     # 默认值
density_bandwidth: 15.0       # 默认值（与 dca_bandwidth_kde 重复）
density_eps: 1.0e-8           # 默认值
physical_injection_enable: true  # 默认值
physical_residual_enable: true   # 默认值
physical_free_flow_epsilon: 0.0  # 默认值
frequency_eta_init: [...]     # 默认值
frequency_beta_init: [...]    # 默认值
frequency_use_softplus_eta: true # 默认值
frequency_hybrid_mix_alpha: 0.5  # 默认值
frequency_patch_embed_num_bands: 4  # 与 frequency_num_bands 重复
phase_e_revin_adapter: vanilla_revin  # 默认值
phase_e_graph_mode: directed  # 默认值
phase_e_field_eta: 0.1        # 默认值
phase_e_use_edge_sigma_fallback: false  # 默认值
new_norm: 1                   # 默认值
```

**小计**: 约 **30 个可选参数**

---

## ✨ 论文终版配置文件（精简版）

### 文件结构

```
Holidiff/configs/
├── paper/
│   ├── fujian30_holidiff.yaml          # 论文终版主配置
│   ├── fujian30_holidiff_quick.yaml    # 快速测试（1 epoch）
│   └── README.md                        # 配置说明
└── ablation/
    ├── wo_lstde.yaml                    # 消融：无 LSTDE
    ├── wo_trend.yaml                    # 消融：无 Trend-Aware
    ├── wo_sfcn.yaml                     # 消融：无 SFCN
    ├── wo_residual.yaml                 # 消融：无 Residual
    ├── mean_agg.yaml                    # 消融：Mean 聚合
    ├── median_agg.yaml                  # 消融：Median 聚合
    └── mom_agg.yaml                     # 消融：MoM 聚合
```

### 精简后的主配置（约 50 行）

```yaml
# ============================================
# HoliDiff Paper Final Configuration
# ============================================

# Task
task_name: long_term_forecast
model: HATEK
is_diff: true

# Data
data: fujian30
root_path: ./Holidiff/data/fujian-30
data_path: fujian30_clean.csv
features: M
freq: 15min
enc_in: 30

# Sequence
seq_len: 96
label_len: 48
pred_len: 12

# Model Architecture
d_model: 128
n_heads: 8
e_layers: 1
d_layers: 1
d_ff: 256
dropout: 0.0

# LSTDE: Fixed FFT (K=4)
frequency_enable: true
frequency_decomp_type: fixed_fft
frequency_num_bands: 4

# Trend-Aware PatchEmbed (concat, patch_len=12)
patch_len: 12
stride: 1
frequency_patch_embed_mode: band_trend
frequency_patch_embed_fusion: concat_proj
frequency_injection_mode: embed_replace

# SFCN: Spatial Field Coupled Normalization (learnable α=0.05)
phase_e_enable: true
phase_e_target_adapter: matrix_ni
phase_e_zeta_init: 0.05
phase_e_edge_var_window: 720
phase_e_field_stats_source: future
phase_e_adj_file: adjacent_gantry.csv

# Hybrid Residual (η=0.5)
frequency_residual_type: hybrid_residual
physical_residual_eta: 0.5

# DCA: Density-Centroid Aggregation
aggregation_mode: dca
dca_bandwidth_kde: 15.0
dca_bandwidth_hist: 20.0
dca_mode: joint

# Diffusion
diff_steps: 100
s_steps: 10
sample_times: 10

# Training
train_epochs: 30
batch_size: 32
learning_rate: 0.0001
loss_type: MAE

# Hardware
use_gpu: true
gpu: 0
num_workers: 4
```

**精简效果**: 从 **156 行** → **约 60 行**（减少 **60%**）

---

## 🔄 代码默认值设置

### 需要在代码中设置的默认值

为了让配置文件更简洁，需要在代码中设置合理的默认值：

#### 1. `exp/exp_basic.py` 或 `train.py`

```python
# 设置默认值
parser.add_argument('--model_id', type=str, default=None)  # 自动生成
parser.add_argument('--checkpoints', type=str, default='./checkpoints')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--scale', type=bool, default=True)
```

#### 2. `frequency/factory.py`

```python
def build_frequency_residual(configs):
    # 设置默认值
    eta = float(getattr(configs, 'physical_residual_eta', 0.5))  # 默认 0.5
    free_flow_epsilon = float(getattr(configs, 'physical_free_flow_epsilon', 0.0))
    # ...
```

#### 3. `micro/sfcn.py`

```python
class SFCN(VanillaNIAdapter):
    def __init__(
        self,
        adj: torch.Tensor,
        coupling_init: float = 0.05,  # 默认 0.05
        eps: float = 1e-5,
        edge_var_window: int = 720,  # 默认 720
        field_stats_source: str = 'future',  # 默认 'future'
    ):
        # ...
```

#### 4. `macro/dca_aggregator.py`

```python
class DensityCentroidAggregator:
    def __init__(
        self,
        bandwidth_kde: float = 15.0,  # 默认 15.0
        bandwidth_hist: float = 20.0,  # 默认 20.0
        mode: str = 'joint',  # 默认 'joint'
        eps: float = 1e-8,
    ):
        # ...
```

---

## 📝 消融实验配置

### 消融实验只需修改少量参数

#### wo_lstde.yaml（无 LSTDE）

```yaml
# 继承主配置，只修改：
frequency_enable: false
```

#### wo_trend.yaml（无 Trend-Aware）

```yaml
# 继承主配置，只修改：
frequency_patch_embed_mode: identity  # 或删除此行使用默认
```

#### wo_sfcn.yaml（无 SFCN）

```yaml
# 继承主配置，只修改：
phase_e_enable: false
```

#### mean_agg.yaml（Mean 聚合）

```yaml
# 继承主配置，只修改：
aggregation_mode: simple  # 或 mean
```

---

## 🎯 清理执行计划

### 步骤 1: 创建论文终版配置

1. 创建 `configs/paper/` 目录
2. 创建精简的 `fujian30_holidiff.yaml`（约 60 行）
3. 创建快速测试版 `fujian30_holidiff_quick.yaml`（train_epochs: 1）

### 步骤 2: 更新代码默认值

1. 更新 `exp/exp_basic.py` 或参数解析器
2. 更新各模块的 `__init__` 方法默认值
3. 确保所有默认值与论文终版一致

### 步骤 3: 创建消融实验配置

1. 创建 `configs/ablation/` 目录
2. 创建各消融实验配置（每个只修改 1-2 个参数）

### 步骤 4: 清理旧配置

1. 将 `compare_fujian30_standard_1epoch.yaml` 移到 `legacy/configs/`
2. 更新文档说明新的配置结构

### 步骤 5: 测试

1. 使用新配置运行快速测试
2. 验证所有参数正确加载
3. 确认模型行为与之前一致

---

## 📊 清理效果对比

| 指标 | 清理前 | 清理后 | 改进 |
|------|--------|--------|------|
| 配置行数 | 156 行 | 60 行 | **-60%** |
| 必需参数 | 混杂 | 45 个 | 清晰 |
| 废弃参数 | 70+ 个 | 0 个 | **-100%** |
| 可读性 | 混乱 | 清晰 | **显著提升** |
| 对应论文 | 模糊 | 直接 | **一一对应** |

---

## 💡 配置文件设计原则

1. **最小化原则**: 只保留论文终版必需的参数
2. **默认值原则**: 其他参数回归代码默认值
3. **清晰性原则**: 配置文件直接对应论文方法
4. **可复现原则**: 便于他人理解和复现
5. **分层原则**: 主配置 + 消融配置分离

---

**规划完成时间**: 2026-05-27  
**下一步**: 创建精简的论文终版配置文件

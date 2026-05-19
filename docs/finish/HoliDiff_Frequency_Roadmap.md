# HoliDiff Frequency Module Roadmap

> 创建时间：2026-05-19  
> 目的：记录 HoliDiff 频域增强模块的实施路径，避免后续遗忘，并为 `Holidiff/frequency` 可插拔架构提供统一参考。

---

## 1. 当前判断

### 1.1 当前保留模型
当前论文保留版本为：
- `Trend-Aware PatchEmbed (concat)`
- `fixed Historical Trend Residual (eta=0.5)`

该组合已经说明两件事：
1. 局部趋势信息对 HoliDiff 是有效的；
2. 输出端的显式趋势校正也有效。

### 1.2 频域方向判断
频域方向是合理的，但不建议一步到位直接上完整 LSTDE。更稳妥的路径是：
- 先验证频带级残差校正是否有效；
- 再做输入侧频域 conditioning；
- 最后再考虑频带级 PatchEmbed / 完整 LSTDE 替换。

结论：
> 方向正确，但第一阶段应优先做低侵入、可验证、可回滚的频域后处理模块。

---

## 2. 总体实施策略

### Phase A：方向验证
目标：不修改 TEK 主干，仅验证频域残差后处理是否优于当前 `eta=0.5` 时间域校正。

计划：
- 从 `raw_history` 做频带分解；
- 计算每个频带的历史斜率；
- 通过固定 `eta_k` 和 `beta_k` 形成频带级 correction；
- 与当前 `physical_residual_eta=0.5` 基线对照。

优先级：最高。

### Phase B：可插拔模块化
目标：将频域逻辑沉淀到 `Holidiff/frequency` 目录，形成统一工厂与接口。

计划：
- 抽出 `LearnableSpectralGate`；
- 抽出 `FrequencyResidualModule`；
- 使用 `factory.py` 统一构建；
- 配置中支持 `none / time_residual / frequency_residual / hybrid_residual`。

### Phase C：输入侧增强
目标：将频域信息以 conditioning / residual branch 的形式注入 TEK，而不是立刻替换主干 patch embed。

计划：
- 新增 `FrequencyConditioner`；
- 从 `raw_history` 生成频带 summary；
- 后续在 TEK 或 backbone 中接入。

### Phase D：完整版 LSTDE
目标：在前面三阶段均证明有效后，再进入完整频带级 PatchEmbed 替换路线。

计划：
- 频带级 Trend-Aware PatchEmbed；
- Band Modulator；
- 频带融合；
- 推理时频带级残差校正。

---

## 3. 推荐代码结构

建议在 `Holidiff/frequency` 下建立如下结构：

```text
Holidiff/frequency/
  __init__.py
  spectral_gate.py
  decompose.py
  residual.py
  conditioner.py
  patch_embed.py
  modulator.py
  fusion.py
  factory.py
  utils.py
```

### 各文件职责

#### `spectral_gate.py`
- 放 `LearnableSpectralGate`
- 负责 RFFT + 频带 soft assignment + IRFFT

#### `decompose.py`
- 放不同分解器统一接口
- 第一阶段至少包括：
  - `IdentityDecomposer`
  - `FixedBandDecomposer`
  - `LearnableBandDecomposer`

#### `residual.py`
- 第一阶段核心模块
- 放：
  - `IdentityResidualModule`
  - `TimeResidualModule`
  - `FrequencyResidualModule`
  - `HybridResidualModule`

#### `conditioner.py`
- 暂时预留 Phase C
- 第一版可以为空实现或 identity

#### `patch_embed.py`
- 暂时预留 Phase D
- 第一版只放占位骨架

#### `modulator.py`
- 暂时预留 Phase D
- 第一版只放占位骨架

#### `fusion.py`
- 暂时预留 Phase C/D
- 第一版只放占位骨架

#### `factory.py`
- 根据 config 构建 decomposer / residual / conditioner 等
- 保证频域能力真正可插拔

#### `utils.py`
- 放公共工具，例如：
  - 历史斜率计算
  - 频带形状检查
  - horizon step 构建

---

## 4. 当前最优实现顺序

### MVP-v1：Frequency Residual Only
第一版只做：
- `frequency/decompose.py`
- `frequency/residual.py`
- `frequency/factory.py`
- `frequency/utils.py`

并通过 `PhysicalInjectionModule` 或上层显式调用接入。

### 对照组
建议至少保留以下实验组：
1. `none`
2. `time_residual`（当前 best, eta=0.5）
3. `frequency_residual`（fixed bands）
4. `frequency_residual`（learnable gate）
5. `hybrid_residual`

### 判断标准
- 是否继续做 Phase C / D，优先看：
  - `CG_MAE`
  - `FF_MAE`
  - `TPR_CG`
- 若频域残差都不能优于当前 `eta=0.5`，则不建议立即大改输入侧主干。

---

## 5. 推荐配置开关

后续建议在配置中逐步加入：

```yaml
frequency_enable: false
frequency_decomp_type: "learnable_fft"
frequency_num_bands: 3
frequency_input_layout: "BTN"

frequency_injection_mode: "none"
frequency_residual_type: "band_trend"
frequency_fusion_type: "concat"

frequency_eta_init: [0.2, 0.5, 1.0]
frequency_beta_init: [0.2, 0.3, 0.5]
frequency_use_learnable_beta: false
frequency_use_softplus_eta: true

frequency_freeze_backbone: false
frequency_warmup_epochs: 0
```

第一阶段不要求全部生效，但建议先统一命名。

---

## 6. 最终原则

1. **先验证方向，再深挖结构**
2. **先外挂分支，再考虑主干替换**
3. **先做 residual MVP，再做 conditioning / patch embed**
4. **所有频域能力都应通过 `factory.py` 控制，保证可插拔、可回滚、可对照**

---

## 7. 当前执行决定

当前决定按以下顺序推进：
1. 在 `docs/` 中记录本路线；
2. 建立 `Holidiff/frequency` 目录与全部骨架 py 文件；
3. 先填好 `Phase A / Phase B` 需要的最小接口；
4. `Phase C / Phase D` 暂时只保留空骨架或 identity 实现。

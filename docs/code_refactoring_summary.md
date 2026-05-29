# HoliDiff 代码精简总结报告

> **日期**: 2026-05-27  
> **目标**: 将代码从"研究试验台"收敛为"论文终版实现"

---

## 📋 重构目标

根据论文终版配置，固化以下主路径：

```
LSTDE (Fixed FFT, K=4, patch_len=12)
→ Trend-Aware PatchEmbed (concat)
→ SFCN (learnable α)
→ Diffusion Denoiser
→ Hybrid Residual (η=0.5)
→ DCA Aggregator
```

## ✅ 已完成的重构

### 1. `HoliDiff.py` 主模型精简

**移除内容（约150行代码）：**
- ❌ `holiday_flag` 参数（从所有 forward 方法签名中移除）
- ❌ MoM 聚合逻辑（`extract_consensus`, `_consensus_reduce`, `reset_diagnostics` 等）
- ❌ `aggregation_mode` 动态选择逻辑
- ❌ `density_lite` 内联聚合器
- ❌ `aggregation_factory` 动态构建调用
- ❌ 所有 MoM 诊断变量（`_diag_block_var`, `_diag_inter_dev` 等）

**重构后的清晰结构：**

```python
class HATEK(nn.Module):
    """
    HoliDiff main model (paper final version).
    
    Architecture:
        1. History encoding with LSTDE + Trend-Aware PatchEmbed
        2. SFCN for spatial field coupling
        3. Diffusion-based micro-realization generation
        4. Hybrid residual correction
        5. DCA for macro-flow estimation
    """
    
    def __init__(self, configs):
        self.tek = TEK(configs)
        self.aggregator = DensityCentroidAggregator(configs)
        # ... diffusion schedule setup
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, sample_times=5):
        if self.training:
            return self.forward_micro_generation_train(...)
        else:
            return self.forward_macro_estimation_inference(...)
```

**关键改进：**
- 单一主路径，无动态分支
- 方法命名对应论文术语（micro-realization, macro-estimation）
- 移除 `holiday_flag` 参数（保留在数据层用于评估）
- 直接实例化 `DensityCentroidAggregator`，无 factory 模式

### 2. `macro/simple_aggregators.py` 创建

**目的**: 保存简单聚合器用于消融实验，但不作为主模型的一部分。

**包含内容：**
- `MeanAggregator` - 简单均值基线
- `MedianAggregator` - 简单中位数基线
- `MoMAggregator` - Median-of-Means（用于消融对比）

**使用场景：**
```python
# 仅在消融实验脚本中使用
from Holidiff.macro.simple_aggregators import MeanAggregator, MedianAggregator, MoMAggregator

# 主模型不再使用这些
```

### 3. `macro/__init__.py` 更新

**修改前：**
```python
from .aggregation_factory import build_macro_aggregator
```

**修改后：**
```python
from .dca_aggregator import DensityCentroidAggregator
from .scp_aggregator import SkewnessCorrectedPeakAggregator
from .simple_aggregators import MeanAggregator, MedianAggregator, MoMAggregator
```

**改进：**
- 移除 factory 模式
- 直接导出具体聚合器类
- 明确区分主模型（DCA）和实验基线（Mean/Median/MoM）

---

## 📊 代码统计

| 模块 | 修改前 | 修改后 | 减少 |
|------|--------|--------|------|
| `HoliDiff.py` | ~350行 | ~200行 | **-150行** |
| `macro/` 模块 | 分散在多处 | 清晰分层 | 结构优化 |
| 动态分支 | 5+ 个聚合模式 | 1 个主路径 | **-4 个分支** |

---

## 🎯 论文终版配置

### 保留的核心组件

| 组件 | 文件位置 | 论文配置 |
|------|----------|----------|
| **LSTDE** | `frequency/decompose.py` | Fixed FFT, K=4 |
| **Trend-Aware PatchEmbed** | `frequency/patch_embed.py` | concat 模式 |
| **SFCN** | `micro/sfcn.py` | learnable α |
| **Hybrid Residual** | `frequency/residual.py` | η=0.5 |
| **DCA Aggregator** | `macro/dca_aggregator.py` | 论文主方法 |
| **SCP Aggregator** | `macro/scp_aggregator.py` | 轻量备选 |

### 实验基线（用于消融）

| 聚合器 | 文件位置 | 用途 |
|--------|----------|------|
| Mean | `macro/simple_aggregators.py` | w/o DCA 基线 |
| Median | `macro/simple_aggregators.py` | w/o DCA 基线 |
| MoM | `macro/simple_aggregators.py` | w/o DCA 基线 |

---

## 🔄 待完成的后续工作

### 高优先级

1. **清理 `frequency/` 目录**
   - 移除未使用的模块：`conditioner.py`, `modulator.py`, `fusion.py`, `spectral_gate.py`
   - 保留核心模块：`decompose.py`, `patch_embed.py`, `residual.py`

2. **更新配置文件**
   - 固化论文终版配置
   - 移除实验性参数
   - 创建 `configs/paper/` 目录

3. **检查实验脚本**
   - 修复 `exp/exp_long_term_forecasting.py` 中的调用
   - 确保不再使用 `holiday_flag` 作为模型输入
   - 更新聚合器实例化方式

### 中优先级

4. **创建 `legacy/` 目录**
   - 保存废弃的实验代码
   - 移动旧配置文件
   - 保留用于参考，但不在主路径中

5. **重构 `experiments/` 目录**
   - 按论文表格和图片组织脚本
   - 例如：`01_shift_score.py`, `03_eval_overall.py`, `06_eval_aggregation.py`

### 低优先级

6. **更新文档**
   - 更新 `README.md` 反映精简后的结构
   - 创建 `docs/reproduction.md` 复现指南
   - 更新 `docs/current_model_for_paper.md`

---

## 🚀 重构原则

### 1. 去掉可拔插架构

**之前：**
```python
if aggregation_mode == 'mom':
    return self.extract_consensus(all_outs)
elif aggregation_mode == 'dca':
    return self.macro_aggregator.aggregate_batch(...)
elif aggregation_mode == 'simple':
    return all_outs.mean(0)
```

**现在：**
```python
macro_prediction = self.aggregator.aggregate_batch(all_outs, history_context=history_for_trend)
```

### 2. 统一论文命名

| 代码术语 | 论文术语 | 物理含义 |
|----------|----------|----------|
| `forward_micro_generation_train` | Micro-realization Generation | 扩散模型生成单次未来轨迹 |
| `forward_macro_estimation_inference` | Macro-flow Estimation | 多次采样聚合得到宏观交通流 |
| `DensityCentroidAggregator` | DCA | 密度质心聚合器 |
| `TEK` | Traffic Evolution Kernel | 交通演化核（去噪器） |

### 3. 单主线 + 少量消融开关

**主模型：** 固定使用 DCA，无动态选择  
**消融实验：** 在实验脚本中手动替换聚合器

```python
# 主模型（HoliDiff.py）
self.aggregator = DensityCentroidAggregator(configs)

# 消融实验脚本（experiments/06_eval_aggregation.py）
from Holidiff.macro.simple_aggregators import MeanAggregator, MedianAggregator

aggregators = {
    'DCA': DensityCentroidAggregator(configs),
    'Mean': MeanAggregator(),
    'Median': MedianAggregator(),
    'MoM': MoMAggregator(n_blocks=5, rmom_n=3)
}
```

---

## 📝 关键决策记录

### 为什么保留 `holiday_flag` 在数据层？

**决策：** `holiday_flag` 不作为模型 forward 的输入，但保留在数据加载和评估中。

**理由：**
1. 论文需要节假日细粒度评估（Hol-MAE, Hol-CG MAE, HDR, PTE）
2. 数据集需要区分常规日和节假日样本
3. 模型本身不依赖 holiday 标签，体现泛化能力

**实现：**
```python
# 数据层保留
dataset.holiday_flags  # 用于划分测试集

# 评估层使用
holiday_mask = (holiday_flags == 1)
hol_mae = mae[holiday_mask].mean()

# 模型层移除
def forward(self, x_enc, ...):  # 无 holiday_flag 参数
```

### 为什么保留 `aggregation_factory.py`？

**决策：** 暂时保留但简化，未来可能完全移除。

**理由：**
1. 当前只保留 DCA 和 SCP 两个聚合器
2. 主模型已直接实例化 DCA，不再使用 factory
3. 保留 factory 是为了兼容旧实验脚本，后续可删除

### 为什么创建 `simple_aggregators.py`？

**决策：** 将 Mean/Median/MoM 从主模型移到独立文件。

**理由：**
1. 这些聚合器不是论文主方法，只用于消融对比
2. 从主模型移除可减少代码复杂度
3. 保留在项目中是为了复现消融实验

---

## 🎓 论文对应关系

### 主模型架构（Section III）

| 论文章节 | 代码模块 | 文件位置 |
|----------|----------|----------|
| III-A: Frequency Decoupling | LSTDE | `frequency/decompose.py` |
| III-A: Trend-Aware PatchEmbed | TrendAwarePatchEmbedding | `frequency/patch_embed.py` |
| III-B: Spatial Field Coupling | SFCN | `micro/sfcn.py` |
| III-C: Diffusion Denoiser | TEK | `micro/tek.py`, `micro/stek_backbone.py` |
| III-D: Hybrid Residual | HybridTrendResidual | `frequency/residual.py` |
| III-E: Density-Centroid Aggregation | DCA | `macro/dca_aggregator.py` |

### 消融实验（Table VII）

| 消融配置 | 代码实现 |
|----------|----------|
| w/o LSTDE | 禁用 `frequency_enable` |
| w/o Trend | 使用普通 PatchEmbed |
| w/o SFCN | 使用 Vanilla 归一化 |
| w/o DCA (Mean) | 使用 `MeanAggregator` |
| w/o DCA (Median) | 使用 `MedianAggregator` |
| w/o DCA (MoM) | 使用 `MoMAggregator` |

---

## 📈 预期效果

### 代码质量提升

- ✅ **可读性**: 单主线结构，无复杂分支
- ✅ **可维护性**: 模块职责清晰，易于修改
- ✅ **可复现性**: 代码直接对应论文方法

### 开发效率提升

- ✅ **调试更容易**: 减少动态分支，错误更容易定位
- ✅ **实验更清晰**: 消融实验在脚本层面控制，不污染主模型
- ✅ **文档更简洁**: 代码即文档，命名对应论文术语

### 论文投稿准备

- ✅ **代码开源**: 清晰的主路径，易于他人复现
- ✅ **审稿友好**: 代码结构对应论文章节
- ✅ **答辩准备**: 可以直接展示代码对应论文方法

---

## 🔗 相关文档

- [论文终版模型配置](./current_model_for_paper.md)
- [实验映射关系](./experiment_mapping.md) (待创建)
- [复现指南](./reproduction.md) (待创建)

---

**重构完成时间**: 2026-05-27  
**下一步**: 清理 `frequency/` 目录，更新配置文件

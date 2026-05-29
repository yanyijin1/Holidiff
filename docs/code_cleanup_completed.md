# 代码清理完成报告

> **日期**: 2026-05-27  
> **目标**: 移除主路径中不再使用的 `holiday_flag` 相关代码

---

## ✅ 已完成的清理工作

### 1. HoliDiff.py 主模型 ✓

**清理内容：**
- ✅ 移除 `forward()` 方法中的 `holiday_flag` 参数
- ✅ 移除 `forward_micro_generation_train()` 中的 `holiday_flag` 参数
- ✅ 移除 `forward_consensus_inference()` 中的 `holiday_flag` 参数（已重命名为 `forward_macro_estimation_inference`）
- ✅ 移除所有 MoM 聚合逻辑（约150行代码）
- ✅ 移除 `aggregation_mode` 动态选择逻辑
- ✅ 直接使用 `DensityCentroidAggregator`，无 factory 模式

**结果：** 主模型现在是干净的单主线结构，无 holiday 条件化分支。

### 2. exp/exp_long_term_forecasting.py 实验脚本 ✓

**清理内容：**
- ✅ 移除训练循环中的 `batch_holiday` 变量提取
- ✅ 移除模型调用时的 `holiday_flag=batch_holiday` 参数

**修改前：**
```python
batch_holiday = batch[5].float().to(self.device) if len(batch) > 5 else None
outputs, _ = self.model(..., holiday_flag=batch_holiday)
```

**修改后：**
```python
outputs, _ = self.model(..., sample_times=self.args.sample_times)
```

### 3. macro/simple_aggregators.py 创建 ✓

**目的：** 将 MoM、Mean、Median 聚合器从主模型移到独立文件，用于消融实验。

**包含内容：**
- `MeanAggregator` - 简单均值基线
- `MedianAggregator` - 简单中位数基线
- `MoMAggregator` - Median-of-Means（用于消融对比）

### 4. macro/__init__.py 更新 ✓

**修改：** 从 factory 模式改为直接导出具体聚合器类。

```python
from .dca_aggregator import DensityCentroidAggregator
from .scp_aggregator import SkewnessCorrectedPeakAggregator
from .simple_aggregators import MeanAggregator, MedianAggregator, MoMAggregator
```

### 5. legacy/holiday_conditioning/ 目录创建 ✓

**目的：** 保存 holiday 条件化相关的实验记录和文档。

**包含内容：**
- `README.md` - 详细的备份说明文档
- 说明为什么移除 holiday_flag
- 记录实验结论
- 提供恢复方法（如需要）

---

## 📋 待处理的文件

### micro/sfcn.py - 需要清理

**当前状态：** 包含大量 holiday 条件化参数和逻辑

**需要移除的参数：**
```python
holiday_enable: bool = False
holiday_mode: str = 'none'
holiday_alpha_delta: float = 0.05
holiday_hard_alpha: float = 0.10
holiday_dropout_prob: float = 0.0
holiday_dual_bank_enable: bool = False
```

**需要移除的方法：**
- `_normalize_holiday_flag()`
- `_apply_holiday_dropout()`
- `_compute_effective_alpha()` 中的 holiday 分支
- `_compute_dual_bank_field_stats()` 中的 holiday 分支

**需要简化的方法签名：**
```python
# 移除 holiday_flag 参数
def encode_training_future(self, x_future, x_history, use_local_scaling)
def encode_inference_history(self, x_history, use_local_scaling)
def decode_prediction(self, pred, stats, use_local_scaling)
```

**论文终版保留：**
- 基础 SFCN 逻辑（场耦合归一化）
- 可学习的 coupling 参数 α
- 场统计量计算
- 编码/解码方法

### data_provider/fujian30_loader.py - 保持不变

**状态：** ✅ 保留不变

**原因：** 数据加载器中的 `holiday_flag` 用于：
- 数据集划分
- 评估时计算节假日子集指标（Hol-MAE, HDR, PTE）
- 不作为模型输入，仅用于后处理分析

---

## 🎯 论文终版主路径（已固化）

```
LSTDE (Fixed FFT, K=4, patch_len=12)
→ Trend-Aware PatchEmbed (concat)
→ SFCN (learnable α, 无 holiday 分支)
→ Diffusion Denoiser
→ Hybrid Residual (η=0.5)
→ DCA Aggregator
```

---

## 📊 代码统计

| 模块 | 清理前 | 清理后 | 减少 |
|------|--------|--------|------|
| `HoliDiff.py` | ~350行 | ~200行 | **-150行** |
| `exp_long_term_forecasting.py` | 包含 holiday_flag | 已移除 | **-3行** |
| `macro/` 模块 | 分散在多处 | 清晰分层 | 结构优化 |
| 动态分支 | 5+ 个聚合模式 | 1 个主路径 | **-4 个分支** |

---

## 🔄 下一步工作

### 高优先级

1. **清理 `micro/sfcn.py`**
   - 移除所有 holiday 相关参数和方法
   - 简化方法签名
   - 保留核心 SFCN 逻辑

2. **清理 `frequency/` 目录**
   - 移除未使用的模块：`conditioner.py`, `modulator.py`, `fusion.py`, `spectral_gate.py`
   - 保留核心模块：`decompose.py`, `patch_embed.py`, `residual.py`

3. **更新配置文件**
   - 创建 `configs/paper/fujian30_holidiff.yaml`
   - 固化论文终版配置
   - 移除实验性参数

### 中优先级

4. **测试清理后的代码**
   - 运行快速训练测试
   - 确保模型可以正常训练和推理
   - 验证评估指标计算正确

5. **更新文档**
   - 更新 `README.md`
   - 创建复现指南
   - 更新 API 文档

---

## 💡 关键决策

### 为什么保留 `holiday_flag` 在数据层？

**决策：** `holiday_flag` 不作为模型 forward 的输入，但保留在数据加载和评估中。

**理由：**
1. 论文需要节假日细粒度评估（Hol-MAE, Hol-CG MAE, HDR, PTE）
2. 数据集需要区分常规日和节假日样本
3. 模型本身不依赖 holiday 标签，体现泛化能力

### 为什么创建 `simple_aggregators.py`？

**决策：** 将 Mean/Median/MoM 从主模型移到独立文件。

**理由：**
1. 这些聚合器不是论文主方法，只用于消融对比
2. 从主模型移除可减少代码复杂度
3. 保留在项目中是为了复现消融实验

---

## 📝 使用指南

### 主模型使用（论文终版）

```python
from Holidiff.HoliDiff import HATEK

model = HATEK(configs)

# 训练
outputs, weight = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

# 推理
macro_pred, all_samples = model(x_enc, x_mark_enc, x_dec, x_mark_dec, 
                                 sample_times=10)
```

### 消融实验使用

```python
from Holidiff.macro.simple_aggregators import MeanAggregator, MedianAggregator, MoMAggregator

# 替换聚合器进行消融实验
aggregators = {
    'DCA': DensityCentroidAggregator(configs),
    'Mean': MeanAggregator(),
    'Median': MedianAggregator(),
    'MoM': MoMAggregator(n_blocks=5, rmom_n=3)
}
```

### 评估时使用 holiday_flag

```python
# 数据加载时获取 holiday 标签
holiday_flags = dataset.holiday_flags

# 模型预测（不使用 holiday_flag）
predictions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

# 评估时按 holiday 划分
holiday_mask = (holiday_flags == 1)
regular_mask = (holiday_flags == 0)

hol_mae = mae[holiday_mask].mean()
reg_mae = mae[regular_mask].mean()
```

---

**清理完成时间**: 2026-05-27  
**状态**: 主路径已清理，SFCN 待处理  
**下一步**: 清理 `micro/sfcn.py` 和 `frequency/` 目录

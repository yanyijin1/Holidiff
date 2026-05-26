# Holiday Conditioning Legacy Code

> **备份日期**: 2026-05-27  
> **原因**: 论文终版主模型不再使用 `holiday_flag` 作为模型输入

## 背景

在早期实验中，我们尝试将 `holiday_flag` 作为模型的显式输入，用于节假日条件化。经过多轮实验验证，论文终版决定：

- **模型层面**: 不使用 `holiday_flag` 作为输入，体现模型的泛化能力
- **数据层面**: 保留 `holiday_flag` 用于数据集划分和评估指标计算
- **评估层面**: 使用 `holiday_flag` 计算节假日细粒度指标（Hol-MAE, HDR, PTE等）

## 备份的代码

### 1. 实验脚本中的 holiday_flag 调用

**位置**: `Holidiff/exp/exp_long_term_forecasting.py`

**原始代码**:
```python
outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 
                       sample_times=self.args.sample_times, 
                       holiday_flag=batch_holiday)  # ← 已移除
```

**清理后**:
```python
outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 
                       sample_times=self.args.sample_times)
```

### 2. SFCN 模块中的 holiday_flag 参数

**位置**: `Holidiff/micro/sfcn.py`

**原始签名**:
```python
def encode_training_future(self, x_future, x_history, use_local_scaling, 
                          holiday_flag=None):  # ← 已移除
```

**清理后**:
```python
def encode_training_future(self, x_future, x_history, use_local_scaling):
```

**相关参数**:
- `holiday_enable: bool = False`
- `holiday_mode: str = 'none'`
- `holiday_alpha_delta: float = 0.05`
- `holiday_hard_alpha: float = 0.10`
- `holiday_dropout_prob: float = 0.0`
- `holiday_dual_bank_enable: bool = False`

这些参数在论文终版中不再使用，已从主路径移除。

### 3. 数据加载器（保留）

**位置**: `Holidiff/data_provider/fujian30_loader.py`

**状态**: **保留不变**

**原因**: 数据加载器中的 `holiday_flag` 用于：
- 数据集划分（训练集/验证集/测试集）
- 评估时计算节假日子集指标
- 不作为模型输入，仅用于后处理分析

## 实验记录

### Holiday Conditioning 实验阶段

在 `Holidiff/logs/holiday_conditioning/` 目录下保存了完整的实验日志：

- `hc_stage0_erc_full.log` - 基线实验
- `hc_stage1_hard_full.log` - 硬编码 holiday alpha
- `hc_stage1_learn_full.log` - 可学习 holiday alpha
- `hc_stage2_cd01_full.log` - 条件dropout实验
- `hc_stage3_haccd_drfb_full.log` - 完整holiday条件化

### 实验结论

经过系统性消融实验，我们发现：

1. **Holiday条件化未带来显著增益**: 在Fujian-30数据集上，显式注入holiday_flag并未显著改善节假日预测性能
2. **模型已具备隐式泛化能力**: 通过LSTDE、SFCN等模块，模型能够从历史数据中隐式学习节假日模式
3. **简化模型更优**: 移除holiday条件化后，模型更简洁，泛化性能更好

## 论文终版配置

**主模型路径**（无holiday条件化）:

```
LSTDE (Fixed FFT, K=4)
→ Trend-Aware PatchEmbed (concat)
→ SFCN (learnable α, 无holiday分支)
→ Diffusion Denoiser
→ Hybrid Residual (η=0.5)
→ DCA Aggregator
```

**评估流程**（使用holiday_flag）:

```python
# 数据加载时获取holiday标签
holiday_flags = dataset.holiday_flags

# 模型预测（不使用holiday_flag）
predictions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

# 评估时按holiday划分
holiday_mask = (holiday_flags == 1)
regular_mask = (holiday_flags == 0)

hol_mae = mae[holiday_mask].mean()
reg_mae = mae[regular_mask].mean()
```

## 如何恢复 Holiday Conditioning

如果需要恢复holiday条件化功能（例如用于对比实验），可以：

1. 参考本目录下的备份代码
2. 查看 `Holidiff/logs/holiday_conditioning/` 下的实验配置
3. 恢复 `SFCN` 中的 `holiday_enable` 参数
4. 在实验脚本中传入 `holiday_flag` 参数

## 相关文档

- [Holiday Conditioning 实验计划](../../docs/HoliDiff_Holiday_Conditioning_Experiment_Plan.md)
- [代码重构总结](../../docs/code_refactoring_summary.md)
- [论文终版模型配置](../../docs/current_model_for_paper.md)

---

**备份完成**: 2026-05-27  
**清理状态**: 主路径已清理，legacy代码已备份

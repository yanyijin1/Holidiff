# HoliDiff Macro Diagnosis Result (Masked)

## 1. 运行元信息

| field | value |
| --- | --- |
| config_path | /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_30epoch_base.yaml |
| checkpoint_path | .../checkpoints/long_term_forecast_Fujian30_96_12_HATEK_fujian30_ftM_sl96_ll48_pl12_dm128_nh8_el1_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_ExpBase30_0/checkpoint.pth |
| split | test |
| mode | standard (mask enabled) |
| model | HATEK |
| data | fujian30 |
| is_diff | True |
| rmom | 5 |
| n_blocks | 5 |
| sample_times_train | 20 |
| sample_times_eval | 10 |
| batch_y_mark_all_zero | True（mark 全零，holiday 信息未注入模型） |

---

## 2. 运行概览

| metric | value | 说明 |
| --- | --- | --- |
| mae_raw | 22.753 | 全量（含零值） |
| mae_raw_masked | 22.577 | 剔除 true=0 后 |
| rmse_raw | 35.488 | 全量 |
| zero_ratio_true_raw | 0.00258 | 真值中零值占比（约 0.26%） |
| negative_ratio_raw | 9.0e-5 | 预测负值占比（极低，可忽略） |
| sample_count | 4137 | |
| node_count | 30 | |
| seq_len / pred_len | 96 / 12 | 15min 间隔，预测 3h |

masked 与 raw 的 MAE 差距仅 0.18，说明零值对整体指标影响很小，mask 逻辑正确。

---

## 3. 训练过程摘要（来自 logs/fujian30_base_30e_mask.log）

| 阶段 | epoch | val_loss | 备注 |
| --- | --- | --- | --- |
| 初始收敛 | 1→7 | 0.2308 → 0.2211 | 每 epoch 均下降 |
| 平台期开始 | 8 | 0.2212 | EarlyStopping counter 1/10 |
| 最佳 checkpoint | 22 | **0.2206** | 保存 |
| 最终 epoch | 30 | 0.2215 | counter 8/10，未触发 early stop |
| 测试集 | — | mae=0.2290, rmse=0.3196 | scaled 空间 |

**关键观察**：epoch 7 之后 lr 已衰减至 1.5e-9 量级，val_loss 在 0.2206~0.2220 之间震荡，模型实际上在 epoch 22 就已收敛，后续 8 个 epoch 几乎无收益。lr schedule (type1, 每 epoch 减半) 衰减过快，导致后期学习率趋近于零。

---

## 4. 样本级汇总指标

| analysis | metric | raw | masked |
| --- | --- | --- | --- |
| overall | mae_mean | 22.753 | 22.574 |
| overall | rmse_mean | 32.312 | 32.025 |
| holiday | mae_mean | 25.235 | 25.145 |
| normal | mae_mean | 21.717 | 21.501 |
| free_flow | mae_mean | 14.211 | 13.890 |
| transition | mae_mean | 22.716 | 22.581 |
| congested | mae_mean | 31.075 | 30.992 |

holiday vs normal MAE 差距：**+3.52**（masked），节假日误差显著更高（MWU p≈0）。

---

## 5. 统计检验

| test | name | statistic | pvalue |
| --- | --- | --- | --- |
| spearman | intra_vs_mae | 0.143 | 2.2e-20 |
| spearman | intra_vs_mae_masked | 0.141 | 8.8e-20 |
| spearman | inter_vs_mae | 0.142 | 3.5e-20 |
| spearman | inter_vs_mae_masked | 0.140 | 1.4e-19 |
| spearman | median_bias_vs_mae | 0.143 | 3.0e-20 |
| spearman | inter_vs_median_bias | **0.989** | 0.0 |
| mannwhitneyu | holiday_intra_gt_normal | 2668279 | 5.1e-143 |
| mannwhitneyu | holiday_inter_gt_normal | 2668174 | 5.5e-143 |
| mannwhitneyu | congested_mae_gt_free | 1871647 | 0.0 |

inter_dev 与 median_bias 的 Spearman 相关高达 **0.989**，说明两者几乎等价，可以合并为一个指标。

---

## 6. 节点级高误差 Top-10

| rank | station_index | node_mae_raw | node_mae_masked | zero_ratio | intra_var | inter_dev |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 44.79 | 44.46 | 0.0016 | 0.00485 | 0.00386 |
| 2 | 10 | 42.91 | 42.13 | 0.0035 | 0.00452 | 0.00362 |
| 3 | 12 | 41.97 | 41.61 | 0.0016 | 0.00456 | 0.00365 |
| 4 | 11 | 41.55 | 41.14 | 0.0018 | 0.00454 | 0.00363 |
| 5 | 13 | 41.51 | 40.51 | 0.0045 | 0.00495 | 0.00396 |
| 6 | 14 | 41.31 | 41.01 | 0.0016 | 0.00493 | 0.00395 |
| 7 | 17 | 32.81 | 32.47 | 0.0025 | 0.00545 | 0.00435 |
| 8 | 16 | 32.16 | 31.69 | 0.0028 | 0.00491 | 0.00392 |
| 9 | 19 | 26.23 | 26.06 | 0.0021 | 0.00546 | 0.00437 |
| 10 | 24 | 25.74 | 25.74 | 0.0000 | 0.00459 | 0.00365 |

station 10~15 是连续的高误差节点群，可能是路网中某段拥堵瓶颈区域。

---

## 7. 预测步长误差（Top-5 最难步）

| rank | timestep | minutes_ahead | mae_raw | mae_masked | intra_var | inter_dev |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 12 | 180 min | 23.993 | 23.726 | 0.004816 | 0.003847 |
| 2 | 11 | 165 min | 23.857 | 23.603 | 0.004791 | 0.003840 |
| 3 | 10 | 150 min | 23.689 | 23.448 | 0.004790 | 0.003834 |
| 4 | 9 | 135 min | 23.521 | 23.295 | 0.004776 | 0.003823 |
| 5 | 8 | 120 min | 23.374 | 23.165 | 0.004801 | 0.003848 |

误差随预测步长单调递增，符合预期。步长 8→12 的 MAE 增量约 0.6，相对平缓。

---

## 8. 小时级 Holiday/Normal MAE 对比

| start_hour | normal | holiday | 差值 |
| --- | --- | --- | --- |
| 0–5 | 10.7–18.5 | 10.8–18.9 | ≈0（夜间差异小） |
| 6 | 20.5 | 25.0 | **+4.5** |
| 7 | 23.6 | 33.4 | **+9.7** |
| 8 | 26.2 | 39.2 | **+13.0** |
| 9 | 27.3 | 38.9 | **+11.5** |
| 10–18 | 24–30 | 32–40 | **+6~10** |
| 19–23 | 12–24 | 12–23 | ≈0（晚间差异收窄） |

节假日早高峰（7:00–9:00）误差激增，是模型最薄弱的时段。

---

## 9. 图像索引

| 文件 | 说明 |
|---|---|
| `step1_intra_var_boxplot.png` | 节假日/非节假日组内方差箱线图 |
| `step1_intra_var_vs_mae.png` | 组内方差与样本级 MAE 散点图 |
| `step1_phase_mae_boxplot.png` | 不同流相态样本 MAE 箱线图 |
| `step2_inter_dev_boxplot.png` | 节假日/非节假日组间离散度箱线图 |
| `step2_inter_dev_vs_median_bias.png` | 组间离散度与中位数偏移散点图 |
| `step2_median_bias_group_compare.png` | 高/低组间离散样本的中位数偏移对比图 |
| `step3_timestep_mae_curve.png` | 预测 horizon 上的 MAE 曲线 |
| `step3_hourly_mae_curve.png` | 小时级 Holiday/Normal MAE 曲线 |
| `step3_node_mae_top10.png` | 节点级 MAE Top-10 柱状图 |
| `step3_inter_dev_heatmap.png` | 节点 × 预测步组间离散度热力图 |
| `step3_intra_var_heatmap.png` | 节点 × 预测步组内方差热力图 |
| `step3_overlay_sample*_combined.png` | worst-case 样本多站点拼接图（真实时间戳，history-future 连续） |

图像目录：`diagnosis/macro/figures/masked`

---

## 10. MoM（Mixture of Moments）问题诊断

### 10.1 当前 MoM 实现流程

```
forward_consensus_inference
  └─ 循环 sample_times=10 次 DPM 采样 → all_outs [10, B, T, N]
       └─ extract_consensus(all_outs)
            └─ 循环 rmom_n=5 次：
                 每次 _consensus_reduce(随机打乱的 all_outs)
                   └─ 分 n_blocks=5 组 → 每组取均值 → 取 median
            └─ 对 5 次结果再取均值
```

### 10.2 已识别的问题

**问题 1：block 划分与 shuffle 的冗余性**

`_consensus_reduce` 先对 10 个样本随机打乱，再分 5 块（每块 2 个），每块取均值后取 median。但 `extract_consensus` 外层又重复 5 次这个过程再取均值。实际上 10 个样本分 5 块每块 2 个，median 退化为对 5 个两两均值取中位数，方差压缩效果有限，且计算量是简单均值的 5×5=25 倍。

**问题 2：block_var 诊断量计算位置错误**

```python
# HoliDiff.py _consensus_reduce
block_var_scalars.append(block_var.mean(dim=[1, 2]))  # shape: (B,)
...
self._diag_block_var.append(block_var_scalar_mean.detach().cpu())  # shape: (B,)
```

但在 `extract_consensus` 中聚合时：

```python
self._diag_block_var.append(torch.stack(recent_block_var, dim=0).mean(dim=0))
```

`recent_block_var` 是 rmom_n=5 次调用各自 append 的列表，每个元素 shape=(B,)，stack 后 mean 是正确的。但 `_consensus_reduce` 内部 `block_var_scalar_mean` 是对 n_blocks 个 scalar 取均值，而 `block_var_map_mean` 是对 n_blocks 个 map 取均值——两者维度不一致，后续 `_diag_block_var_map` 的 shape 是 `(B, T, N)` 而 `_diag_block_var` 是 `(B,)`，在 `step1_collect_diagnostics.py` 中 `fit_len` / `fit_map` 分别处理，逻辑上没有崩溃，但 scalar 诊断量丢失了空间分布信息。

**问题 3：median 在 n_blocks=5 时的偏差**

当 `n_blocks=5`（奇数），`torch.median` 取第 3 小的值，在分布对称时无偏，但当 block 均值分布偏斜（如拥堵时流量分布右偏）时，median 会系统性低估均值，导致 `median_bias` 偏高。这与诊断数据中 `inter_vs_median_bias` Spearman=0.989 吻合——两者几乎完全共线，说明 median_bias 没有提供独立信息。

**问题 4：holiday mark 全零，MoM 无法利用节假日信息**

```
batch_y_mark_all_zero: True
```

`x_mark_enc` 传入 TEK/STEKBackbone 但实际全为零，模型无法区分节假日与工作日。这是节假日误差显著更高（+3.5 MAE）的根本原因，与 MoM 本身无关，但会放大 MoM 的不确定性估计误差。

**问题 5：STEKBackbone U-Net skip connection 未正确使用**

```python
# stek_backbone.py forward
for attention_layer, mlp, dropout_layer, norm in zip(
        self.Attentions_over_token_up, self.Attentions_mlp,
        self.Attentions_dropout, self.Attentions_norm):   # ← dropout/norm 用的是 down 的
    prev = skip.pop()
    outputs = dropout_layer(mlp(torch.cat((prev, inputs), dim=-1)))
    outputs = norm(outputs.reshape(b * c, t, -1)).reshape(b, c, t, -1)
    output = attention_layer(inputs)          # ← outputs 被丢弃，直接用 inputs
    inputs = dropout_layer(output)            # ← skip 融合结果 outputs 完全没用上
```

`outputs`（融合了 skip 的结果）计算后被丢弃，`attention_layer` 仍然接收原始 `inputs`。U-Net 的 skip connection 形同虚设，上采样路径退化为与下采样路径相同的结构。

### 10.3 修改建议（优先级排序）

| 优先级 | 问题 | 建议修改 |
| --- | --- | --- |
| P0 | holiday mark 全零 | 修复 `Fujian30CsvDataset` 的 mark 构造，将 is_holiday/hour 编码写入 `x_mark` |
| P0 | STEKBackbone skip 未使用 | 将 `output = attention_layer(inputs)` 改为 `output = attention_layer(outputs)`，让上采样路径接收融合后的特征 |
| P1 | MoM median 偏差 | 将 `_consensus_reduce` 中的 `torch.median` 改为加权均值，或将 n_blocks 改为偶数（如 4/6）以减少 median 偏差 |
| P1 | MoM 计算冗余 | 考虑将 rmom_n × n_blocks 的双重循环简化：直接对 sample_times 个样本做一次 bootstrap 聚合，减少推理时间 |
| P2 | inter_dev 与 median_bias 共线 | 去掉 median_bias 诊断量，或改用 `std(block_means)` 替代 inter_dev，提供更直观的不确定性度量 |
| P2 | lr schedule 衰减过快 | 将 `lradj=type1`（每 epoch 减半）改为 cosine annealing 或 warmup+cosine，避免后期 lr 趋零 |

---

## 11. 路径索引

| 类型 | 路径 |
|---|---|
| 图像目录 | `diagnosis/macro/figures/masked` |
| 输出数据目录 | `diagnosis/macro/output/masked` |
| 训练日志 | `logs/fujian30_base_30e_mask.log` |
| 模型主文件 | `Holidiff/HoliDiff.py` |
| Backbone | `Holidiff/micro/stek_backbone.py` |
| 数据加载 | `Holidiff/data_provider/fujian30_loader.py` |

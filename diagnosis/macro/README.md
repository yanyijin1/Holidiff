# HoliDiff Macro Diagnosis

本目录用于按照 `docs/SimDiff_MoM_Flow_Diagnosis_Guide.md` 对 `Holidiff` 的 macro/MoM 聚合过程做诊断。

## 目录说明

- `scripts/`: 结果生成脚本，按 `step1/2/3` 命名
- `plots/`: 绘图脚本，按 `step1/2/3` 命名
- `output/`: 存放诊断生成的 CSV、NPY、Markdown 中间结果
- `figures/`: 存放绘制出的 PNG 图
- `result.md`: 客观结果整理文档

## 脚本说明

### scripts

- `scripts/step1_collect_diagnostics.py`
  - 加载 HoliDiff 配置与 checkpoint
  - 在测试集上运行推理
  - 读取模型内的诊断缓存
  - 生成样本级、节点级、时间步级、小时级结果 CSV
  - 输出热力图所需矩阵 CSV/NPY

- `scripts/step2_summarize_statistics.py`
  - 对 `step1` 输出做汇总与统计检验
  - 生成显著性检验表、Top 节点表、Top 时间步表、小时对比表

- `scripts/step3_generate_result_md.py`
  - 将关键结果表整理为 `result.md`
  - 不做主观解释，只做客观表格汇总和图像索引

### plots

- `plots/step1_plot_sample_level.py`
  - 生成样本级图：组内方差箱线图、方差-MAE 散点图、流相态 MAE 箱线图

- `plots/step2_plot_aggregation_level.py`
  - 生成聚合级图：组间离散度箱线图、组间离散度-中位数偏移散点图、高低离散度偏移对比图

- `plots/step3_plot_error_maps.py`
  - 生成误差与热力图：预测 horizon MAE 曲线、小时级 MAE 曲线、节点 Top-10、时空热力图

## 输出文件说明

### output 中的主要文件

- `step1_sample_metrics.csv`: 每个测试样本的误差、节假日标签、流相态、组内方差、组间离散度、中位数偏移
- `step1_node_metrics.csv`: 每个节点的平均误差与平均诊断量
- `step1_timestep_metrics.csv`: 每个预测步长的平均误差与平均诊断量
- `step1_hourly_metrics.csv`: 按 `is_holiday × start_hour` 聚合的小时级结果
- `step1_node_timestep_mae_raw.csv`: 节点 × 预测步 的 MAE 矩阵
- `step1_node_timestep_inter_dev.csv`: 节点 × 预测步 的组间离散度矩阵
- `step1_node_timestep_intra_var.csv`: 节点 × 预测步 的组内方差矩阵
- `step2_summary_metrics.csv`: 汇总后的整体指标表
- `step2_stat_tests.csv`: 统计检验结果表
- `step2_top10_nodes.csv`: 高误差节点 Top-10
- `step2_hardest_timesteps.csv`: 高误差预测步 Top 列表
- `step2_hourly_compare.csv`: 小时级节假日/常规日对比表

## 使用顺序

### 1. 生成诊断结果

```bash
python /root/yanyijin/STdiff/diagnosis/macro/scripts/step1_collect_diagnostics.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml
```

如需指定 checkpoint：

```bash
python /root/yanyijin/STdiff/diagnosis/macro/scripts/step1_collect_diagnostics.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --checkpoint /path/to/checkpoint.pth
```

### 2. 生成统计摘要

```bash
python /root/yanyijin/STdiff/diagnosis/macro/scripts/step2_summarize_statistics.py
```

### 3. 绘图

```bash
python /root/yanyijin/STdiff/diagnosis/macro/plots/step1_plot_sample_level.py
python /root/yanyijin/STdiff/diagnosis/macro/plots/step2_plot_aggregation_level.py
python /root/yanyijin/STdiff/diagnosis/macro/plots/step3_plot_error_maps.py
```

### 4. 生成结果 Markdown

```bash
python /root/yanyijin/STdiff/diagnosis/macro/scripts/step3_generate_result_md.py
```

## 依赖说明

建议环境中具备：

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `scipy`（若没有，统计检验结果会输出为 NaN）

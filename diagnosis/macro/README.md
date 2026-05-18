# HoliDiff Macro Diagnosis

这个目录只做一件事：
对 `Holidiff` 的 macro / MoM 聚合做诊断，并把结果整理成可复用的表和图。

## 目录

- `scripts/`
  - `step1_collect_diagnostics.py`：跑推理，导出原始诊断结果
  - `step2_summarize_statistics.py`：做汇总统计和检验
  - `step3_generate_result_md.py`：生成客观结果文档 `result.md`
- `plots/`
  - `step1_plot_sample_level.py`：样本级图
  - `step2_plot_aggregation_level.py`：聚合级图
  - `step3_plot_error_maps.py`：误差图、热力图、worst-case 时序对比图
- `output/`：CSV / NPY / JSON 输出
- `figures/`：PNG / PDF 图像输出
- `result.md`：结果陈列文档

## 关键说明

当前 `Fujian30CsvDataset` 里的 `x_mark / y_mark` 是全零占位，
所以这里的：

- `is_holiday`
- `start_hour`

不是从 `batch_y_mark` 直接取的，而是根据原始 CSV 时间轴和窗口索引恢复的。

## step1 主要输出

- `step1_sample_metrics.csv`
- `step1_node_metrics.csv`
- `step1_timestep_metrics.csv`
- `step1_hourly_metrics.csv`
- `step1_node_timestep_mae_raw.csv`
- `step1_node_timestep_inter_dev.csv`
- `step1_node_timestep_intra_var.csv`
- `step1_worst_case_overlays.csv`
- `step1_overview.csv`
- `step1_run_meta.json`

其中 `step1_worst_case_overlays.csv` 用于画最直观的预测对比图：

- 前半段蓝线：历史真实输入
- 后半段蓝虚线：未来真实值
- 后半段红线：未来预测值

## 推荐执行顺序

### 1. 导出诊断结果

```bash
python /root/yanyijin/STdiff/diagnosis/macro/scripts/step1_collect_diagnostics.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_30epoch_base.yaml \
  --output_dir /root/yanyijin/STdiff/diagnosis/macro/output/base
```

### 2. 汇总统计

```bash
python /root/yanyijin/STdiff/diagnosis/macro/scripts/step2_summarize_statistics.py \
  --input_dir /root/yanyijin/STdiff/diagnosis/macro/output/base \
  --output_dir /root/yanyijin/STdiff/diagnosis/macro/output/base
```

### 3. 生成图像

```bash
python /root/yanyijin/STdiff/diagnosis/macro/plots/step1_plot_sample_level.py \
  --input_dir /root/yanyijin/STdiff/diagnosis/macro/output/base \
  --figures_dir /root/yanyijin/STdiff/diagnosis/macro/figures/base

python /root/yanyijin/STdiff/diagnosis/macro/plots/step2_plot_aggregation_level.py \
  --input_dir /root/yanyijin/STdiff/diagnosis/macro/output/base \
  --figures_dir /root/yanyijin/STdiff/diagnosis/macro/figures/base

python /root/yanyijin/STdiff/diagnosis/macro/plots/step3_plot_error_maps.py \
  --input_dir /root/yanyijin/STdiff/diagnosis/macro/output/base \
  --figures_dir /root/yanyijin/STdiff/diagnosis/macro/figures/base
```

### 4. 生成结果文档

```bash
python /root/yanyijin/STdiff/diagnosis/macro/scripts/step3_generate_result_md.py \
  --input_dir /root/yanyijin/STdiff/diagnosis/macro/output/base \
  --figures_dir /root/yanyijin/STdiff/diagnosis/macro/figures/base \
  --output_md /root/yanyijin/STdiff/diagnosis/macro/result.md
```

## 双配置建议

如果你同时比较 `base` 和 `doc` 两套配置，建议分别输出到：

- `output/base`, `figures/base`
- `output/doc`, `figures/doc`

避免覆盖。

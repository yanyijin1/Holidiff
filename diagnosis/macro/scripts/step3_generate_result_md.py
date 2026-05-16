import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def to_md(df):
    return df.to_markdown(index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'output'))
    p.add_argument('--figures_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'figures'))
    p.add_argument('--output_md', default=str(ROOT / 'diagnosis' / 'macro' / 'result.md'))
    c = p.parse_args()

    in_dir = Path(c.input_dir)
    fig_dir = Path(c.figures_dir)
    out_md = Path(c.output_md)

    overview = pd.read_csv(in_dir / 'step1_overview.csv')
    summary = pd.read_csv(in_dir / 'step2_summary_metrics.csv')
    stats = pd.read_csv(in_dir / 'step2_stat_tests.csv')
    top_nodes = pd.read_csv(in_dir / 'step2_top10_nodes.csv')
    hard_steps = pd.read_csv(in_dir / 'step2_hardest_timesteps.csv')
    hourly = pd.read_csv(in_dir / 'step2_hourly_compare.csv')

    text = f'''# HoliDiff Macro Diagnosis Result

## 1. 运行概览

{to_md(overview)}

## 2. 样本级汇总指标

{to_md(summary)}

## 3. 统计检验

{to_md(stats)}

## 4. 节点级高误差 Top-10

{to_md(top_nodes)}

## 5. 预测步长高误差 Top 列表

{to_md(hard_steps)}

## 6. 小时级 Holiday/Normal 对比

{to_md(hourly.head(24))}

## 7. 图像索引

| 文件 | 说明 |
|---|---|
| `step1_intra_var_boxplot.png` | 节假日/非节假日的组内方差箱线图 |
| `step1_intra_var_vs_mae.png` | 组内方差与样本级 MAE 散点图 |
| `step2_inter_dev_boxplot.png` | 节假日/非节假日的组间离散度箱线图 |
| `step2_inter_dev_vs_median_bias.png` | 组间离散度与中位数偏移散点图 |
| `step3_timestep_mae_curve.png` | 预测 horizon 上的 MAE 曲线 |
| `step3_hourly_mae_curve.png` | 小时级 Holiday/Normal MAE 曲线 |
| `step3_node_mae_top10.png` | 节点级 MAE Top-10 柱状图 |
| `step3_inter_dev_heatmap.png` | 节点 × 预测步的组间离散度热力图 |
| `step3_intra_var_heatmap.png` | 节点 × 预测步的组内方差热力图 |

## 8. 图像目录

当前图像目录：`{fig_dir}`
'''
    out_md.write_text(text, encoding='utf-8')
    print(f'Wrote markdown report to {out_md}')


if __name__ == '__main__':
    main()

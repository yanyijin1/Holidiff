import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def to_md(df):
    if df.empty:
        return '暂无数据。'
    headers = [str(x) for x in df.columns.tolist()]
    rows = [[str(v) for v in row] for row in df.fillna('NaN').values.tolist()]
    sep = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    lines = ['| ' + ' | '.join(headers) + ' |', sep]
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join(lines)


def load_optional_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'output'))
    p.add_argument('--figures_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'figures'))
    p.add_argument('--output_md', default=str(ROOT / 'diagnosis' / 'macro' / 'result.md'))
    c = p.parse_args()

    in_dir = Path(c.input_dir)
    fig_dir = Path(c.figures_dir)
    out_md = Path(c.output_md)

    overview = load_optional_csv(in_dir / 'step1_overview.csv')
    summary = load_optional_csv(in_dir / 'step2_summary_metrics.csv')
    stats = load_optional_csv(in_dir / 'step2_stat_tests.csv')
    top_nodes = load_optional_csv(in_dir / 'step2_top10_nodes.csv')
    hard_steps = load_optional_csv(in_dir / 'step2_hardest_timesteps.csv')
    hourly = load_optional_csv(in_dir / 'step2_hourly_compare.csv')
    baseline = load_optional_csv(in_dir / 'step1_baseline_metrics.csv')
    meta = json.loads((in_dir / 'step1_run_meta.json').read_text(encoding='utf-8')) if (in_dir / 'step1_run_meta.json').exists() else {}

    sections = ['# HoliDiff Macro Diagnosis Result', '']
    sections.append('## 1. 运行元信息')
    sections.append('')
    if meta:
        meta_df = pd.DataFrame({'field': list(meta.keys()), 'value': [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for v in meta.values()]})
        sections.append(to_md(meta_df))
    else:
        sections.append('暂无运行元信息。')
    sections.append('')

    sections.append('## 2. 运行概览')
    sections.append('')
    sections.append(to_md(overview) if len(overview) else '暂无 `step1_overview.csv`。')
    sections.append('')

    sections.append('## 3. 基线测试日志摘录')
    sections.append('')
    sections.append(to_md(baseline) if len(baseline) else '暂无 `step1_baseline_metrics.csv`。')
    sections.append('')

    sections.append('## 4. 样本级汇总指标')
    sections.append('')
    sections.append(to_md(summary) if len(summary) else '暂无 `step2_summary_metrics.csv`。')
    sections.append('')

    sections.append('## 5. 统计检验')
    sections.append('')
    sections.append(to_md(stats) if len(stats) else '暂无 `step2_stat_tests.csv`。')
    sections.append('')

    sections.append('## 6. 节点级高误差 Top-10')
    sections.append('')
    sections.append(to_md(top_nodes) if len(top_nodes) else '暂无 `step2_top10_nodes.csv`。')
    sections.append('')

    sections.append('## 7. 预测步长高误差 Top 列表')
    sections.append('')
    sections.append(to_md(hard_steps) if len(hard_steps) else '暂无 `step2_hardest_timesteps.csv`。')
    sections.append('')

    sections.append('## 8. 小时级 Holiday/Normal 对比')
    sections.append('')
    sections.append(to_md(hourly.head(24)) if len(hourly) else '暂无 `step2_hourly_compare.csv`。')
    sections.append('')

    sections.append('## 9. 图像索引')
    sections.append('')
    sections.append('| 文件 | 说明 |')
    sections.append('|---|---|')
    sections.append('| `step1_intra_var_boxplot.png` | 节假日/非节假日组内方差箱线图 |')
    sections.append('| `step1_intra_var_vs_mae.png` | 组内方差与样本级 MAE 散点图 |')
    sections.append('| `step1_phase_mae_boxplot.png` | 不同流相态样本 MAE 箱线图 |')
    sections.append('| `step2_inter_dev_boxplot.png` | 节假日/非节假日组间离散度箱线图 |')
    sections.append('| `step2_inter_dev_vs_median_bias.png` | 组间离散度与中位数偏移散点图 |')
    sections.append('| `step2_median_bias_group_compare.png` | 高/低组间离散样本的中位数偏移对比图 |')
    sections.append('| `step3_timestep_mae_curve.png` | 预测 horizon 上的 MAE 曲线 |')
    sections.append('| `step3_hourly_mae_curve.png` | 小时级 Holiday/Normal MAE 曲线 |')
    sections.append('| `step3_node_mae_top10.png` | 节点级 MAE Top-10 柱状图 |')
    sections.append('| `step3_inter_dev_heatmap.png` | 节点 × 预测步组间离散度热力图 |')
    sections.append('| `step3_intra_var_heatmap.png` | 节点 × 预测步组内方差热力图 |')
    sections.append('')

    sections.append('## 10. 图像目录')
    sections.append('')
    sections.append(f'当前图像目录：`{fig_dir}`')
    sections.append('')
    sections.append('## 11. 结果目录')
    sections.append('')
    sections.append(f'当前结果目录：`{in_dir}`')

    out_md.write_text('\n'.join(sections), encoding='utf-8')
    print(f'Wrote markdown report to {out_md}')


if __name__ == '__main__':
    main()

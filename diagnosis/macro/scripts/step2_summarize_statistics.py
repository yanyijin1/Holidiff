import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, spearmanr
except Exception:
    mannwhitneyu = None
    spearmanr = None

ROOT = Path(__file__).resolve().parents[3]


def safe_spearman(x, y):
    if spearmanr is None:
        return np.nan, np.nan
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def safe_mwu(x, y, alternative='two-sided'):
    if mannwhitneyu is None or len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    r = mannwhitneyu(x, y, alternative=alternative)
    return float(r.statistic), float(r.pvalue)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'output'))
    p.add_argument('--output_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'output'))
    c = p.parse_args()

    in_dir = Path(c.input_dir)
    out_dir = Path(c.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(in_dir / 'step1_sample_metrics.csv')
    node = pd.read_csv(in_dir / 'step1_node_metrics.csv')
    timestep = pd.read_csv(in_dir / 'step1_timestep_metrics.csv')
    hourly = pd.read_csv(in_dir / 'step1_hourly_metrics.csv')
    overview = pd.read_csv(in_dir / 'step1_overview.csv')
    baseline = pd.read_csv(in_dir / 'step1_baseline_metrics.csv') if (in_dir / 'step1_baseline_metrics.csv').exists() else None
    meta = json.loads((in_dir / 'step1_run_meta.json').read_text(encoding='utf-8')) if (in_dir / 'step1_run_meta.json').exists() else {}

    holiday = sample[sample['is_holiday'] == 1]
    normal = sample[sample['is_holiday'] == 0]
    free_flow = sample[sample['phase'] == 'free_flow']
    transition = sample[sample['phase'] == 'transition']
    congested = sample[sample['phase'] == 'congested']
    high_inter = sample[sample['inter_dev'] >= sample['inter_dev'].quantile(0.8)]
    low_inter = sample[sample['inter_dev'] < sample['inter_dev'].quantile(0.8)]

    summary_rows = [
        {'analysis': 'overall', 'metric': 'mae_raw_mean', 'value': float(sample['sample_mae_raw'].mean())},
        {'analysis': 'overall', 'metric': 'mae_raw_masked_mean', 'value': float(sample['sample_mae_raw_masked'].mean())},
        {'analysis': 'overall', 'metric': 'rmse_raw_mean', 'value': float(sample['sample_rmse_raw'].mean())},
        {'analysis': 'overall', 'metric': 'rmse_raw_masked_mean', 'value': float(sample['sample_rmse_raw_masked'].mean())},
        {'analysis': 'overall', 'metric': 'intra_var_mean', 'value': float(sample['intra_var'].mean())},
        {'analysis': 'overall', 'metric': 'inter_dev_mean', 'value': float(sample['inter_dev'].mean())},
        {'analysis': 'overall', 'metric': 'median_bias_mean', 'value': float(sample['median_bias'].mean())},
        {'analysis': 'overall', 'metric': 'zero_ratio_true_raw_mean', 'value': float(sample['zero_ratio_true_raw'].mean())},
        {'analysis': 'overall', 'metric': 'negative_ratio_raw_mean', 'value': float(sample['negative_ratio_raw'].mean())},
        {'analysis': 'holiday', 'metric': 'sample_count', 'value': int(len(holiday))},
        {'analysis': 'normal', 'metric': 'sample_count', 'value': int(len(normal))},
        {'analysis': 'holiday', 'metric': 'mae_raw_mean', 'value': float(holiday['sample_mae_raw'].mean()) if len(holiday) else np.nan},
        {'analysis': 'normal', 'metric': 'mae_raw_mean', 'value': float(normal['sample_mae_raw'].mean()) if len(normal) else np.nan},
        {'analysis': 'holiday', 'metric': 'mae_raw_masked_mean', 'value': float(holiday['sample_mae_raw_masked'].mean()) if len(holiday) else np.nan},
        {'analysis': 'normal', 'metric': 'mae_raw_masked_mean', 'value': float(normal['sample_mae_raw_masked'].mean()) if len(normal) else np.nan},
        {'analysis': 'holiday', 'metric': 'intra_var_mean', 'value': float(holiday['intra_var'].mean()) if len(holiday) else np.nan},
        {'analysis': 'normal', 'metric': 'intra_var_mean', 'value': float(normal['intra_var'].mean()) if len(normal) else np.nan},
        {'analysis': 'holiday', 'metric': 'inter_dev_mean', 'value': float(holiday['inter_dev'].mean()) if len(holiday) else np.nan},
        {'analysis': 'normal', 'metric': 'inter_dev_mean', 'value': float(normal['inter_dev'].mean()) if len(normal) else np.nan},
        {'analysis': 'high_inter', 'metric': 'median_bias_mean', 'value': float(high_inter['median_bias'].mean()) if len(high_inter) else np.nan},
        {'analysis': 'low_inter', 'metric': 'median_bias_mean', 'value': float(low_inter['median_bias'].mean()) if len(low_inter) else np.nan},
        {'analysis': 'free_flow', 'metric': 'mae_raw_mean', 'value': float(free_flow['sample_mae_raw'].mean()) if len(free_flow) else np.nan},
        {'analysis': 'transition', 'metric': 'mae_raw_mean', 'value': float(transition['sample_mae_raw'].mean()) if len(transition) else np.nan},
        {'analysis': 'congested', 'metric': 'mae_raw_mean', 'value': float(congested['sample_mae_raw'].mean()) if len(congested) else np.nan},
        {'analysis': 'free_flow', 'metric': 'mae_raw_masked_mean', 'value': float(free_flow['sample_mae_raw_masked'].mean()) if len(free_flow) else np.nan},
        {'analysis': 'transition', 'metric': 'mae_raw_masked_mean', 'value': float(transition['sample_mae_raw_masked'].mean()) if len(transition) else np.nan},
        {'analysis': 'congested', 'metric': 'mae_raw_masked_mean', 'value': float(congested['sample_mae_raw_masked'].mean()) if len(congested) else np.nan},
    ]
    if baseline is not None and len(baseline) > 0:
        for col in ['mae', 'mse', 'rmse']:
            if col in baseline.columns:
                summary_rows.append({'analysis': 'baseline_log', 'metric': col, 'value': float(baseline[col].iloc[-1])})
    pd.DataFrame(summary_rows).to_csv(out_dir / 'step2_summary_metrics.csv', index=False)

    stat_rows = []
    for x_col, y_col, name in [
        ('intra_var', 'sample_mae_raw', 'intra_vs_mae'),
        ('intra_var', 'sample_mae_raw_masked', 'intra_vs_mae_masked'),
        ('inter_dev', 'sample_mae_raw', 'inter_vs_mae'),
        ('inter_dev', 'sample_mae_raw_masked', 'inter_vs_mae_masked'),
        ('median_bias', 'sample_mae_raw', 'median_bias_vs_mae'),
        ('median_bias', 'sample_mae_raw_masked', 'median_bias_vs_mae_masked'),
        ('inter_dev', 'median_bias', 'inter_vs_median_bias'),
    ]:
        rho, pval = safe_spearman(sample[x_col], sample[y_col])
        stat_rows.append({'test': 'spearman', 'name': name, 'statistic': rho, 'pvalue': pval})

    u, pval = safe_mwu(holiday['intra_var'], normal['intra_var'], alternative='greater')
    stat_rows.append({'test': 'mannwhitneyu', 'name': 'holiday_intra_gt_normal', 'statistic': u, 'pvalue': pval})
    u, pval = safe_mwu(holiday['inter_dev'], normal['inter_dev'], alternative='greater')
    stat_rows.append({'test': 'mannwhitneyu', 'name': 'holiday_inter_gt_normal', 'statistic': u, 'pvalue': pval})
    u, pval = safe_mwu(congested['sample_mae_raw'], free_flow['sample_mae_raw'], alternative='greater')
    stat_rows.append({'test': 'mannwhitneyu', 'name': 'congested_mae_gt_free', 'statistic': u, 'pvalue': pval})
    pd.DataFrame(stat_rows).to_csv(out_dir / 'step2_stat_tests.csv', index=False)

    top_nodes = node.nlargest(10, 'node_mae_raw').copy()
    top_nodes['rank'] = np.arange(1, len(top_nodes) + 1)
    top_nodes.to_csv(out_dir / 'step2_top10_nodes.csv', index=False)

    hardest_steps = timestep.nlargest(min(5, len(timestep)), 'timestep_mae_raw').copy()
    hardest_steps['rank'] = np.arange(1, len(hardest_steps) + 1)
    hardest_steps.to_csv(out_dir / 'step2_hardest_timesteps.csv', index=False)

    hourly_pivot = hourly.pivot(index='start_hour', columns='is_holiday', values='sample_mae_raw').reset_index()
    hourly_pivot.columns = ['start_hour'] + [f'sample_mae_raw_holiday_{int(x)}' for x in hourly_pivot.columns[1:]]
    hourly_pivot.to_csv(out_dir / 'step2_hourly_compare.csv', index=False)

    overview.assign(source='step1_overview').to_csv(out_dir / 'step2_overview_echo.csv', index=False)
    pd.DataFrame([meta]).to_csv(out_dir / 'step2_run_meta_echo.csv', index=False)
    print(f'Saved step2 summaries to {out_dir}')


if __name__ == '__main__':
    main()

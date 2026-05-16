import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
plt.style.use('seaborn-v0_8-whitegrid')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'output'))
    p.add_argument('--figures_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'figures'))
    c = p.parse_args()

    in_dir = Path(c.input_dir)
    fig_dir = Path(c.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(in_dir / 'step1_sample_metrics.csv')

    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [sample.loc[sample['is_holiday'] == 0, 'inter_dev'], sample.loc[sample['is_holiday'] == 1, 'inter_dev']]
    ax.boxplot(groups, labels=['normal', 'holiday'])
    ax.set_title('Inter-group deviation by holiday flag')
    ax.set_ylabel('inter_dev')
    fig.tight_layout()
    fig.savefig(fig_dir / 'step2_inter_dev_boxplot.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sample['is_holiday'].map({0: '#4C78A8', 1: '#E45756'})
    ax.scatter(sample['inter_dev'], sample['median_bias'], s=18, c=colors, alpha=0.7)
    ax.set_xlabel('inter_dev')
    ax.set_ylabel('median_bias')
    ax.set_title('Inter-group deviation vs median bias')
    fig.tight_layout()
    fig.savefig(fig_dir / 'step2_inter_dev_vs_median_bias.png', dpi=200)
    plt.close(fig)

    cutoff = sample['inter_dev'].quantile(0.8)
    sample['inter_group'] = ['high_inter' if x >= cutoff else 'low_inter' for x in sample['inter_dev']]
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [sample.loc[sample['inter_group'] == 'low_inter', 'median_bias'], sample.loc[sample['inter_group'] == 'high_inter', 'median_bias']]
    ax.boxplot(vals, labels=['low_inter', 'high_inter'])
    ax.set_title('Median bias under low/high inter deviation')
    ax.set_ylabel('median_bias')
    fig.tight_layout()
    fig.savefig(fig_dir / 'step2_median_bias_group_compare.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()

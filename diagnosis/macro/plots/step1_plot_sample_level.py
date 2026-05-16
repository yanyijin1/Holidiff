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
    groups = [sample.loc[sample['is_holiday'] == 0, 'intra_var'], sample.loc[sample['is_holiday'] == 1, 'intra_var']]
    ax.boxplot(groups, labels=['normal', 'holiday'])
    ax.set_title('Intra-group variance by holiday flag')
    ax.set_ylabel('intra_var')
    fig.tight_layout()
    fig.savefig(fig_dir / 'step1_intra_var_boxplot.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sample['is_holiday'].map({0: '#4C78A8', 1: '#E45756'})
    ax.scatter(sample['intra_var'], sample['sample_mae_raw'], s=18, c=colors, alpha=0.7)
    ax.set_xlabel('intra_var')
    ax.set_ylabel('sample_mae_raw')
    ax.set_title('Intra-group variance vs sample MAE')
    fig.tight_layout()
    fig.savefig(fig_dir / 'step1_intra_var_vs_mae.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    phase_order = ['free_flow', 'transition', 'congested']
    phase_vals = [sample.loc[sample['phase'] == p, 'sample_mae_raw'] for p in phase_order]
    ax.boxplot(phase_vals, labels=phase_order)
    ax.set_title('Sample MAE by flow phase')
    ax.set_ylabel('sample_mae_raw')
    fig.tight_layout()
    fig.savefig(fig_dir / 'step1_phase_mae_boxplot.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()

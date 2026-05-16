import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    timestep = pd.read_csv(in_dir / 'step1_timestep_metrics.csv')
    hourly = pd.read_csv(in_dir / 'step1_hourly_metrics.csv')
    nodes = pd.read_csv(in_dir / 'step1_node_metrics.csv').nlargest(10, 'node_mae_raw')
    inter_heat = pd.read_csv(in_dir / 'step1_node_timestep_inter_dev.csv')
    intra_heat = pd.read_csv(in_dir / 'step1_node_timestep_intra_var.csv')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timestep['minutes_ahead'], timestep['timestep_mae_raw'], marker='o')
    ax.set_xlabel('minutes ahead')
    ax.set_ylabel('timestep_mae_raw')
    ax.set_title('MAE over prediction horizon')
    fig.tight_layout()
    fig.savefig(fig_dir / 'step3_timestep_mae_curve.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for flag, label, color in [(0, 'normal', '#4C78A8'), (1, 'holiday', '#E45756')]:
        sub = hourly[hourly['is_holiday'] == flag].sort_values('start_hour')
        if len(sub):
            ax.plot(sub['start_hour'], sub['sample_mae_raw'], marker='o', label=label, color=color)
    ax.set_xlabel('start hour')
    ax.set_ylabel('sample_mae_raw')
    ax.set_title('Hourly MAE by holiday flag')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / 'step3_hourly_mae_curve.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(nodes['station_index'].astype(str), nodes['node_mae_raw'])
    ax.set_xlabel('station_index')
    ax.set_ylabel('node_mae_raw')
    ax.set_title('Top-10 node MAE')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / 'step3_node_mae_top10.png', dpi=200)
    plt.close(fig)

    for data, name, title in [
        (inter_heat.to_numpy().T, 'step3_inter_dev_heatmap.png', 'Inter deviation heatmap'),
        (intra_heat.to_numpy().T, 'step3_intra_var_heatmap.png', 'Intra variance heatmap'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(data, aspect='auto', cmap='viridis')
        ax.set_xlabel('prediction timestep')
        ax.set_ylabel('station index column')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    main()

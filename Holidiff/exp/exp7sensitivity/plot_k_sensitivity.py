#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_k_sensitivity(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    if 'MAE_mean' in df.columns:
        x = df['K']
        mae = df['MAE_mean']
        mae_err = df['MAE_std'] if 'MAE_std' in df.columns else None
        time_val = df['time_mean'] if 'time_mean' in df.columns else df['inference_time_seconds_mean']
        time_err = df['time_std'] if 'time_std' in df.columns else None
    else:
        x = df['K']
        mae = df['MAE']
        mae_err = None
        time_val = df['inference_time_seconds']
        time_err = None

    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax1 = plt.subplots(figsize=(4.8, 3.4))
    color_mae = '#1f77b4'
    color_time = '#d6273a'
    if mae_err is not None:
        ax1.errorbar(x, mae, yerr=mae_err, color=color_mae, marker='o', linewidth=1.6, capsize=3, label='MAE')
    else:
        ax1.plot(x, mae, color=color_mae, marker='o', linewidth=1.6, label='MAE')
    ax1.set_xlabel('Number of sampled realizations K')
    ax1.set_ylabel('MAE', color=color_mae)
    ax1.tick_params(axis='y', labelcolor=color_mae)
    ax1.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    ax2 = ax1.twinx()
    if time_err is not None:
        ax2.errorbar(x, time_val, yerr=time_err, color=color_time, marker='s', linestyle='--', linewidth=1.5, capsize=3, label='Inference time')
    else:
        ax2.plot(x, time_val, color=color_time, marker='s', linestyle='--', linewidth=1.5, label='Inference time')
    ax2.set_ylabel('Inference time (s)', color=color_time)
    ax2.tick_params(axis='y', labelcolor=color_time)
    ax1.set_xticks(list(x))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)
    fig.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    print(f'Saved figure to {output}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out', type=str, default='figures/k_sensitivity.png')
    args = parser.parse_args()
    plot_k_sensitivity(args.csv, args.out)


if __name__ == '__main__':
    main()
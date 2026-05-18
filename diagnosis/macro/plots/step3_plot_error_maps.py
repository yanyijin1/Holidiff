import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
plt.style.use('seaborn-v0_8-whitegrid')


def draw_heatmap(data, fig_path_png, fig_path_pdf, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect='auto', cmap='viridis')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_path_png, dpi=200)
    fig.savefig(fig_path_pdf)
    plt.close(fig)


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
    overlay_path = in_dir / 'step1_worst_case_overlays.csv'
    overlays = pd.read_csv(overlay_path) if overlay_path.exists() else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timestep['minutes_ahead'], timestep['timestep_mae_raw'], marker='o')
    ax.set_xlabel('minutes ahead')
    ax.set_ylabel('timestep_mae_raw')
    ax.set_title('MAE over prediction horizon')
    fig.tight_layout()
    fig.savefig(fig_dir / 'step3_timestep_mae_curve.png', dpi=200)
    fig.savefig(fig_dir / 'step3_timestep_mae_curve.pdf')
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
    fig.savefig(fig_dir / 'step3_hourly_mae_curve.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(nodes['station_index'].astype(str), nodes['node_mae_raw'])
    ax.set_xlabel('station_index')
    ax.set_ylabel('node_mae_raw')
    ax.set_title('Top-10 node MAE')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / 'step3_node_mae_top10.png', dpi=200)
    fig.savefig(fig_dir / 'step3_node_mae_top10.pdf')
    plt.close(fig)

    draw_heatmap(inter_heat.to_numpy().T, fig_dir / 'step3_inter_dev_heatmap.png', fig_dir / 'step3_inter_dev_heatmap.pdf', 'Inter deviation heatmap', 'prediction timestep', 'station index column')
    draw_heatmap(intra_heat.to_numpy().T, fig_dir / 'step3_intra_var_heatmap.png', fig_dir / 'step3_intra_var_heatmap.pdf', 'Intra variance heatmap', 'prediction timestep', 'station index column')

    if len(overlays):
        # Group overlays by sample for combined subplot figures
        grouped = overlays.groupby('sample_id')

        for sample_id, group in grouped:
            n_nodes = len(group)
            fig, axes = plt.subplots(1, n_nodes, figsize=(7 * n_nodes, 4), squeeze=False)
            axes = axes[0]

            for ax_idx, (_, row) in enumerate(group.iterrows()):
                ax = axes[ax_idx]
                history = json.loads(row['history_values'])
                future_true = np.array(json.loads(row['future_true_values']), dtype=np.float64)
                future_pred = np.array(json.loads(row['future_pred_values']), dtype=np.float64)

                seq_len = len(history)
                pred_len = len(future_true)
                total_len = seq_len + pred_len

                # Build real timestamps: pred_start_time is the first future step
                pred_start = pd.Timestamp(row['pred_start_time'])
                freq = pd.Timedelta(minutes=15)
                hist_times = pd.date_range(end=pred_start - freq, periods=seq_len, freq=freq)
                fut_times = pd.date_range(start=pred_start, periods=pred_len, freq=freq)
                all_times = hist_times.append(fut_times)

                valid = future_true != 0
                masked_count = int((~valid).sum())

                fut_true_plot = np.where(valid, future_true, np.nan)
                fut_pred_plot = np.where(valid, future_pred, np.nan)

                # History line
                ax.plot(hist_times, history, color='#4C78A8', linewidth=2.0, label='history true')

                # Connect history last point to future first point (bridge segment)
                bridge_x = [hist_times[-1], fut_times[0]]
                bridge_true_y = [history[-1], fut_true_plot[0]]
                bridge_pred_y = [history[-1], fut_pred_plot[0]]
                ax.plot(bridge_x, bridge_true_y, color='#4C78A8', linestyle='--', linewidth=2.0)
                ax.plot(bridge_x, bridge_pred_y, color='#E45756', linestyle='-', linewidth=2.0)

                # Future lines
                ax.plot(fut_times, fut_true_plot, color='#4C78A8', linestyle='--', linewidth=2.0, label='future true')
                ax.plot(fut_times, fut_pred_plot, color='#E45756', linewidth=2.0, label='future pred')

                if masked_count > 0:
                    ax.scatter(fut_times[~valid], np.zeros(masked_count), color='gray', marker='x',
                               s=40, zorder=5, label=f'masked (n={masked_count})')

                ax.axvline(hist_times[-1], color='gray', linestyle=':', linewidth=1.2)
                ax.set_xlabel('time')
                ax.set_ylabel('flow')
                ax.set_title(
                    f"sample {int(row['sample_id'])} | station {int(row['station_index'])} | "
                    f"phase={row['phase']} | holiday={int(row['is_holiday'])}"
                )
                ax.legend(fontsize=8)
                ax.tick_params(axis='x', rotation=30)

            fig.tight_layout()
            stem = f"step3_overlay_sample{int(sample_id)}_combined"
            fig.savefig(fig_dir / f'{stem}.png', dpi=200)
            fig.savefig(fig_dir / f'{stem}.pdf')
            plt.close(fig)

        # Also generate individual plots for reference
        for _, row in overlays.iterrows():
            history = json.loads(row['history_values'])
            future_true = np.array(json.loads(row['future_true_values']), dtype=np.float64)
            future_pred = np.array(json.loads(row['future_pred_values']), dtype=np.float64)

            seq_len = len(history)
            pred_len = len(future_true)

            pred_start = pd.Timestamp(row['pred_start_time'])
            freq = pd.Timedelta(minutes=15)
            hist_times = pd.date_range(end=pred_start - freq, periods=seq_len, freq=freq)
            fut_times = pd.date_range(start=pred_start, periods=pred_len, freq=freq)

            valid = future_true != 0
            masked_count = int((~valid).sum())

            fut_true_plot = np.where(valid, future_true, np.nan)
            fut_pred_plot = np.where(valid, future_pred, np.nan)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(hist_times, history, color='#4C78A8', linewidth=2.0, label='history true')

            bridge_x = [hist_times[-1], fut_times[0]]
            bridge_true_y = [history[-1], fut_true_plot[0]]
            bridge_pred_y = [history[-1], fut_pred_plot[0]]
            ax.plot(bridge_x, bridge_true_y, color='#4C78A8', linestyle='--', linewidth=2.0)
            ax.plot(bridge_x, bridge_pred_y, color='#E45756', linestyle='-', linewidth=2.0)

            ax.plot(fut_times, fut_true_plot, color='#4C78A8', linestyle='--', linewidth=2.0, label='future true')
            ax.plot(fut_times, fut_pred_plot, color='#E45756', linewidth=2.0, label='future pred')
            if masked_count > 0:
                ax.scatter(fut_times[~valid], np.zeros(masked_count), color='gray', marker='x',
                           s=40, zorder=5, label=f'masked (n={masked_count})')
            ax.axvline(hist_times[-1], color='gray', linestyle=':', linewidth=1.2)
            ax.set_xlabel('time')
            ax.set_ylabel('flow')
            ax.set_title(
                f"Worst-case sample {int(row['sample_id'])} | station {int(row['station_index'])} | "
                f"phase={row['phase']} | holiday={int(row['is_holiday'])}"
            )
            ax.legend()
            ax.tick_params(axis='x', rotation=30)
            fig.tight_layout()
            stem = f"step3_overlay_sample{int(row['sample_id'])}_station{int(row['station_index'])}"
            fig.savefig(fig_dir / f'{stem}.png', dpi=200)
            fig.savefig(fig_dir / f'{stem}.pdf')
            plt.close(fig)


if __name__ == '__main__':
    main()

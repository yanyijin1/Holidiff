"""Plot Fujian-30 protocol analysis figures.

This version aligns curves by absolute time-of-day and aggregates on the clean CSV.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / 'data' / 'fujian-30'
CSV_PATH = DATA_ROOT / 'fujian30_clean.csv'
OUT_PATH = ROOT / 'fujian30_protocol_analysis.png'

GRID_COLOR = '#D9D9D9'
SHADE_COLOR = '#EDEDED'
BLUE = '#4E79A7'
ORANGE = '#F28E2B'
GREEN = '#76B7B2'
PURPLE = '#B07AA1'
RED = '#E15759'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'legend.fontsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 120,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    df['date'] = df['time_slot'].dt.date
    df['time_of_day'] = df['time_slot'].dt.strftime('%H:%M')
    df['is_holiday'] = df['is_holiday'].astype(int)
    df['station_index'] = df['station_index'].astype(int)
    df['traffic_flow'] = pd.to_numeric(df['traffic_flow'], errors='coerce')
    return df.dropna(subset=['traffic_flow']).sort_values(['date', 'time_of_day', 'station_index']).reset_index(drop=True)


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def hour_labels():
    return [f'{h:02d}:00' for h in range(24)]


def tick_labels(labels: list[str]):
    targets = ['00:00', '06:00', '12:00', '18:00', '23:00']
    idx, txt = [], []
    for t in targets:
        if t in labels:
            idx.append(labels.index(t))
            txt.append('24:00' if t == '23:00' else t)
    return idx, txt


def split_time_slots(df: pd.DataFrame, mode: str):
    times = pd.Index(sorted(df['time_slot'].unique()))
    t_total = len(times)
    train_end = int(t_total * 0.7)
    val_end = int(t_total * 0.8)
    input_len, pred_len = 96, 12
    window = input_len + pred_len
    starts = list(range(0, t_total - window + 1))
    holiday = df.groupby('time_slot')['is_holiday'].first().reindex(times).values

    buckets = {'train': set(), 'val': set(), 'test': set()}
    for s in starts:
        e = s + window
        if mode == 'standard':
            if e <= train_end:
                buckets['train'].update(times[s:e])
            elif train_end <= s and e <= val_end:
                buckets['val'].update(times[s:e])
            elif val_end <= s:
                buckets['test'].update(times[s:e])
        else:
            has_holiday = holiday[s:e].sum() > 0
            if e <= train_end and not has_holiday:
                buckets['train'].update(times[s:e])
            elif train_end <= s and e <= val_end and has_holiday:
                buckets['val'].update(times[s:e])
            elif val_end <= s and has_holiday:
                buckets['test'].update(times[s:e])
    return buckets


def daily_profile(df: pd.DataFrame, time_slots, holiday_filter=None):
    sub = df[df['time_slot'].isin(time_slots)].copy()
    if holiday_filter == 'regular':
        sub = sub[sub['is_holiday'] == 0]
    elif holiday_filter == 'holiday':
        sub = sub[sub['is_holiday'] == 1]
    g = sub.groupby(['date', 'time_of_day'], as_index=False)['traffic_flow'].mean()
    p = g.groupby('time_of_day')['traffic_flow'].agg(['mean', 'std', 'count']).reindex(hour_labels())
    p['mean'] = p['mean'].rolling(2, center=True, min_periods=1).mean().interpolate(limit_direction='both')
    p['std'] = p['std'].rolling(2, center=True, min_periods=1).mean()
    p['sem'] = p['std'] / np.sqrt(p['count'].clip(lower=1))
    p['sem'] = p['sem'].fillna(0.0)
    return p


def density(values, bins=40, x_range=(0, 400)):
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    values = values[(values >= x_range[0]) & (values <= x_range[1])]
    hist, edges = np.histogram(values, bins=bins, range=x_range, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist


def _plot_profile_panel(ax, prof_reg, prof_hol, title, legend_loc='upper right'):
    x = np.arange(len(hour_labels()))
    ax.axvspan(0, 7, color=SHADE_COLOR, alpha=0.65, zorder=0)
    ax.axvspan(21, 23, color=SHADE_COLOR, alpha=0.65, zorder=0)
    reg_mean = prof_reg['mean'].values
    reg_sem = prof_reg['sem'].values
    hol_mean = prof_hol['mean'].values
    hol_sem = prof_hol['sem'].values
    ax.plot(x, reg_mean, color=BLUE, lw=2.1, label=f'Regular ({np.nanmean(reg_mean):.1f} ± {np.nanmean(reg_sem):.1f})')
    ax.fill_between(x, reg_mean - reg_sem, reg_mean + reg_sem, color=BLUE, alpha=0.12)
    ax.plot(x, hol_mean, color=RED, lw=2.1, label=f'Holiday ({np.nanmean(hol_mean):.1f} ± {np.nanmean(hol_sem):.1f})')
    ax.fill_between(x, hol_mean - hol_sem, hol_mean + hol_sem, color=RED, alpha=0.12)
    gap = np.abs(hol_mean - reg_mean)
    peak_idx = int(np.nanargmax(gap))
    peak_y = max(float(reg_mean[peak_idx]), float(hol_mean[peak_idx]))
    ax.annotate('', xy=(peak_idx, peak_y * 0.985), xytext=(peak_idx, peak_y * 1.015),
                arrowprops={'arrowstyle': '<->', 'color': '#666666', 'lw': 1.0})
    ti, tt = tick_labels(hour_labels())
    ax.set_xticks(ti)
    ax.set_xticklabels(tt, fontsize=12)
    ax.set_xlabel('Time of day', fontsize=10, labelpad=3)
    ax.set_ylabel('Average flow', fontsize=10, labelpad=3)
    ax.set_title(title, fontsize=15, pad=6)
    ax.legend(frameon=False, loc=legend_loc, fontsize=9)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.6, color=GRID_COLOR)
    ax.spines[['top', 'right']].set_visible(False)


def plot_protocol_analysis():
    df = load_data()
    buckets_std = split_time_slots(df, 'standard')
    buckets_hol = split_time_slots(df, 'holiday_probe')

    std_profile = {split: daily_profile(df, buckets_std[split]) for split in ['train', 'val', 'test']}
    hol_profile = {split: daily_profile(df, buckets_hol[split]) for split in ['train', 'val', 'test']}

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    # (a) full dataset density: regular vs holiday
    reg_vals = df[df['is_holiday'] == 0]['traffic_flow'].values
    hol_vals = df[df['is_holiday'] == 1]['traffic_flow'].values
    x1, y1 = density(reg_vals)
    x2, y2 = density(hol_vals)
    ax = axes[0, 0]
    ax.plot(x1, y1, label=f'Regular (σ={np.std(reg_vals):.1f})', color=BLUE, linewidth=2.2)
    ax.plot(x2, y2, label=f'Holiday (σ={np.std(hol_vals):.1f})', color=RED, linewidth=2.2)
    ax.set_title('(a) Full dataset: regular vs holiday density')
    ax.set_xlabel('Flow')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 400)
    ax.legend(frameon=False)

    # (b) full dataset temporal shift
    ax = axes[0, 1]
    full_reg = daily_profile(df, df[df['is_holiday'] == 0]['time_slot'].unique(), 'regular')
    full_hol = daily_profile(df, df[df['is_holiday'] == 1]['time_slot'].unique(), 'holiday')
    _plot_profile_panel(ax, full_reg, full_hol, '(b) Full dataset: temporal shift', legend_loc='upper right')

    # (c) standard split density
    ax = axes[1, 0]
    for split, color in [('train', BLUE), ('val', ORANGE), ('test', GREEN)]:
        vals = df[df['time_slot'].isin(buckets_std[split])]['traffic_flow'].values
        xx, yy = density(vals)
        ax.plot(xx, yy, label=f'{split} (σ={np.std(vals):.1f})', color=color, linewidth=2.2)
    ax.set_title('(c) Standard split: train/val/test density')
    ax.set_xlabel('Flow')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 400)
    ax.legend(frameon=False)

    # (d) holiday probe split density
    ax = axes[1, 1]
    for split, color in [('train', BLUE), ('val', ORANGE), ('test', GREEN)]:
        vals = df[df['time_slot'].isin(buckets_hol[split])]['traffic_flow'].values
        xx, yy = density(vals)
        ax.plot(xx, yy, label=f'{split} (σ={np.std(vals):.1f})', color=color, linewidth=2.0)
    ax.set_title('(d) Holiday probe: train/val/test density')
    ax.set_xlabel('Flow')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 400)
    ax.legend(frameon=False)

    # (e) standard split temporal shift
    ax = axes[2, 0]
    x = np.arange(len(hour_labels()))
    for split, color in [('train', BLUE), ('val', ORANGE), ('test', GREEN)]:
        prof = std_profile[split]
        ax.plot(x, prof['mean'].values, label=f'{split} (±{np.nanmean(prof["sem"]):.1f})', color=color, linewidth=2.2)
        ax.fill_between(x, (prof['mean'] - prof['sem']).values, (prof['mean'] + prof['sem']).values,
                        color=color, alpha=0.12)
    ax.set_title('(e) Standard split: temporal shift')
    ax.set_xlabel('Time of day')
    ax.set_ylabel('Average flow')
    ti, tt = tick_labels(hour_labels())
    ax.set_xticks(ti)
    ax.set_xticklabels(tt, fontsize=12)
    ax.set_xlim(0, 23)
    ax.legend(frameon=False)

    # (f) holiday probe temporal shift
    ax = axes[2, 1]
    for split, color in [('train', BLUE), ('val', ORANGE), ('test', GREEN)]:
        prof = hol_profile[split]
        ax.plot(x, prof['mean'].values, label=f'{split} (±{np.nanmean(prof["sem"]):.1f})', color=color, linewidth=2.0)
        ax.fill_between(x, (prof['mean'] - prof['sem']).values, (prof['mean'] + prof['sem']).values,
                        color=color, alpha=0.12)
    ax.set_title('(f) Holiday probe: temporal shift')
    ax.set_xlabel('Time of day')
    ax.set_ylabel('Average flow')
    ti, tt = tick_labels(hour_labels())
    ax.set_xticks(ti)
    ax.set_xticklabels(tt, fontsize=12)
    ax.set_xlim(0, 23)
    ax.legend(frameon=False)

    fig.tight_layout(pad=0.6)
    save_figure(fig, OUT_PATH)
    print(f'saved figure to {OUT_PATH}')


if __name__ == '__main__':
    plot_protocol_analysis()

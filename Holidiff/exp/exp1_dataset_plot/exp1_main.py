from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Holidiff.data_provider.traffic_warehouse_loader import load_adj


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_path: Path
    adj_path: Path


DATASETS = {
    'fujian30': DatasetConfig(
        'fujian30',
        Path('/root/autodl-tmp/STdiff_data/fujian-30/fujian30_clean.csv'),
        Path('/root/autodl-tmp/STdiff_data/fujian-30/adjacent_gantry.csv'),
    ),
    'pems04-local50': DatasetConfig(
        'pems04-local50',
        Path('/root/autodl-tmp/STdiff_data/PEMS04-Local50/warehouse/clean.csv'),
        Path('/root/autodl-tmp/STdiff_data/PEMS04-Local50/PEMS04-Local50.csv'),
    ),
    'pems07-local60': DatasetConfig(
        'pems07-local60',
        Path('/root/autodl-tmp/STdiff_data/PEMS07-Local60/warehouse/clean.csv'),
        Path('/root/autodl-tmp/STdiff_data/PEMS07-Local60/PEMS07-Local60.csv'),
    ),
    'pems08-local40': DatasetConfig(
        'pems08-local40',
        Path('/root/autodl-tmp/STdiff_data/PEMS08-Local40/warehouse/clean.csv'),
        Path('/root/autodl-tmp/STdiff_data/PEMS08-Local40/PEMS08-Local40.csv'),
    ),
}


class Exp1DatasetPlot:
    def __init__(self, args=None):
        self.args = args

    def run(self):
        run_diagnostics(self.args)


def bounds(T: int):
    return int(T * 0.7), int(T * 0.8)


def qstats(x: np.ndarray, name: str):
    if x.size == 0:
        return {f'{name}_{k}': float('nan') for k in ['mean', 'median', 'p75', 'p90', 'p95']}
    return {
        f'{name}_mean': float(np.mean(x)),
        f'{name}_median': float(np.median(x)),
        f'{name}_p75': float(np.quantile(x, 0.75)),
        f'{name}_p90': float(np.quantile(x, 0.90)),
        f'{name}_p95': float(np.quantile(x, 0.95)),
    }


def load_dataset(cfg: DatasetConfig):
    df = pd.read_csv(cfg.csv_path)
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    if 'is_holiday' not in df.columns:
        df['is_holiday'] = 0.0
    df = df.sort_values(['time_slot', 'station_index']).reset_index(drop=True)
    pivot = df.pivot(index='time_slot', columns='station_index', values='traffic_flow').sort_index()
    data = pivot.values.astype(np.float32)
    holiday = df.groupby('time_slot').first().sort_index()['is_holiday'].values.astype(np.float32)
    adj = load_adj(cfg.adj_path, default_num_nodes=data.shape[1])
    src, dst = np.where(adj > 0)
    keep = src != dst
    return data, holiday, pd.DatetimeIndex(pivot.index), src[keep].astype(np.int64), dst[keep].astype(np.int64)


def starts_for_split(T: int, seq_len: int, pred_len: int, stride: int, split: str):
    tr, va = bounds(T)
    w = seq_len + pred_len
    xs = []
    for s in range(0, T - w + 1, stride):
        e = s + w
        if split == 'all':
            xs.append(s)
        elif split == 'train' and e <= tr:
            xs.append(s)
        elif split == 'val' and tr <= s and e <= va:
            xs.append(s)
        elif split == 'test' and va <= s:
            xs.append(s)
    return np.asarray(xs, dtype=np.int64)


def sample_starts(starts: np.ndarray, limit: int | None):
    if limit is None or len(starts) <= limit:
        return starts
    idx = np.linspace(0, len(starts) - 1, limit, dtype=int)
    return starts[idx]


def scale_from_train(data: np.ndarray):
    tr, _ = bounds(len(data))
    mean = data[:tr].mean(axis=0, keepdims=True)
    std = data[:tr].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((data - mean) / std).astype(np.float32)


def build_embed(z: np.ndarray, s: int, seq_len: int, pred_len: int, src: np.ndarray, dst: np.ndarray):
    x = z[s:s + seq_len]
    tail = x[-pred_len:]
    a = x.mean(axis=1)
    b = x.std(axis=1)
    c = np.mean(np.abs(tail[:, src] - tail[:, dst]), axis=1) if len(src) else np.zeros(pred_len, np.float32)
    return np.concatenate([a, b, c], axis=0).astype(np.float32)


def build_train_bank(z: np.ndarray, starts: np.ndarray, seq_len: int, pred_len: int, src: np.ndarray, dst: np.ndarray):
    embeds, futures = [], []
    for s in starts:
        embeds.append(build_embed(z, s, seq_len, pred_len, src, dst))
        futures.append(z[s + seq_len:s + seq_len + pred_len].reshape(-1).astype(np.float32))
    return np.stack(embeds), np.stack(futures)


def knn_future_var(bank_e: np.ndarray, bank_y: np.ndarray, q: np.ndarray, k: int):
    d = np.sum((bank_e - q[None, :]) ** 2, axis=1)
    if d.size == 0:
        return float('nan')
    k = max(1, min(k, d.size))
    idx = np.argpartition(d, k - 1)[:k]
    return float(np.mean(np.var(bank_y[idx], axis=0)))


def diagnose(data, holiday, time_idx, src, dst, seq_len, pred_len, split, stride, knn_k, max_eval, max_bank):
    z = scale_from_train(data)
    eval_starts = sample_starts(starts_for_split(len(data), seq_len, pred_len, stride, split), max_eval)
    bank_starts = sample_starts(starts_for_split(len(data), seq_len, pred_len, stride, 'train'), max_bank)
    bank_e, bank_y = build_train_bank(z, bank_starts, seq_len, pred_len, src, dst)
    tr, _ = bounds(len(data))
    train_net = z[:tr].mean(axis=1)
    q1, q2 = np.quantile(train_net, [0.33, 0.67])
    rows = []
    for i, s in enumerate(eval_starts):
        e, p = s + seq_len, s + seq_len + pred_len
        x, y = z[s:e], z[e:p]
        tail = x[-pred_len:]
        mu, sig = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-5)
        rb = np.mean(np.abs(y.mean(axis=0) - mu) / sig)
        hm, fm = tail.mean(axis=1), y.mean(axis=1)
        peak = abs(int(np.argmax(hm)) - int(np.argmax(fm))) / max(pred_len - 1, 1)
        if len(src):
            hf = np.mean(np.abs(tail[:, src] - tail[:, dst]), axis=1)
            ff = np.mean(np.abs(y[:, src] - y[:, dst]), axis=1)
            field = np.mean(np.abs(ff - hf)) / (np.std(hf) + 1e-5)
        else:
            field = 0.0
        hs = int(np.bincount(np.digitize(hm, [q1, q2]), minlength=3).argmax())
        cs = float(np.mean(np.digitize(fm, [q1, q2]) != hs))
        kv = knn_future_var(bank_e, bank_y, build_embed(z, s, seq_len, pred_len, src, dst), knn_k)
        rows.append({
            'sample_id': i,
            'start_idx': int(s),
            'start_time': str(time_idx[s]),
            'future_start_time': str(time_idx[e]),
            'pred_len': pred_len,
            'future_holiday_ratio': float(np.mean(holiday[e:p])) if len(holiday) else 0.0,
            'R_b': float(rb),
            'peak_shift_score': float(peak),
            'field_shift_score': float(field),
            'congestion_state_score': float(cs),
            'knn_future_variance': float(kv),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {'num_windows': 0}
    summary = {'num_windows': int(len(detail))}
    for m in ['future_holiday_ratio', 'R_b', 'peak_shift_score', 'field_shift_score', 'congestion_state_score', 'knn_future_variance']:
        summary.update(qstats(detail[m].values.astype(np.float64), m))
    return detail, summary


def run_diagnostics(args):
    names = list(DATASETS) if args.datasets == 'all' else [x.strip() for x in args.datasets.split(',') if x.strip()]
    horizons = [int(x) for x in args.horizons.split(',') if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        if name not in DATASETS:
            raise ValueError(f'Unknown dataset: {name}')
        data, holiday, time_idx, src, dst = load_dataset(DATASETS[name])
        ds_out = out_dir / name
        ds_out.mkdir(parents=True, exist_ok=True)
        for h in horizons:
            detail, summary = diagnose(data, holiday, time_idx, src, dst, args.seq_len, h, args.split, args.stride, args.knn_k, args.max_eval_windows, args.max_knn_pool)
            detail.to_csv(ds_out / f'diagnostics_h{h}_{args.split}_detail.csv', index=False)
            rows.append({'dataset': name, 'pred_len': h, 'split': args.split, 'seq_len': args.seq_len, 'stride': args.stride, 'num_nodes': int(data.shape[1]), 'num_edges': int(len(src)), **summary})
    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / f'diagnostics_summary_{args.split}.csv'
    summary_df.to_csv(summary_path, index=False)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print(f'\nSaved summary to: {summary_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Drift/shift diagnostics for four datasets')
    p.add_argument('--datasets', type=str, default='all')
    p.add_argument('--horizons', type=str, default='12,24,36')
    p.add_argument('--seq-len', dest='seq_len', type=int, default=96)
    p.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'])
    p.add_argument('--stride', type=int, default=12)
    p.add_argument('--knn-k', dest='knn_k', type=int, default=10)
    p.add_argument('--max-eval-windows', dest='max_eval_windows', type=int, default=800)
    p.add_argument('--max-knn-pool', dest='max_knn_pool', type=int, default=2000)
    p.add_argument('--out-dir', dest='out_dir', type=str, default='/root/autodl-tmp/STdiff_runs/diagnostics/four_dataset_shift')
    run_diagnostics(p.parse_args())

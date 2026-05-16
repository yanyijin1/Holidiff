import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Holidiff.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from Holidiff.data_provider.fujian30_loader import Fujian30CsvDataset

try:
    import yaml
except Exception as exc:
    yaml = None
    YAML_IMPORT_ERROR = exc


def load_yaml(path):
    if yaml is None:
        raise RuntimeError(f'PyYAML import failed: {YAML_IMPORT_ERROR}')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ns(cfg):
    cfg = dict(cfg)
    cfg['use_gpu'] = bool(torch.cuda.is_available() and cfg.get('use_gpu', True))
    if cfg.get('use_multi_gpu', False):
        ids = str(cfg.get('devices', '0')).replace(' ', '').split(',')
        cfg['device_ids'] = [int(x) for x in ids]
        cfg['gpu'] = cfg['device_ids'][0]
    return SimpleNamespace(**cfg)


def setting(a):
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        a.task_name, a.model_id, a.model, a.data, a.features, a.seq_len, a.label_len,
        a.pred_len, a.d_model, a.n_heads, a.e_layers, a.d_layers, a.d_ff, a.expand,
        a.d_conv, a.factor, a.embed, a.distil, a.des, 0)


def inv(ds, arr):
    flat = arr.reshape(-1, arr.shape[-1])
    return ds.scaler.inverse_transform(flat).reshape(arr.shape)


def core(model):
    return model.module if hasattr(model, 'module') else model


def load_meta(csv_path):
    df = pd.read_csv(csv_path, usecols=['time_slot', 'station_index', 'traffic_flow', 'is_holiday'])
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    df = df.sort_values(['time_slot', 'station_index']).reset_index(drop=True)
    pv = df.pivot(index='time_slot', columns='station_index', values='traffic_flow').sort_index()
    meta = df.groupby('time_slot').first().sort_index()
    return pv.index, pv.columns.to_numpy(), meta['is_holiday'].to_numpy(dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--sample_times', type=int, default=None)
    p.add_argument('--mode', default=None)
    p.add_argument('--output_dir', default=str(ROOT / 'diagnosis' / 'macro' / 'output'))
    c = p.parse_args()

    cfg = load_yaml(Path(c.config))
    a = ns(cfg)
    if c.batch_size is not None:
        a.batch_size = c.batch_size
    if c.sample_times is not None:
        a.vs_times = c.sample_times
    if c.mode is not None:
        a.mode = c.mode

    exp = Exp_Long_Term_Forecast(a)
    s = setting(a)
    ckpt = Path(c.checkpoint) if c.checkpoint else Path(a.checkpoints) / s / 'checkpoint.pth'
    if ckpt.is_dir():
        ckpt = ckpt / 'checkpoint.pth'
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    csv_path = Path(a.root_path) / a.data_path
    ds = Fujian30CsvDataset(str(csv_path), str(Path(a.root_path) / 'adjacent_gantry.csv'), c.split, getattr(a, 'mode', 'standard'), a.seq_len, a.pred_len, getattr(a, 'stride', 1), getattr(a, 'scale', True))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, drop_last=False, num_workers=c.num_workers)
    time_idx, station_ids, holiday_flag = load_meta(csv_path)

    m = core(exp.model)
    if hasattr(m, 'reset_diagnostics'):
        m.reset_diagnostics()

    preds, trues = [], []
    with torch.no_grad():
        for bx, by, bxm, bym in dl:
            bx, by = bx.float().to(exp.device), by.float().to(exp.device)
            bxm, bym = bxm.float().to(exp.device), bym.float().to(exp.device)
            dec = torch.zeros_like(by[:, -a.pred_len:, :]).float()
            dec = torch.cat([by[:, :a.label_len, :], dec], dim=1).float().to(exp.device)
            out, _ = exp.model(bx, bxm, dec, bym, sample_times=a.vs_times) if a.is_diff else (exp.model(bx, bxm, dec, bym), None)
            out = exp._process_model_output(out, is_diff=a.is_diff)
            preds.append(out.detach().cpu().numpy())
            trues.append(by[:, -a.pred_len:, :].detach().cpu().numpy())

    preds, trues = np.concatenate(preds, 0), np.concatenate(trues, 0)
    preds_raw, trues_raw = (inv(ds, preds), inv(ds, trues)) if getattr(a, 'scale', True) else (preds.copy(), trues.copy())
    starts = np.asarray(ds.indices[:len(preds)], dtype=np.int64)
    pred_starts = starts + a.seq_len
    pred_ends = pred_starts + a.pred_len - 1
    start_times = pd.to_datetime(time_idx[pred_starts])
    end_times = pd.to_datetime(time_idx[pred_ends])
    is_holiday = np.asarray([float(holiday_flag[s + a.seq_len:s + a.seq_len + a.pred_len].sum() > 0) for s in starts])
    flow_mean_raw = trues_raw.mean(axis=(1, 2))
    q33, q66 = np.percentile(flow_mean_raw, [33, 66])
    phase = np.where(flow_mean_raw < q33, 'free_flow', np.where(flow_mean_raw > q66, 'congested', 'transition'))

    intra = torch.cat(m._diag_block_var, 0).numpy() if m._diag_block_var else np.zeros(len(preds))
    inter = torch.cat(m._diag_inter_dev, 0).numpy() if m._diag_inter_dev else np.zeros(len(preds))
    bias = torch.cat(m._diag_median_bias, 0).numpy() if m._diag_median_bias else np.zeros(len(preds))
    intra_map = torch.cat(m._diag_block_var_map, 0).numpy() if m._diag_block_var_map else np.zeros_like(preds)
    inter_map = torch.cat(m._diag_inter_dev_map, 0).numpy() if m._diag_inter_dev_map else np.zeros_like(preds)

    out_dir = Path(c.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.DataFrame({
        'sample_id': np.arange(len(preds)), 'window_start_index': starts,
        'pred_start_index': pred_starts, 'pred_end_index': pred_ends,
        'pred_start_time': start_times.astype(str), 'pred_end_time': end_times.astype(str),
        'start_hour': start_times.hour.to_numpy(), 'is_holiday': is_holiday.astype(int),
        'phase': phase, 'sample_mae_scaled': np.abs(preds - trues).mean(axis=(1, 2)),
        'sample_rmse_scaled': np.sqrt(((preds - trues) ** 2).mean(axis=(1, 2))),
        'sample_mae_raw': np.abs(preds_raw - trues_raw).mean(axis=(1, 2)),
        'sample_rmse_raw': np.sqrt(((preds_raw - trues_raw) ** 2).mean(axis=(1, 2))),
        'flow_mean_raw': flow_mean_raw, 'negative_ratio_raw': (preds_raw < 0).mean(axis=(1, 2)),
        'intra_var': intra, 'inter_dev': inter, 'median_bias': bias,
    })
    sample.to_csv(out_dir / 'step1_sample_metrics.csv', index=False)

    pd.DataFrame({
        'station_index': station_ids,
        'node_mae_raw': np.abs(preds_raw - trues_raw).mean(axis=(0, 1)),
        'node_rmse_raw': np.sqrt(((preds_raw - trues_raw) ** 2).mean(axis=(0, 1))),
        'node_intra_var': intra_map.mean(axis=(0, 1)),
        'node_inter_dev': inter_map.mean(axis=(0, 1)),
    }).sort_values('node_mae_raw', ascending=False).to_csv(out_dir / 'step1_node_metrics.csv', index=False)

    pd.DataFrame({
        'timestep': np.arange(1, a.pred_len + 1), 'minutes_ahead': np.arange(1, a.pred_len + 1) * 15,
        'timestep_mae_raw': np.abs(preds_raw - trues_raw).mean(axis=(0, 2)),
        'timestep_rmse_raw': np.sqrt(((preds_raw - trues_raw) ** 2).mean(axis=(0, 2))),
        'timestep_intra_var': intra_map.mean(axis=2).mean(axis=0),
        'timestep_inter_dev': inter_map.mean(axis=2).mean(axis=0),
    }).to_csv(out_dir / 'step1_timestep_metrics.csv', index=False)

    sample.groupby(['is_holiday', 'start_hour'], as_index=False).agg({
        'sample_mae_raw': 'mean', 'sample_rmse_raw': 'mean', 'intra_var': 'mean',
        'inter_dev': 'mean', 'median_bias': 'mean', 'sample_id': 'count'
    }).rename(columns={'sample_id': 'sample_count'}).to_csv(out_dir / 'step1_hourly_metrics.csv', index=False)

    pd.DataFrame([{'metric': 'mae_raw', 'value': float(np.abs(preds_raw - trues_raw).mean())},
                  {'metric': 'rmse_raw', 'value': float(np.sqrt(((preds_raw - trues_raw) ** 2).mean()))},
                  {'metric': 'negative_ratio_raw_mean', 'value': float((preds_raw < 0).mean())},
                  {'metric': 'sample_count', 'value': int(len(preds))},
                  {'metric': 'node_count', 'value': int(len(station_ids))}]).to_csv(out_dir / 'step1_overview.csv', index=False)

    pd.DataFrame(np.abs(preds_raw - trues_raw).mean(axis=0), columns=[str(s) for s in station_ids]).to_csv(out_dir / 'step1_node_timestep_mae_raw.csv', index=False)
    pd.DataFrame(inter_map.mean(axis=0), columns=[str(s) for s in station_ids]).to_csv(out_dir / 'step1_node_timestep_inter_dev.csv', index=False)
    pd.DataFrame(intra_map.mean(axis=0), columns=[str(s) for s in station_ids]).to_csv(out_dir / 'step1_node_timestep_intra_var.csv', index=False)

    np.save(out_dir / 'step1_preds_raw.npy', preds_raw)
    np.save(out_dir / 'step1_trues_raw.npy', trues_raw)
    np.save(out_dir / 'step1_intra_var_map.npy', intra_map)
    np.save(out_dir / 'step1_inter_dev_map.npy', inter_map)
    print(f'Saved diagnostics to {out_dir}')


if __name__ == '__main__':
    main()

"""
Fujian-30 Dataset Loader

约定:
1) 先从原始 train_15min.csv 提取干净的总 CSV
2) 再基于干净 CSV 划分 standard / holiday_probe 两种协议
3) 不再生成 time_feat.npy
4) 路网统一保存为 adj.npy
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class _StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def build_clean_csv(csv_path, out_csv, target_col='traffic_flow'):
    """从原始 train_15min.csv 提取干净总 CSV。"""
    usecols = ['time_slot', 'station_index', target_col, 'is_holiday']
    df = pd.read_csv(csv_path, usecols=usecols)
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    df = df.sort_values(['time_slot', 'station_index']).reset_index(drop=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def _load_adj(adj_path):
    adj_path = Path(adj_path)
    if not adj_path.exists():
        return np.eye(30, dtype=np.float32)
    adj_df = pd.read_csv(adj_path)
    if {'src_FID', 'nbr_FID'}.issubset(adj_df.columns):
        nodes = sorted(set(adj_df['src_FID'].astype(int)).union(set(adj_df['nbr_FID'].astype(int))))
        node_to_idx = {nid: i for i, nid in enumerate(nodes)}
        adj = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for _, row in adj_df.iterrows():
            adj[node_to_idx[int(row['src_FID'])], node_to_idx[int(row['nbr_FID'])]] = 1.0
    else:
        adj = adj_df.values.astype(np.float32)
    return adj


def preprocess_csv(csv_path, out_dir, target_col='traffic_flow'):
    """生成干净总 CSV，并按两种协议输出 split-dict npy。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = out_dir / 'fujian30_clean.csv'
    df = pd.read_csv(csv_path, usecols=['time_slot', 'station_index', target_col, 'is_holiday'])
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    df = df.sort_values(['time_slot', 'station_index']).reset_index(drop=True)
    df.to_csv(clean_csv, index=False)

    pivot = df.pivot(index='time_slot', columns='station_index', values=target_col).sort_index()
    data = pivot.values.astype(np.float32)
    stations = pivot.columns.values.astype(np.int32)
    meta = df.groupby('time_slot').first().sort_index()
    holiday = meta['is_holiday'].values.astype(np.float32)

    adj = _load_adj(out_dir / 'adjacent_gantry.csv')

    train_end = int(len(data) * 0.7)
    val_end = int(len(data) * 0.8)
    input_len, pred_len = 96, 12
    window_size = input_len + pred_len
    all_starts = list(range(0, len(data) - window_size + 1))

    def split_indices(mode_name):
        idx = {'train': [], 'val': [], 'test': []}
        for s in all_starts:
            e = s + window_size
            if mode_name == 'standard':
                if e <= train_end:
                    idx['train'].append(s)
                elif train_end <= s and e <= val_end:
                    idx['val'].append(s)
                elif val_end <= s:
                    idx['test'].append(s)
            else:
                has_holiday = holiday[s:e].sum() > 0
                if e <= train_end and not has_holiday:
                    idx['train'].append(s)
                elif train_end <= s and e <= val_end and has_holiday:
                    idx['val'].append(s)
                elif val_end <= s and has_holiday:
                    idx['test'].append(s)
        return idx

    def pack_windows(starts):
        xs, ys, hs = [], [], []
        for s in starts:
            e = s + input_len
            p = e + pred_len
            xs.append(data[s:e])
            ys.append(data[e:p])
            hs.append(float(holiday[e:p].sum() > 0))
        return {
            'x': np.stack(xs) if xs else np.empty((0, input_len, data.shape[1]), dtype=np.float32),
            'y': np.stack(ys) if ys else np.empty((0, pred_len, data.shape[1]), dtype=np.float32),
            'holiday': np.asarray(hs, dtype=np.float32) if hs else np.empty((0,), dtype=np.float32),
        }

    for mode_name in ['standard', 'holiday_probe']:
        mode_dir = out_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        idx = split_indices(mode_name)
        for split in ['train', 'val', 'test']:
            np.save(mode_dir / f'{split}.npy', pack_windows(idx[split]))
        np.save(mode_dir / 'stations.npy', stations)
        np.save(mode_dir / 'adj.npy', adj)

    print(f'预处理完成: T={data.shape[0]}, N={data.shape[1]}')
    print(f'  clean csv: {clean_csv}')
    print(f'  standard/: {out_dir / "standard"}')
    print(f'  holiday_probe/: {out_dir / "holiday_probe"}')
    return out_dir


class Fujian30CsvDataset(Dataset):
    """完整 CSV 数据集，按窗口在线切分。"""
    def __init__(self, csv_path, adj_path, split='train', mode='standard',
                 input_len=96, pred_len=12, stride=1, scale=True):
        super().__init__()
        self.split = split
        self.mode = mode
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride
        self.scale = scale

        df = pd.read_csv(csv_path)
        df['time_slot'] = pd.to_datetime(df['time_slot'])
        df = df.sort_values(['time_slot', 'station_index']).reset_index(drop=True)

        pivot = df.pivot(index='time_slot', columns='station_index', values='traffic_flow').sort_index()
        raw_data = pivot.values.astype(np.float32)
        self.N = raw_data.shape[1]
        self.T_total = raw_data.shape[0]

        train_end = int(self.T_total * 0.7)
        self.scaler = _StandardScaler()
        if self.scale:
            self.scaler.fit(raw_data[:train_end])
            self.raw_data = self.scaler.transform(raw_data).astype(np.float32)
        else:
            self.raw_data = raw_data
        self.raw_zero_mask = (raw_data != 0).astype(np.float32)

        meta = df.groupby('time_slot').first().sort_index()
        self.holiday_flag = meta['is_holiday'].values.astype(np.float32)

        self.adj = torch.from_numpy(_load_adj(adj_path)).float()
        self.indices = self._build_indices()

    def _build_indices(self):
        window_size = self.input_len + self.pred_len
        all_starts = list(range(0, self.T_total - window_size + 1, self.stride))
        train_end = int(self.T_total * 0.7)
        val_end = int(self.T_total * 0.8)
        valid = []
        for s in all_starts:
            e = s + window_size
            if self.mode == 'standard':
                if self.split == 'train' and e <= train_end:
                    valid.append(s)
                elif self.split == 'val' and train_end <= s and e <= val_end:
                    valid.append(s)
                elif self.split == 'test' and val_end <= s:
                    valid.append(s)
            elif self.mode == 'holiday_probe':
                # Train: 整个窗口都不含节假日，确保只见过纯常规日分布
                whole_window_holiday = self.holiday_flag[s:e].sum() > 0
                # Val/Test: 只要求预测段 [e:p] 含节假日，测试“常规历史 -> 节假日未来”迁移
                p = e + self.pred_len
                future_holiday = self.holiday_flag[e:p].sum() > 0
                if self.split == 'train' and e <= train_end and not whole_window_holiday:
                    valid.append(s)
                elif self.split == 'val' and train_end <= s and e <= val_end and future_holiday:
                    valid.append(s)
                elif self.split == 'test' and val_end <= s and future_holiday:
                    valid.append(s)
            else:
                raise ValueError(f'Unknown mode: {self.mode}')
        return valid

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.input_len
        p = e + self.pred_len
        x = torch.from_numpy(self.raw_data[s:e]).float()
        y = torch.from_numpy(self.raw_data[e:p]).float()
        y_mask = torch.from_numpy(self.raw_zero_mask[e:p]).float()
        holiday_flag = torch.tensor(float(self.holiday_flag[e:p].sum() > 0), dtype=torch.float32)
        x_mark = torch.zeros(self.input_len, 1)
        y_mark = torch.zeros(self.input_len + self.pred_len, 1)
        return x, y, x_mark, y_mark, y_mask, holiday_flag


def get_dataloader(csv_path, adj_path, split, mode,
                   input_len=96, pred_len=12, stride=1,
                   batch_size=16, num_workers=4, shuffle=None):
    if shuffle is None:
        shuffle = (split == 'train')
    ds = Fujian30CsvDataset(
        csv_path=csv_path,
        adj_path=adj_path,
        split=split,
        mode=mode,
        input_len=input_len,
        pred_len=pred_len,
        stride=stride,
    )
    print(f'[{mode}] {split}: {len(ds)} samples')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class Fujian30NpyDataset(Dataset):
    """仅用于已经划分好的 npy 数据（画图/调试）。"""
    def __init__(self, data_dir, split='train', mode='standard'):
        super().__init__()
        data_dir = Path(data_dir) / mode
        pack = np.load(data_dir / f'{split}.npy', allow_pickle=True).item()
        self.x = pack['x']
        self.y = pack['y']
        self.holiday = pack['holiday']
        self.stations = np.load(data_dir / 'stations.npy')
        self.adj = torch.from_numpy(np.load(data_dir / 'adj.npy')).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': torch.from_numpy(self.x[idx]).float().unsqueeze(-1).permute(1, 0, 2),
            'y': torch.from_numpy(self.y[idx]).float().unsqueeze(-1).permute(1, 0, 2),
            'holiday': torch.tensor([float(self.holiday[idx])], dtype=torch.float32),
            'adj': self.adj,
        }

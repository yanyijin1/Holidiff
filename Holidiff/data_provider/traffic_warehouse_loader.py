"""
Shared traffic warehouse loader.

Unified contract:
1) A clean long-table CSV with at least:
   - time_slot
   - station_index
   - target column, e.g. traffic_flow
   - optional is_holiday
2) A graph file or adjacency file for road topology.
3) Online window slicing with standard / holiday_probe protocols.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


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


def split_train_val_test_tail_history(
    data,
    train_ratio=0.7,
    val_ratio=0.1,
    history_ratio=0.7,
):
    """
    Fixed chronological split with tail-history truncation.

    Full sequence:
        [0, train_end)       full train
        [train_end, val_end) validation
        [val_end, T)         test

    For history_ratio=gamma, train uses:
        [train_end - gamma*T, train_end)

    Here gamma is defined over the full time series, not over the full train.
    With train_ratio=0.7, gamma must be in (0, train_ratio].
    """
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    if not (0 < history_ratio <= train_ratio):
        raise ValueError(
            f'history_ratio must be in (0, {train_ratio}], got {history_ratio}'
        )

    keep_len = max(1, int(T * history_ratio))
    train_start = max(0, train_end - keep_len)

    train_data = data[train_start:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    split_info = {
        'T': int(T),
        'train_ratio': float(train_ratio),
        'val_ratio': float(val_ratio),
        'history_ratio': float(history_ratio),
        'train_start': int(train_start),
        'train_end': int(train_end),
        'val_start': int(train_end),
        'val_end': int(val_end),
        'test_start': int(val_end),
        'test_end': int(T),
        'train_timesteps': int(train_end - train_start),
        'val_timesteps': int(val_end - train_end),
        'test_timesteps': int(T - val_end),
    }

    return train_data, val_data, test_data, split_info


def split_train_val_test_tail_history_with_mask(
    data,
    valid_mask,
    train_ratio=0.7,
    val_ratio=0.1,
    history_ratio=0.7,
):
    train_data, val_data, test_data, split_info = split_train_val_test_tail_history(
        data=data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        history_ratio=history_ratio,
    )

    train_start = split_info['train_start']
    train_end = split_info['train_end']
    val_end = split_info['val_end']

    train_mask = valid_mask[train_start:train_end]
    val_mask = valid_mask[train_end:val_end]
    test_mask = valid_mask[val_end:]

    return train_data, val_data, test_data, train_mask, val_mask, test_mask, split_info


def build_clean_csv(
    csv_path,
    out_csv,
    target_col='traffic_flow',
    time_col='time_slot',
    node_col='station_index',
    holiday_col='is_holiday',
):
    preview = pd.read_csv(csv_path, nrows=0)
    columns = set(preview.columns)
    usecols = [time_col, node_col, target_col]
    has_holiday = bool(holiday_col) and holiday_col in columns
    if has_holiday:
        usecols.append(holiday_col)

    df = pd.read_csv(csv_path, usecols=usecols)
    df[time_col] = pd.to_datetime(df[time_col])
    if has_holiday:
        df[holiday_col] = df[holiday_col].fillna(0).astype(np.float32)
    else:
        df['is_holiday'] = 0.0
        holiday_col = 'is_holiday'
    df = df.sort_values([time_col, node_col]).reset_index(drop=True)
    df = df.rename(columns={time_col: 'time_slot', node_col: 'station_index', target_col: 'traffic_flow', holiday_col: 'is_holiday'})
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def build_clean_csv_from_pems_npz(
    npz_path,
    out_csv,
    node_txt_path=None,
    array_key='data',
    channel_idx=0,
    freq='5min',
    start_time='2000-01-01 00:00:00',
):
    npz_path = Path(npz_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    obj = np.load(npz_path, allow_pickle=True)
    if array_key not in obj.files:
        raise KeyError(f"Key '{array_key}' not found in {npz_path}. Available keys: {obj.files}")

    data = obj[array_key]
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(f'Expected PEMS array with shape [T, N, C], got {data.shape}')
    if channel_idx < 0 or channel_idx >= data.shape[-1]:
        raise IndexError(f'channel_idx={channel_idx} out of range for shape {data.shape}')

    flow = np.asarray(data[:, :, channel_idx], dtype=np.float32)
    t_steps, n_nodes = flow.shape

    if node_txt_path is not None and Path(node_txt_path).exists():
        with open(node_txt_path, 'r', encoding='utf-8') as f:
            node_ids = [line.strip() for line in f if line.strip()]
        if len(node_ids) != n_nodes:
            raise ValueError(
                f'Node count mismatch: txt has {len(node_ids)} ids but npz has {n_nodes} nodes.'
            )
        node_ids = np.asarray(node_ids)
    else:
        node_ids = np.arange(n_nodes)

    time_index = pd.date_range(start=start_time, periods=t_steps, freq=freq)
    time_col = np.repeat(time_index.values, n_nodes)
    node_col = np.tile(node_ids, t_steps)
    flow_col = flow.reshape(-1)

    df = pd.DataFrame({
        'time_slot': pd.to_datetime(time_col),
        'station_index': node_col,
        'traffic_flow': flow_col,
    })
    df.to_csv(out_csv, index=False)
    return out_csv


def load_adj(adj_path, default_num_nodes=None):
    adj_path = Path(adj_path)
    if not adj_path.exists():
        n = int(default_num_nodes or 1)
        adj = np.eye(n, dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        return adj

    if adj_path.suffix == '.npy':
        adj = np.load(adj_path).astype(np.float32)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f'Adjacency matrix must be square: {adj_path}')
        np.fill_diagonal(adj, 0.0)
        return adj

    adj_df = pd.read_csv(adj_path)
    if {'src_FID', 'nbr_FID'}.issubset(adj_df.columns):
        src_col, dst_col = 'src_FID', 'nbr_FID'
    elif {'from', 'to'}.issubset(adj_df.columns):
        src_col, dst_col = 'from', 'to'
    else:
        values = adj_df.values.astype(np.float32)
        if values.ndim == 2 and values.shape[0] == values.shape[1]:
            np.fill_diagonal(values, 0.0)
            return values
        raise ValueError(f'Unsupported adjacency file format: {adj_path}')

    nodes = sorted(set(adj_df[src_col].astype(int)).union(set(adj_df[dst_col].astype(int))))
    node_to_idx = {nid: i for i, nid in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for _, row in adj_df.iterrows():
        src = node_to_idx[int(row[src_col])]
        dst = node_to_idx[int(row[dst_col])]
        if src != dst:
            adj[src, dst] = 1.0
    np.fill_diagonal(adj, 0.0)
    return adj


def preprocess_traffic_csv(
    csv_path,
    out_dir,
    target_col='traffic_flow',
    time_col='time_slot',
    node_col='station_index',
    holiday_col='is_holiday',
    adj_path=None,
    clean_name='clean.csv',
    input_len=96,
    pred_len=12,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = out_dir / clean_name
    build_clean_csv(
        csv_path=csv_path,
        out_csv=clean_csv,
        target_col=target_col,
        time_col=time_col,
        node_col=node_col,
        holiday_col=holiday_col,
    )

    df = pd.read_csv(clean_csv)
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    df = df.sort_values(['time_slot', 'station_index']).reset_index(drop=True)

    pivot = df.pivot(index='time_slot', columns='station_index', values='traffic_flow').sort_index()
    data = pivot.values.astype(np.float32)
    stations = pivot.columns.values.astype(np.int32)
    has_holiday = 'is_holiday' in df.columns
    if has_holiday:
        meta = df.groupby('time_slot').first().sort_index()
        holiday = meta['is_holiday'].values.astype(np.float32)
    else:
        holiday = np.zeros(len(pivot), dtype=np.float32)

    if adj_path is None:
        adj_path = out_dir / 'adjacent_gantry.csv'
    adj = load_adj(adj_path, default_num_nodes=len(stations))

    train_end = int(len(data) * 0.7)
    val_end = int(len(data) * 0.8)
    window_size = input_len + pred_len
    all_starts = list(range(0, len(data) - window_size + 1))

    def split_indices(mode_name):
        idx = {'train': [], 'val': [], 'test': []}
        for s in all_starts:
            e = s + window_size
            p = s + input_len + pred_len
            if mode_name == 'standard':
                if e <= train_end:
                    idx['train'].append(s)
                elif train_end <= s and e <= val_end:
                    idx['val'].append(s)
                elif val_end <= s:
                    idx['test'].append(s)
            elif mode_name == 'holiday_probe':
                whole_window_holiday = holiday[s:e].sum() > 0
                future_holiday = holiday[s + input_len:p].sum() > 0
                if e <= train_end and not whole_window_holiday:
                    idx['train'].append(s)
                elif train_end <= s and e <= val_end and future_holiday:
                    idx['val'].append(s)
                elif val_end <= s and future_holiday:
                    idx['test'].append(s)
            else:
                raise ValueError(f'Unknown mode: {mode_name}')
        return idx

    def pack_windows(starts, include_holiday):
        xs, ys, hs = [], [], []
        for s in starts:
            e = s + input_len
            p = e + pred_len
            xs.append(data[s:e])
            ys.append(data[e:p])
            if include_holiday:
                hs.append(holiday[e:p].astype(np.float32))
        output = {
            'x': np.stack(xs) if xs else np.empty((0, input_len, data.shape[1]), dtype=np.float32),
            'y': np.stack(ys) if ys else np.empty((0, pred_len, data.shape[1]), dtype=np.float32),
        }
        if include_holiday:
            output['holiday'] = np.stack(hs) if hs else np.empty((0, pred_len), dtype=np.float32)
        return output

    mode_names = ['standard', 'holiday_probe'] if has_holiday else ['standard']
    for mode_name in mode_names:
        mode_dir = out_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        idx = split_indices(mode_name)
        for split in ['train', 'val', 'test']:
            np.save(mode_dir / f'{split}.npy', pack_windows(idx[split], include_holiday=has_holiday))
        np.save(mode_dir / 'stations.npy', stations)
        np.save(mode_dir / 'adj.npy', adj)

    print(f'预处理完成: T={data.shape[0]}, N={data.shape[1]}, has_holiday={has_holiday}')
    print(f'  clean csv: {clean_csv}')
    print(f'  generated modes: {mode_names}')
    return out_dir


def preprocess_pems_npz(
    npz_path,
    out_dir,
    adj_path,
    node_txt_path=None,
    array_key='data',
    channel_idx=0,
    freq='5min',
    start_time='2000-01-01 00:00:00',
    clean_name='clean.csv',
    input_len=96,
    pred_len=12,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = out_dir / clean_name
    build_clean_csv_from_pems_npz(
        npz_path=npz_path,
        out_csv=clean_csv,
        node_txt_path=node_txt_path,
        array_key=array_key,
        channel_idx=channel_idx,
        freq=freq,
        start_time=start_time,
    )

    df = pd.read_csv(clean_csv)
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    df = df.sort_values(['time_slot', 'station_index']).reset_index(drop=True)

    pivot = df.pivot(index='time_slot', columns='station_index', values='traffic_flow').sort_index()
    data = pivot.values.astype(np.float32)
    stations = pivot.columns.to_numpy()
    adj = load_adj(adj_path, default_num_nodes=len(stations))

    train_end = int(len(data) * 0.7)
    val_end = int(len(data) * 0.8)
    window_size = input_len + pred_len
    all_starts = list(range(0, len(data) - window_size + 1))

    def split_indices():
        idx = {'train': [], 'val': [], 'test': []}
        for s in all_starts:
            e = s + window_size
            if e <= train_end:
                idx['train'].append(s)
            elif train_end <= s and e <= val_end:
                idx['val'].append(s)
            elif val_end <= s:
                idx['test'].append(s)
        return idx

    def pack_windows(starts):
        xs, ys = [], []
        for s in starts:
            e = s + input_len
            p = e + pred_len
            xs.append(data[s:e])
            ys.append(data[e:p])
        return {
            'x': np.stack(xs) if xs else np.empty((0, input_len, data.shape[1]), dtype=np.float32),
            'y': np.stack(ys) if ys else np.empty((0, pred_len, data.shape[1]), dtype=np.float32),
        }

    mode_dir = out_dir / 'standard'
    mode_dir.mkdir(parents=True, exist_ok=True)
    idx = split_indices()
    for split in ['train', 'val', 'test']:
        np.save(mode_dir / f'{split}.npy', pack_windows(idx[split]))
    np.save(mode_dir / 'stations.npy', stations)
    np.save(mode_dir / 'adj.npy', adj)

    print(f'预处理完成: T={data.shape[0]}, N={data.shape[1]}, has_holiday=False')
    print(f'  clean csv: {clean_csv}')
    print(f'  generated modes: [\'standard\']')
    return out_dir


class TrafficWarehouseCsvDataset(Dataset):
    """Unified long-table traffic warehouse dataset with online window slicing."""

    def __init__(
        self,
        csv_path,
        adj_path,
        split='train',
        mode='standard',
        input_len=96,
        pred_len=12,
        stride=1,
        scale=True,
        target_col='traffic_flow',
        time_col='time_slot',
        node_col='station_index',
        holiday_col='is_holiday',
        zero_as_missing=False,
        extreme_filter_threshold=None,
        train_ratio=0.7,
        val_ratio=0.1,
        history_ratio=0.7,
    ):
        super().__init__()
        self.split = split
        self.mode = mode
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride
        self.scale = scale
        self.target_col = target_col
        self.time_col = time_col
        self.node_col = node_col
        self.holiday_col = holiday_col
        self.zero_as_missing = zero_as_missing
        self.extreme_filter_threshold = extreme_filter_threshold
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.history_ratio = float(history_ratio)

        df = pd.read_csv(csv_path)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        if self.holiday_col not in df.columns:
            df[self.holiday_col] = 0.0
        df = df.sort_values([self.time_col, self.node_col]).reset_index(drop=True)

        pivot = df.pivot(index=self.time_col, columns=self.node_col, values=self.target_col).sort_index()
        raw_data = pivot.values.astype(np.float32)
        valid_mask = (np.isfinite(raw_data) & (raw_data > 0)).astype(np.float32)
        if self.zero_as_missing:
            clean_df = pivot.mask(~np.isfinite(raw_data) | (raw_data <= 0))
            clean_df = clean_df.interpolate(axis=0, limit_direction='both').ffill().bfill()
            raw_data = clean_df.values.astype(np.float32)
        self.N = raw_data.shape[1]
        self.T_total = raw_data.shape[0]
        self.stations = pivot.columns.to_numpy()

        train_data, val_data, test_data, train_mask, val_mask, test_mask, split_info = split_train_val_test_tail_history_with_mask(
            data=raw_data,
            valid_mask=valid_mask,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            history_ratio=self.history_ratio,
        )
        self.split_info = split_info
        self.train_end = int(split_info['train_end'])
        self.val_end = int(split_info['val_end'])
        self.train_start = int(split_info['train_start'])
        self.train_timesteps = int(split_info['train_timesteps'])
        self.val_timesteps = int(split_info['val_timesteps'])
        self.test_timesteps = int(split_info['test_timesteps'])

        self.scaler = _StandardScaler()
        if self.scale:
            self.scaler.fit(train_data)
            train_scaled = self.scaler.transform(train_data).astype(np.float32)
            val_scaled = self.scaler.transform(val_data).astype(np.float32)
            test_scaled = self.scaler.transform(test_data).astype(np.float32)
        else:
            train_scaled = train_data.astype(np.float32)
            val_scaled = val_data.astype(np.float32)
            test_scaled = test_data.astype(np.float32)

        if self.split == 'train':
            self.data = train_scaled
            self.mask = train_mask.astype(np.float32)
            self.global_start_offset = self.train_start
        elif self.split == 'val':
            self.data = val_scaled
            self.mask = val_mask.astype(np.float32)
            self.global_start_offset = self.train_end
        elif self.split == 'test':
            self.data = test_scaled
            self.mask = test_mask.astype(np.float32)
            self.global_start_offset = self.val_end
        else:
            raise ValueError(f'Unknown split: {self.split}')

        meta = df.groupby(self.time_col).first().sort_index()
        holiday_flag = meta[self.holiday_col].values.astype(np.float32)
        self.train_holiday = holiday_flag[self.train_start:self.train_end]
        self.val_holiday = holiday_flag[self.train_end:self.val_end]
        self.test_holiday = holiday_flag[self.val_end:]
        if self.split == 'train':
            self.holiday_flag = self.train_holiday
        elif self.split == 'val':
            self.holiday_flag = self.val_holiday
        else:
            self.holiday_flag = self.test_holiday

        self.adj = torch.from_numpy(load_adj(adj_path, default_num_nodes=self.N)).float()
        self.indices = self._build_indices()
        self._apply_extreme_filter()

    def _build_indices(self):
        window_size = self.input_len + self.pred_len
        total_len = self.data.shape[0]
        if total_len < window_size:
            return []

        valid = []
        for local_start in range(0, total_len - window_size + 1, self.stride):
            e = local_start + window_size
            p = local_start + self.input_len + self.pred_len
            if self.mode == 'standard':
                valid.append(self.global_start_offset + local_start)
            elif self.mode == 'holiday_probe':
                whole_window_holiday = self.holiday_flag[local_start:e].sum() > 0
                future_holiday = self.holiday_flag[local_start + self.input_len:p].sum() > 0
                if self.split == 'train' and not whole_window_holiday:
                    valid.append(self.global_start_offset + local_start)
                elif self.split in {'val', 'test'} and future_holiday:
                    valid.append(self.global_start_offset + local_start)
            else:
                raise ValueError(f'Unknown mode: {self.mode}')
        return valid

    def _apply_extreme_filter(self):
        threshold = self.extreme_filter_threshold
        if self.split != 'train' or threshold is None or threshold <= 0:
            return

        kept = []
        removed = 0
        for s in self.indices:
            local_s = s - self.global_start_offset
            e = local_s + self.input_len
            p = e + self.pred_len
            seq_x = self.data[local_s:e]
            seq_y = self.data[e:p]
            hist_mean = seq_x.mean(axis=0, keepdims=True)
            hist_std = seq_x.std(axis=0, keepdims=True)
            score = np.max(np.abs((seq_y - hist_mean) / (hist_std + 1e-5)))
            if np.isfinite(score) and score <= threshold:
                kept.append(s)
            else:
                removed += 1
        total = len(self.indices)
        self.indices = kept
        ratio = 0.0 if total == 0 else removed / total
        print(f'\t[extreme_filter] threshold={threshold:g}, removed={removed}/{total} ({ratio:.2%})')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        local_s = s - self.global_start_offset
        e = local_s + self.input_len
        p = e + self.pred_len
        x = torch.from_numpy(self.data[local_s:e]).float()
        y = torch.from_numpy(self.data[e:p]).float()
        y_mask = torch.from_numpy(self.mask[e:p]).float()
        holiday_seq = torch.from_numpy(self.holiday_flag[e:p]).float()
        x_mark = torch.zeros(self.input_len, 1)
        y_mark = torch.zeros(self.input_len + self.pred_len, 1)
        return x, y, x_mark, y_mark, y_mask, holiday_seq


class TrafficWarehouseNpyDataset(Dataset):
    """Numpy-packed traffic warehouse dataset for debugging and plotting."""

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
            'holiday': torch.from_numpy(self.holiday[idx]).float(),
            'adj': self.adj,
        }


def get_dataloader(csv_path, adj_path, split, mode, input_len=96, pred_len=12, stride=1, batch_size=16, num_workers=4, shuffle=None, scale=True, train_ratio=0.7, val_ratio=0.1, history_ratio=0.7):
    if shuffle is None:
        shuffle = split == 'train'
    ds = TrafficWarehouseCsvDataset(
        csv_path=csv_path,
        adj_path=adj_path,
        split=split,
        mode=mode,
        input_len=input_len,
        pred_len=pred_len,
        stride=stride,
        scale=scale,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        history_ratio=history_ratio,
    )
    print(f'[{mode}] {split}: {len(ds)} samples')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

from collections import deque
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def load_flow(npz_path):
    obj = np.load(npz_path, allow_pickle=True)
    data = obj['data']
    if data.ndim == 3:
        data = data[:, :, 0]
    if data.ndim != 2:
        raise ValueError(f'Expected [T,N] or [T,N,C], got {data.shape}')
    return np.asarray(data, dtype=np.float32), np.arange(data.shape[1])


def load_flow_from_clean_csv(csv_path):
    df = pd.read_csv(csv_path, usecols=['time_slot', 'station_index', 'traffic_flow'])
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    pivot = df.pivot(index='time_slot', columns='station_index', values='traffic_flow').sort_index()
    return pivot.values.astype(np.float32), pivot.columns.to_numpy()


def interpolate_invalid(raw):
    valid_mask = np.isfinite(raw) & (raw > 0)
    clean = pd.DataFrame(raw).mask(~valid_mask).interpolate(axis=0, limit_direction='both').ffill().bfill()
    return clean.values.astype(np.float32), valid_mask.astype(np.float32)


def load_graph(adj_path, node_ids):
    edges = pd.read_csv(adj_path)
    if not {'from', 'to'}.issubset(edges.columns):
        raise ValueError(f'Expected edge list with from/to columns: {adj_path}')
    node_to_col = {int(node_id): i for i, node_id in enumerate(node_ids)}
    neighbors = [set() for _ in range(len(node_ids))]
    for _, row in edges.iterrows():
        src_id = int(row['from'])
        dst_id = int(row['to'])
        if src_id in node_to_col and dst_id in node_to_col:
            u = node_to_col[src_id]
            v = node_to_col[dst_id]
            if u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)
    return neighbors


def graph_distances(neighbors, source):
    dist = np.full(len(neighbors), np.inf, dtype=np.float64)
    dist[source] = 0.0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if not np.isfinite(dist[v]):
                dist[v] = dist[u] + 1.0
                q.append(v)
    return dist


def connected_component(neighbors, source):
    seen = {source}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def compute_node_scores(raw, clean, valid_mask, seq_len=96, train_ratio=0.7, day_len=288, low_std_eps=1e-3):
    train_end = int(clean.shape[0] * train_ratio)
    x = clean[:train_end]
    valid = valid_mask[:train_end]

    valid_rate = valid.mean(axis=0)
    s_var = np.nanstd(x, axis=0)
    s_change = np.nanmean(np.abs(np.diff(x, axis=0)), axis=0)

    complete_days = train_end // day_len
    if complete_days > 1:
        daily = x[:complete_days * day_len].reshape(complete_days, day_len, x.shape[1])
        peak_pos = np.argmax(daily, axis=1)
        s_peak = np.std(peak_pos, axis=0)
    else:
        s_peak = np.zeros(x.shape[1], dtype=np.float64)

    starts = range(0, max(train_end - seq_len + 1, 0), seq_len)
    low_counts = np.zeros(x.shape[1], dtype=np.float64)
    total = 0
    for s in starts:
        hist_std = np.std(x[s:s + seq_len], axis=0)
        low_counts += hist_std < low_std_eps
        total += 1
    low_ratio = low_counts / max(total, 1)

    s = (minmax(s_var) + minmax(s_change) + minmax(s_peak)) / 3.0
    q_star = valid_rate * (1.0 - low_ratio)
    return {
        'S_var': s_var,
        'S_change': s_change,
        'S_peak': s_peak,
        'S': s,
        'Q': valid_rate,
        'L': low_ratio,
        'Q_star': q_star,
    }


def select_nested_nodes(scores, neighbors, target_sizes=(40, 60, 100), alpha=0.5, beta=0.3, lam=0.2):
    n_nodes = len(neighbors)
    candidate_score = scores['S'] * scores['Q_star']
    order = np.argsort(-candidate_score)
    anchor = None
    component = None
    for node in order:
        comp = connected_component(neighbors, int(node))
        if len(comp) >= max(target_sizes):
            anchor = int(node)
            component = comp
            break
    if anchor is None:
        raise RuntimeError(f'No connected component can cover target size {max(target_sizes)}')

    dist = graph_distances(neighbors, anchor)
    selected = [anchor]
    selected_set = {anchor}
    frontier = set(neighbors[anchor]) & component
    while len(selected) < max(target_sizes):
        if not frontier:
            raise RuntimeError(f'Frontier exhausted at size {len(selected)}')
        best = max(
            frontier,
            key=lambda v: (alpha * scores['S'][v] + beta * scores['Q_star'][v] - lam * dist[v], scores['S'][v], -dist[v]),
        )
        selected.append(int(best))
        selected_set.add(best)
        frontier.remove(best)
        frontier.update((neighbors[best] & component) - selected_set)
    return anchor, selected


def induced_edge_list(neighbors, nodes):
    node_to_local = {node: i for i, node in enumerate(nodes)}
    rows = []
    for u in nodes:
        for v in neighbors[u]:
            if v in node_to_local:
                rows.append({'from': node_to_local[u], 'to': node_to_local[v], 'cost': 1.0})
    return pd.DataFrame(rows).drop_duplicates().sort_values(['from', 'to']).reset_index(drop=True)


def diameter(nodes, neighbors):
    node_set = set(nodes)
    max_d = 0
    for src in nodes:
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v in node_set and v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if len(dist) != len(nodes):
            return np.inf
        max_d = max(max_d, max(dist.values()))
    return max_d


def r_diagnostics(clean, nodes, seq_len=96, pred_len=12, stride=12):
    data = clean[:, nodes]
    train_end = int(data.shape[0] * 0.7)
    train = data[:train_end]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (data - mean) / std
    r_values = []
    low_var = 0
    total = 0
    for s in range(0, data.shape[0] - seq_len - pred_len + 1, stride):
        e = s + seq_len
        p = e + pred_len
        x = z[s:e]
        y = z[e:p]
        hist_mean = x.mean(axis=0, keepdims=True)
        hist_std = x.std(axis=0, keepdims=True)
        r = np.max(np.abs((y - hist_mean) / (hist_std + 1e-5)))
        if np.isfinite(r):
            r_values.append(float(r))
        if np.any(hist_std < 1e-3):
            low_var += 1
        total += 1
    r_values = np.asarray(r_values, dtype=np.float64)
    return {
        'R99': float(np.percentile(r_values, 99)),
        'R999': float(np.percentile(r_values, 99.9)),
        'Rmax': float(np.max(r_values)),
        'low_var_window_ratio': low_var / max(total, 1),
    }


def write_local_dataset(name, out_root, raw, clean, valid_mask, nodes, neighbors, scores, source_node_ids):
    out_dir = out_root / name
    warehouse = out_dir / 'warehouse'
    warehouse.mkdir(parents=True, exist_ok=True)

    time_index = pd.date_range('2000-01-01 00:00:00', periods=clean.shape[0], freq='5min')
    local_data = clean[:, nodes]
    rows = []
    for local_idx in range(len(nodes)):
        rows.append(pd.DataFrame({
            'time_slot': time_index,
            'station_index': local_idx,
            'traffic_flow': local_data[:, local_idx],
            'is_holiday': 0.0,
        }))
    df = pd.concat(rows, ignore_index=True).sort_values(['time_slot', 'station_index'])
    df.to_csv(warehouse / 'clean.csv', index=False)

    edge_df = induced_edge_list(neighbors, nodes)
    edge_df.to_csv(out_dir / f'{name}.csv', index=False)
    pd.DataFrame({
        'local_index': np.arange(len(nodes)),
        'source_col': nodes,
        'source_node': np.asarray(source_node_ids)[nodes],
    }).to_csv(out_dir / 'node_mapping.csv', index=False)

    diag = r_diagnostics(clean, nodes)
    undirected_edges = len(edge_df) // 2
    avg_degree = 0.0 if len(nodes) == 0 else 2.0 * undirected_edges / len(nodes)
    stat = {
        'dataset': name,
        'num_nodes': len(nodes),
        'num_edges_undirected': undirected_edges,
        'connected': np.isfinite(diameter(nodes, neighbors)),
        'avg_degree': avg_degree,
        'diameter': diameter(nodes, neighbors),
        'mean_S': float(np.mean(scores['S'][nodes])),
        'mean_Q': float(np.mean(scores['Q'][nodes])),
        **diag,
    }
    return stat


def main():
    data_root = Path('/root/yanyijin/STdiff/Holidiff/data')
    jobs = [
        {'source': 'PEMS03', 'sizes': [40], 'flow_csv': data_root / 'PEMS03' / 'warehouse' / 'clean.csv'},
        {'source': 'PEMS04', 'sizes': [40], 'flow_csv': data_root / 'PEMS04' / 'warehouse' / 'clean.csv'},
        {'source': 'PEMS07', 'sizes': [60, 100], 'npz_path': data_root / 'PEMS07' / 'PEMS07.npz'},
    ]

    all_stats = []
    for job in jobs:
        source = job['source']
        adj_path = data_root / source / f'{source}.csv'
        print(f'==== {source} ====', flush=True)
        if 'npz_path' in job:
            print(f'Loading flow: {job["npz_path"]}', flush=True)
            raw, node_ids = load_flow(job['npz_path'])
        else:
            print(f'Loading flow: {job["flow_csv"]}', flush=True)
            raw, node_ids = load_flow_from_clean_csv(job['flow_csv'])
        print(f'Flow shape: {raw.shape}', flush=True)

        clean, valid_mask = interpolate_invalid(raw)
        neighbors = load_graph(adj_path, node_ids)
        scores = compute_node_scores(raw, clean, valid_mask)
        anchor, selected = select_nested_nodes(scores, neighbors, target_sizes=tuple(job['sizes']))
        print(f'Anchor: {anchor}, S={scores["S"][anchor]:.6f}, Q*={scores["Q_star"][anchor]:.6f}', flush=True)

        stats = []
        for size in job['sizes']:
            name = f'{source}-Local{size}'
            nodes = selected[:size]
            stat = write_local_dataset(name, data_root, raw, clean, valid_mask, nodes, neighbors, scores, node_ids)
            stats.append(stat)
            all_stats.append(stat)
            print(stat, flush=True)

        stats_df = pd.DataFrame(stats)
        stats_path = data_root / source / 'local_benchmark_stats.csv'
        stats_df.to_csv(stats_path, index=False)
        print(f'Stats saved to {stats_path}', flush=True)

    all_stats_path = data_root / 'pems_local_benchmark_stats.csv'
    pd.DataFrame(all_stats).to_csv(all_stats_path, index=False)
    print(f'All stats saved to {all_stats_path}', flush=True)


if __name__ == '__main__':
    main()

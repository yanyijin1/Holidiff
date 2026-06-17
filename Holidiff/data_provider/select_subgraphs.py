from collections import deque, defaultdict
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



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


def ego_connected_nodes(neighbors, anchor, size):
    comp = connected_component(neighbors, anchor)
    if len(comp) < size:
        return None

    dist = graph_distances(neighbors, anchor)
    degree = np.asarray([len(nbrs) for nbrs in neighbors], dtype=np.float64)
    candidates = [v for v in comp if np.isfinite(dist[v])]
    candidates = sorted(candidates, key=lambda v: (dist[v], -degree[v], v))
    nodes = candidates[:size]
    return np.asarray(sorted(nodes), dtype=np.int64)


def induced_edge_list(neighbors, nodes):
    node_to_local = {node: i for i, node in enumerate(nodes)}
    rows = []
    for u in nodes:
        for v in neighbors[u]:
            if v in node_to_local:
                rows.append({'from': node_to_local[u], 'to': node_to_local[v], 'cost': 1.0})
    return pd.DataFrame(rows).drop_duplicates().sort_values(['from', 'to']).reset_index(drop=True)


def induced_neighbors(neighbors, nodes):
    node_to_local = {node: i for i, node in enumerate(nodes)}
    local_neighbors = [set() for _ in range(len(nodes))]
    for u in nodes:
        lu = node_to_local[u]
        for v in neighbors[u]:
            if v in node_to_local:
                lv = node_to_local[v]
                if lu != lv:
                    local_neighbors[lu].add(lv)
    return local_neighbors


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


def make_windows(data, seq_len=96, pred_len=12, stride=12):
    xs, ys, starts = [], [], []
    total = data.shape[0] - seq_len - pred_len + 1
    for s in range(0, max(total, 0), stride):
        e = s + seq_len
        p = e + pred_len
        xs.append(data[s:e])
        ys.append(data[e:p])
        starts.append(s)
    if len(xs) == 0:
        return None, None, None
    return np.stack(xs, axis=0), np.stack(ys, axis=0), np.asarray(starts, dtype=np.int64)


def split_train_val_test(data, valid_mask=None, train_ratio=0.7, val_ratio=0.1):
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    out = {'train': data[:train_end], 'val': data[train_end:val_end], 'test': data[val_end:]}
    if valid_mask is not None:
        out_valid = {'train': valid_mask[:train_end], 'val': valid_mask[train_end:val_end], 'test': valid_mask[val_end:]}
        return out, out_valid
    return out, None


def low_var_window_ratio(x_hist, eps=1e-3):
    hist_std = x_hist.std(axis=1)
    return float(np.mean(np.any(hist_std < eps, axis=1)))


def rmax_diagnostic(x_hist, y_future):
    mu = x_hist.mean(axis=1, keepdims=True)
    std = x_hist.std(axis=1, keepdims=True) + 1e-5
    r = np.abs((y_future - mu) / std)
    rb = np.max(r, axis=(1, 2))
    return {
        'R_mean': float(np.mean(rb)),
        'R95': float(np.percentile(rb, 95)),
        'R99': float(np.percentile(rb, 99)),
        'Rmax': float(np.max(rb)),
    }


def discretize_by_quantiles(x, q=(33.3, 66.7)):
    x = np.asarray(x, dtype=np.float64)
    if np.nanstd(x) < 1e-12:
        return np.zeros_like(x, dtype=np.int64)
    qs = np.nanpercentile(x, q)
    return np.digitize(x, qs).astype(np.int64)


def conditional_state_ambiguity(x_hist, y_future, starts, thresholds, day_len=288, tod_bucket=12, recent_len=12, min_bucket_count=8):
    B, L, N = x_hist.shape
    x_state = (x_hist > thresholds[None, None, :]).astype(np.float32)
    y_state = (y_future > thresholds[None, None, :]).astype(np.float32)
    future_start = starts + L
    tod = future_start % day_len
    tod_bin = (tod // tod_bucket).astype(np.int64)
    recent_state_ratio = x_state[:, -recent_len:, :].mean(axis=(1, 2))
    recent_state_bin = np.digitize(recent_state_ratio, [0.25, 0.50, 0.75]).astype(np.int64)
    recent = x_hist[:, -recent_len:, :]
    slope = recent[:, -1, :].mean(axis=1) - recent[:, 0, :].mean(axis=1)
    slope_bin = discretize_by_quantiles(slope)
    future_state_ratio = y_state.mean(axis=(1, 2))
    buckets = defaultdict(list)
    for i in range(B):
        key = (int(tod_bin[i]), int(recent_state_bin[i]), int(slope_bin[i]))
        buckets[key].append(float(future_state_ratio[i]))
    total_weight = 0.0
    ambiguity = 0.0
    separation = 0.0
    effective_buckets = 0
    for vals in buckets.values():
        vals = np.asarray(vals, dtype=np.float64)
        if len(vals) < min_bucket_count:
            continue
        w = len(vals)
        total_weight += w
        effective_buckets += 1
        ambiguity += w * float(np.var(vals))
        separation += w * float(np.percentile(vals, 90) - np.percentile(vals, 10))
    if total_weight <= 0:
        return {'ConditionalStateAmbiguity': 0.0, 'FutureModeSeparation': 0.0, 'effective_bucket_count': 0}
    return {
        'ConditionalStateAmbiguity': float(ambiguity / total_weight),
        'FutureModeSeparation': float(separation / total_weight),
        'effective_bucket_count': int(effective_buckets),
    }


def state_balance(y_future, thresholds):
    y_state = (y_future > thresholds[None, None, :]).astype(np.float32)
    r = float(y_state.mean())
    return float(max(0.0, 1.0 - 2.0 * abs(r - 0.5)))


def spatial_state_disagreement(y_future, thresholds, local_neighbors):
    y_state = (y_future > thresholds[None, None, :]).astype(np.float32)
    vals = []
    visited = set()
    for i in range(len(local_neighbors)):
        for j in local_neighbors[i]:
            key = tuple(sorted((i, j)))
            if key in visited:
                continue
            visited.add(key)
            vals.append(np.mean(np.abs(y_state[:, :, i] - y_state[:, :, j])))
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def compute_diffusion_potential(train_data, train_valid, local_neighbors, seq_len=96, pred_len=12, stride=12, day_len=288):
    x_hist, y_future, starts = make_windows(train_data, seq_len=seq_len, pred_len=pred_len, stride=stride)
    if x_hist is None:
        return None
    thresholds = np.percentile(train_data, 75, axis=0)
    amb = conditional_state_ambiguity(x_hist=x_hist, y_future=y_future, starts=starts, thresholds=thresholds, day_len=day_len, tod_bucket=12, recent_len=12, min_bucket_count=8)
    balance = state_balance(y_future, thresholds)
    spatial_dis = spatial_state_disagreement(y_future, thresholds, local_neighbors)
    q = {'mean_valid_rate': float(train_valid.mean()), 'low_var_window_ratio': low_var_window_ratio(x_hist, eps=1e-3), **rmax_diagnostic(x_hist, y_future)}
    amb_scaled = float(np.clip(amb['ConditionalStateAmbiguity'] / 0.25, 0.0, 1.0))
    sep_scaled = float(np.clip(amb['FutureModeSeparation'], 0.0, 1.0))
    balance_scaled = float(np.clip(balance, 0.0, 1.0))
    spatial_scaled = float(np.clip(spatial_dis, 0.0, 1.0))
    diffusion_potential = 0.40 * amb_scaled + 0.25 * sep_scaled + 0.20 * balance_scaled + 0.15 * spatial_scaled
    return {
        **amb,
        'StateBalance': balance,
        'SpatialStateDisagreement': spatial_dis,
        'DiffusionPotential': float(diffusion_potential),
        **q,
    }


def evaluate_candidate(clean, valid_mask, neighbors, nodes, seq_len=96, pred_len=12, stride=12, train_ratio=0.7, val_ratio=0.1, day_len=288):
    local_data = clean[:, nodes]
    local_valid = valid_mask[:, nodes]
    local_neighbors = induced_neighbors(neighbors, nodes)
    splits, valid_splits = split_train_val_test(local_data, local_valid, train_ratio=train_ratio, val_ratio=val_ratio)
    train_data = splits['train']
    train_valid = valid_splits['train']
    diag = compute_diffusion_potential(train_data=train_data, train_valid=train_valid, local_neighbors=local_neighbors, seq_len=seq_len, pred_len=pred_len, stride=stride, day_len=day_len)
    if diag is None:
        return None
    edge_df = induced_edge_list(neighbors, nodes)
    undirected_edges = len(edge_df) // 2
    return {
        'num_nodes': int(len(nodes)),
        'num_edges_undirected': int(undirected_edges),
        'avg_degree': float(2.0 * undirected_edges / max(len(nodes), 1)),
        'diameter': float(diameter(nodes, neighbors)),
        **diag,
        'nodes': ','.join(map(str, nodes.tolist())),
    }


def generate_ego_candidates(clean, valid_mask, neighbors, sizes=(30,), seq_len=96, pred_len=12, stride=12, train_ratio=0.7, val_ratio=0.1, day_len=288, max_r=30.0, max_low_var=0.20, min_valid=0.98):
    rows = []
    seen = set()
    for size in sizes:
        for anchor in range(len(neighbors)):
            nodes = ego_connected_nodes(neighbors, anchor, size)
            if nodes is None:
                continue
            key = (size, tuple(nodes.tolist()))
            if key in seen:
                continue
            seen.add(key)
            stat = evaluate_candidate(clean=clean, valid_mask=valid_mask, neighbors=neighbors, nodes=nodes, seq_len=seq_len, pred_len=pred_len, stride=stride, train_ratio=train_ratio, val_ratio=val_ratio, day_len=day_len)
            if stat is None:
                continue
            stat['anchor'] = int(anchor)
            stat['size'] = int(size)
            quality_ok = stat['mean_valid_rate'] >= min_valid and stat['low_var_window_ratio'] <= max_low_var and stat['Rmax'] <= max_r and np.isfinite(stat['diameter'])
            stat['quality_ok'] = bool(quality_ok)
            rows.append(stat)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df['RankScore'] = df['DiffusionPotential']
    df.loc[~df['quality_ok'], 'RankScore'] = -1.0
    df = df.sort_values(['size', 'RankScore', 'ConditionalStateAmbiguity', 'FutureModeSeparation'], ascending=[True, False, False, False]).reset_index(drop=True)
    return df


def slice_head_fraction(data, fraction):
    fraction = float(fraction)
    if not (0 < fraction <= 1.0):
        raise ValueError(f'data_fraction must be in (0, 1], got {fraction}')
    keep_len = max(1, int(len(data) * fraction))
    return data[:keep_len]


def parse_nodes(nodes_str):
    return [int(x) for x in str(nodes_str).split(',') if str(x).strip() != '']


def node_overlap_ratio(nodes_a, nodes_b):
    a = set(map(int, nodes_a))
    b = set(map(int, nodes_b))
    return len(a & b) / max(1, min(len(a), len(b)))


def select_diverse_quality_candidates(df, size=30, num_regions=5, overlap_threshold=0.5, seed=2026):
    rng = np.random.default_rng(seed)

    cand = df[(df['quality_ok']) & (df['size'] == size)].copy()
    if len(cand) == 0:
        raise ValueError(f'No quality candidates found for size={size}')

    cand = cand.sort_values('RankScore', ascending=True).reset_index(drop=True)

    bins = np.array_split(cand.index.to_numpy(), num_regions)

    selected_rows = []
    selected_nodes = []

    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue

        sub = cand.loc[bin_idx].copy()
        sub['bin_center_dist'] = np.abs(sub['RankScore'] - sub['RankScore'].median())
        sub = sub.sort_values(['bin_center_dist', 'RankScore'], ascending=[True, False])

        top_pool = sub.head(min(10, len(sub))).sample(
            frac=1.0,
            random_state=int(rng.integers(0, 1_000_000)),
        )

        chosen = None
        for _, row in top_pool.iterrows():
            nodes = parse_nodes(row['nodes'])
            if all(node_overlap_ratio(nodes, prev) <= overlap_threshold for prev in selected_nodes):
                chosen = row
                selected_nodes.append(nodes)
                break

        if chosen is None:
            best_row = None
            best_overlap = float('inf')
            for _, row in sub.iterrows():
                nodes = parse_nodes(row['nodes'])
                overlap = max(
                    [node_overlap_ratio(nodes, prev) for prev in selected_nodes],
                    default=0.0,
                )
                if overlap < best_overlap:
                    best_overlap = overlap
                    best_row = row

            chosen = best_row
            selected_nodes.append(parse_nodes(chosen['nodes']))

        selected_rows.append(chosen)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    if len(selected) < num_regions:
        selected_node_sets = [parse_nodes(row['nodes']) for _, row in selected.iterrows()]
        selected_keys = set(selected['nodes'].astype(str).tolist())

        remaining = cand[~cand['nodes'].astype(str).isin(selected_keys)].copy()
        remaining = remaining.sample(frac=1.0, random_state=seed)

        for _, row in remaining.iterrows():
            nodes = parse_nodes(row['nodes'])
            if all(node_overlap_ratio(nodes, prev) <= overlap_threshold for prev in selected_node_sets):
                selected = pd.concat([selected, row.to_frame().T], ignore_index=True)
                selected_node_sets.append(nodes)

            if len(selected) >= num_regions:
                break

    if len(selected) < num_regions:
        raise RuntimeError(
            f'Only selected {len(selected)} regions, fewer than required {num_regions}. '
            f'Try increasing overlap_threshold or relaxing quality filters.'
        )

    return selected.head(num_regions).reset_index(drop=True)


def write_local_dataset(name, out_root, clean, nodes, neighbors, source_node_ids, stats=None, data_fraction=1.0):
    if abs(float(data_fraction) - 1.0) > 1e-8:
        raise ValueError(
            'For PeMS applicability-boundary experiments, subgraph generation '
            'must use full time series. Please keep data_fraction=1.0 and use '
            '--history_ratio in training.'
        )

    out_dir = out_root / name
    warehouse = out_dir / 'warehouse'
    warehouse.mkdir(parents=True, exist_ok=True)
    local_data = clean[:, nodes]
    time_index = pd.date_range('2000-01-01 00:00:00', periods=local_data.shape[0], freq='5min')
    rows = []
    for local_idx in range(len(nodes)):
        rows.append(pd.DataFrame({'time_slot': time_index, 'station_index': local_idx, 'traffic_flow': local_data[:, local_idx], 'is_holiday': 0.0}))
    df = pd.concat(rows, ignore_index=True).sort_values(['time_slot', 'station_index'])
    df.to_csv(warehouse / 'clean.csv', index=False)
    edge_df = induced_edge_list(neighbors, nodes)
    edge_df.to_csv(out_dir / f'{name}.csv', index=False)
    pd.DataFrame({'local_index': np.arange(len(nodes)), 'source_col': nodes, 'source_node': np.asarray(source_node_ids)[nodes]}).to_csv(out_dir / 'node_mapping.csv', index=False)
    if stats is not None:
        stats = dict(stats)
        stats['data_fraction'] = float(data_fraction)
        stats['data_fraction_anchor'] = 'full_series'
        stats['written_timesteps'] = int(local_data.shape[0])
        with open(out_dir / 'candidate_diagnostics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    return out_dir


def write_selected_candidates(selected_df, output_root, source, clean, neighbors, node_ids, name_prefix='R30'):
    written = []

    for rank, (_, row) in enumerate(selected_df.iterrows(), start=1):
        nodes = np.asarray(parse_nodes(row['nodes']), dtype=np.int64)
        name = f'{source}-{name_prefix}-{rank:03d}'

        stats = row.to_dict()
        out_dir = write_local_dataset(
            name=name,
            out_root=output_root,
            clean=clean,
            nodes=nodes,
            neighbors=neighbors,
            source_node_ids=node_ids,
            stats=stats,
            data_fraction=1.0,
        )

        written.append({
            'dataset': name,
            'path': str(out_dir),
            'size': int(row['size']),
            'rank': int(rank),
            'anchor': int(row['anchor']) if 'anchor' in row else -1,
            'RankScore': float(row['RankScore']),
            'ConditionalStateAmbiguity': float(row['ConditionalStateAmbiguity']),
            'FutureModeSeparation': float(row['FutureModeSeparation']),
            'StateBalance': float(row['StateBalance']),
            'SpatialStateDisagreement': float(row['SpatialStateDisagreement']),
            'Rmax': float(row['Rmax']),
            'R95': float(row['R95']),
            'R99': float(row['R99']),
            'low_var_window_ratio': float(row['low_var_window_ratio']),
            'mean_valid_rate': float(row['mean_valid_rate']),
            'diameter': float(row['diameter']),
            'avg_degree': float(row['avg_degree']),
            'nodes': row['nodes'],
        })

    return pd.DataFrame(written)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/STdiff_data/datasets')
    parser.add_argument('--output_root', type=str, default='/root/autodl-tmp/STdiff_data/datasets')
    parser.add_argument('--candidate_root', type=str, default='/root/autodl-tmp/STdiff_data/candidates')
    parser.add_argument('--source', type=str, default='PEMS08')
    parser.add_argument('--sizes', type=int, nargs='+', default=[30])
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--day_len', type=int, default=288)
    parser.add_argument('--max_r', type=float, default=30.0)
    parser.add_argument('--max_low_var', type=float, default=0.20)
    parser.add_argument('--min_valid', type=float, default=0.98)
    parser.add_argument('--num_regions', type=int, default=5)
    parser.add_argument('--overlap_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--name_prefix', type=str, default='R30')
    parser.add_argument('--data_fraction', type=float, default=1.0)
    args = parser.parse_args()

    if set(args.sizes) != {30}:
        raise ValueError('This script is restricted to 30-node connected subgraphs. Please use --sizes 30.')
    if abs(float(args.data_fraction) - 1.0) > 1e-8:
        raise ValueError(
            'Subgraph generation must use the full time series. '
            'Keep --data_fraction 1.0 and use --history_ratio during training.'
        )

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    candidate_root = Path(args.candidate_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_root.mkdir(parents=True, exist_ok=True)
    source = args.source
    source_dir = data_root / source
    adj_path = source_dir / f'{source}.csv'
    flow_csv = source_dir / 'warehouse' / 'clean.csv'
    npz_path = source_dir / f'{source}.npz'

    print(f'==== {source} ====', flush=True)
    if flow_csv.exists():
        print(f'Loading flow from clean csv: {flow_csv}', flush=True)
        raw, node_ids = load_flow_from_clean_csv(flow_csv)
    elif npz_path.exists():
        print(f'Loading flow from npz: {npz_path}', flush=True)
        raw, node_ids = load_flow(npz_path)
    else:
        raise FileNotFoundError(f'Cannot find flow file under {source_dir}')
    print(f'Flow shape: {raw.shape}', flush=True)

    clean, valid_mask = interpolate_invalid(raw)
    neighbors = load_graph(adj_path, node_ids)
    print('Generating deterministic connected ego-subgraph candidates...', flush=True)
    df = generate_ego_candidates(
        clean=clean,
        valid_mask=valid_mask,
        neighbors=neighbors,
        sizes=tuple(args.sizes),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        day_len=args.day_len,
        max_r=args.max_r,
        max_low_var=args.max_low_var,
        min_valid=args.min_valid,
    )
    if len(df) == 0:
        print('No candidates generated.', flush=True)
        return

    out_summary = candidate_root / f'{source}_subgraph_candidates.csv'
    df.to_csv(out_summary, index=False)
    print(f'Candidate summary saved to: {out_summary}', flush=True)

    selected_df = select_diverse_quality_candidates(
        df=df,
        size=30,
        num_regions=args.num_regions,
        overlap_threshold=args.overlap_threshold,
        seed=args.seed,
    )

    selected_summary = candidate_root / f'{source}_selected_R30_subgraphs.csv'
    selected_df.to_csv(selected_summary, index=False)
    print(f'Selected R30 summary saved to: {selected_summary}', flush=True)

    cols = [
        'anchor', 'size', 'RankScore',
        'ConditionalStateAmbiguity', 'FutureModeSeparation',
        'StateBalance', 'SpatialStateDisagreement',
        'Rmax', 'R95', 'R99',
        'low_var_window_ratio', 'mean_valid_rate',
        'diameter', 'avg_degree',
    ]
    print(selected_df[cols].to_string(index=False), flush=True)

    written_df = write_selected_candidates(
        selected_df=selected_df,
        output_root=output_root,
        source=source,
        clean=clean,
        neighbors=neighbors,
        node_ids=node_ids,
        name_prefix=args.name_prefix,
    )

    out_written = candidate_root / f'{source}_written_R30_subgraphs.csv'
    written_df.to_csv(out_written, index=False)
    print(f'Written R30 datasets saved to: {out_written}', flush=True)
    print(written_df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

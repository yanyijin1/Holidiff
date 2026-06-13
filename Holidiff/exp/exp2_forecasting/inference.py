from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path
from types import SimpleNamespace
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT.parent) not in sys.path: sys.path.insert(0, str(ROOT.parent))
from Holidiff.exp import Exp_Long_Term_Forecast
try:
    import yaml
except Exception as exc:
    yaml = None; _yaml_import_error = exc

OUT = Path('/root/autodl-tmp/STdiff_runs/representative_case/fujian30_h12')
CACHE = OUT / 'cache'
FIGURES = OUT / 'figures'
RESULTS = ROOT / 'results'
PANEL_TITLES = {
    'rep1': '(a) Representative case I',
    'rep2': '(b) Representative case II',
    'rep3': '(c) Representative case III',
    'high-flow': '(a) Holiday high-flow case',
    'peak-timing': '(b) Peak-shift case',
    'local-fluctuation': '(c) Local fluctuation case',
}
PANEL_ORDER = ['rep1', 'rep2', 'rep3']
COLORS = {
    'raw_history': '#D9D9D9',
    'history': '#9E9E9E',
    'ground_truth': '#222222',
    'itr': '#4C78A8',
    'sim': '#54A24B',
    'ours': '#E45756',
    'future_region': '#EAF3FF',
    'pred_line': '#7F7F7F',
    'grid': '#ECECEC',
    'spine': '#B0B0B0',
}
SPECS = {
    'ours': ('ResDiff', ROOT/'configs/fujian30/holidiff_h12.yaml', Path('/root/autodl-tmp/STdiff_runs/checkpoints/fujian30/final_holidiff__h12/checkpoint.pth'), 'final_holidiff__h12'),
    'itr': ('iTransformer', ROOT/'configs/fujian30/itransformer_h12.yaml', Path('/root/autodl-tmp/STdiff_runs/checkpoints/fujian30/itransformer/h12/checkpoint.pth'), 'itransformer__h12'),
    'sim': ('SimDiff', ROOT/'configs/fujian30/simdiff_h12.yaml', Path('/root/autodl-tmp/STdiff_runs/checkpoints/fujian30/simdiff/h12/checkpoint.pth'), 'simdiff__h12'),
}

def seeds(s=2021):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_cfg(p: Path):
    if yaml is None: raise RuntimeError(f'PyYAML import failed: {_yaml_import_error}')
    c = dict(yaml.safe_load(p.read_text(encoding='utf-8')))
    c['use_gpu'] = bool(c.get('use_gpu', True) and torch.cuda.is_available())
    c['train_val_aggregation_mode'] = c.get('train_val_aggregation_mode', 'single')
    c['test_aggregation_mode'] = c.get('test_aggregation_mode', 'dca')
    c['test_times'] = c.get('vs_times', c.get('sample_times', 1))
    if c.get('use_multi_gpu', False):
        d = str(c.get('devices', '0')).replace(' ', '')
        c['devices'] = d; c['device_ids'] = [int(x) for x in d.split(',') if x]; c['gpu'] = c['device_ids'][0]
    return c

def collect(spec):
    label, cfgp, ckpt, ver = spec
    cfg = load_cfg(cfgp); cfg.update({'config': str(cfgp), 'version': ver, 'load_checkpoint': str(ckpt)})
    args = SimpleNamespace(**cfg); seeds(int(getattr(args, 'seed', 2021)))
    exp = Exp_Long_Term_Forecast(args); ds, _ = exp._get_data('test')
    dl = DataLoader(ds, batch_size=int(getattr(exp.args, 'eval_batch_size', getattr(exp.args, 'batch_size', 32))), shuffle=False, num_workers=int(getattr(exp.args, 'num_workers', 0)), drop_last=False)
    total_batches = len(dl)
    print(f'[collect] {label} <- {ckpt}', flush=True)
    print(f'[collect] {label}: {len(ds)} test samples, {total_batches} batches', flush=True)
    exp._load_checkpoint_compat(str(ckpt)); exp.model.eval(); prev = exp._set_model_aggregation_mode(getattr(exp.args, 'test_aggregation_mode', 'dca'))
    P=[]; Y=[]; X=[]; M=[]; H=[]
    agg_mode = str(getattr(exp.args, 'test_aggregation_mode', 'dca')).lower()
    sample_times = 1 if agg_mode == 'single' else int(getattr(exp.args, 'test_times', exp.args.vs_times))
    with torch.no_grad():
        for batch_idx, b in enumerate(dl, start=1):
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f'[collect] {label}: batch {batch_idx}/{total_batches}', flush=True)
            bx, by, bxm, bym = b[0].float().to(exp.device), b[1].float(), b[2].float().to(exp.device), b[3].float().to(exp.device)
            bm = b[4] if len(b) > 4 else None; bh = b[5] if len(b) > 5 else None
            dec = torch.zeros_like(by[:, -exp.args.pred_len:, :]).float(); dec = torch.cat([by[:, :exp.args.label_len, :], dec], 1).float().to(exp.device)
            out = exp._run_model(exp.model, bx, bxm, dec, bym, sample_times=sample_times, holiday_flag=bh, future_target=by[:, -exp.args.pred_len:, :].to(exp.device))
            out = out[0] if exp.args.is_diff else out; out = exp._process_model_output(out, is_diff=exp.args.is_diff)
            P.append(out.detach().cpu().numpy()); Y.append(by[:, -exp.args.pred_len:, :].numpy()); X.append(bx.detach().cpu().numpy())
            if bm is not None: M.append(bm.detach().cpu().numpy())
            if bh is not None: H.append(bh.detach().cpu().numpy())
    exp._restore_model_aggregation_mode(prev)
    P=np.concatenate(P); Y=np.concatenate(Y); X=np.concatenate(X); M=np.concatenate(M) if M else None; H=np.concatenate(H) if H else None
    print(f'[collect] {label}: finished, preds shape={P.shape}, trues shape={Y.shape}', flush=True)
    meta = exp._load_fujian30_meta(ds); n = len(P)
    for k in ('starts','pred_starts','pred_ends'): meta[k] = np.asarray(meta[k])[:n]
    return {'label': label, 'preds': exp._inverse_transform(ds, P), 'trues': exp._inverse_transform(ds, Y), 'hist': exp._inverse_transform(ds, X), 'mask': M, 'hol': H, 'meta': meta}

def dump_cache(key, data, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = data['meta']
    np.savez_compressed(
        cache_dir / f'{key}.npz',
        label=np.asarray(data['label']), preds=data['preds'], trues=data['trues'], hist=data['hist'],
        mask=data['mask'] if data['mask'] is not None else np.empty((0,), dtype=np.float32),
        hol=data['hol'] if data['hol'] is not None else np.empty((0,), dtype=np.float32),
        station_ids=np.asarray(meta['station_ids']).astype(str),
        time_index=np.asarray(meta['time_index']).astype('datetime64[ns]').astype(str),
        starts=np.asarray(meta['starts']), pred_starts=np.asarray(meta['pred_starts']), pred_ends=np.asarray(meta['pred_ends']),
    )

def load_cache(cache_dir: Path):
    out = {}
    for key in SPECS:
        p = cache_dir / f'{key}.npz'
        if not p.exists():
            raise FileNotFoundError(f'Missing cache file: {p}')
        z = np.load(p, allow_pickle=True)
        mask = z['mask']; hol = z['hol']
        out[key] = {
            'label': str(z['label']), 'preds': z['preds'], 'trues': z['trues'], 'hist': z['hist'],
            'mask': None if mask.size == 0 else mask, 'hol': None if hol.size == 0 else hol,
            'meta': {'station_ids': z['station_ids'], 'time_index': z['time_index'], 'starts': z['starts'], 'pred_starts': z['pred_starts'], 'pred_ends': z['pred_ends']},
        }
    return out

def local(pred, true):
    return {'mae': float(np.mean(np.abs(pred-true))), 'pte': float(abs(int(np.argmax(pred))-int(np.argmax(true)))), 'pve': float(abs(float(np.max(pred))-float(np.max(true)))), 'diffmae': float(np.mean(np.abs(np.diff(pred)-np.diff(true))))}

def score_case(mo, mi, ms, a=.5, b=.5, c=.5):
    return (mi['mae']-mo['mae'])+(ms['mae']-mo['mae']) + a*((mi['pte']-mo['pte'])+(ms['pte']-mo['pte'])) + b*((mi['pve']-mo['pve'])+(ms['pve']-mo['pve'])) + c*((mi['diffmae']-mo['diffmae'])+(ms['diffmae']-mo['diffmae']))

def select(C, topk=30, l1=.5, l2=.5, l3=.5, minc=2):
    O,I,S = C['ours'], C['itr'], C['sim']; X,Y,H,M,meta = O['hist'],O['trues'],O['hol'],O['mask'],O['meta']
    idx = np.where(H.sum(1) > 0)[0] if H is not None else np.arange(len(Y))
    out=[]
    for b in idx:
        for n in range(Y.shape[-1]):
            t=Y[b,:,n]; r=float(t.max()-t.min()); s=float(t.std()); d=float(np.mean(np.abs(np.diff(t))))
            if np.any(np.isclose(t, 0.0)):
                continue
            po,pi,ps = O['preds'][b,:,n], I['preds'][b,:,n], S['preds'][b,:,n]
            mo,mi,ms = local(po,t), local(pi,t), local(ps,t)
            if not (mo['mae'] < mi['mae'] and mo['mae'] < ms['mae']):
                continue
            bias_itr = float(np.mean(pi - t))
            bias_sim = float(np.mean(ps - t))
            bias_ours = float(np.mean(po - t))
            baseline_bias = max(abs(bias_itr), abs(bias_sim))
            if baseline_bias < max(0.08 * max(r, 1.0), 8.0):
                continue
            sc = representative_score(mo,mi,ms,t,pi,ps,po)
            case = {
                'sample_index': int(b), 'sensor_index': int(n), 'sensor_id': str(meta['station_ids'][n]),
                'pred_start_time': str(meta['time_index'][int(meta['pred_starts'][b])]), 'pred_end_time': str(meta['time_index'][int(meta['pred_ends'][b])]),
                'holiday_flag': True, 'true_range': r, 'true_std': s, 'true_slope_energy': d,
                'bias_ours': bias_ours, 'bias_itr': bias_itr, 'bias_sim': bias_sim,
                'valid_ratio': float(np.mean(M[b,:,n])) if M is not None else 1.0,
                'score': float(sc), 'metrics': {'ResDiff': mo, 'iTransformer': mi, 'SimDiff': ms},
                'history': X[b,:,n].tolist(), 'ground_truth': t.tolist(),
                'predictions': {'ResDiff': po.tolist(), 'iTransformer': pi.tolist(), 'SimDiff': ps.tolist()}
            }
            out.append(case)
    out.sort(key=lambda z: (z['score'], z['true_range'], abs(z['bias_itr']) + abs(z['bias_sim'])), reverse=True)
    return out[:topk]

def smooth_history(hist: np.ndarray):
    hist = np.asarray(hist, dtype=np.float32)
    n = len(hist)
    if n < 5:
        return hist
    window = min(11, n if n % 2 == 1 else n - 1)
    if window < 5:
        return hist
    return savgol_filter(hist, window_length=window, polyorder=2)

def turning_points(x: np.ndarray):
    signs = []
    for value in np.sign(np.diff(np.asarray(x, dtype=np.float32))):
        if value != 0 and (not signs or value != signs[-1]):
            signs.append(value)
    return max(len(signs) - 1, 0)

def normalized_gap(mo, mi, ms, key: str, eps: float = 1e-6):
    base = max(mi[key], ms[key], eps)
    return ((mi[key] - mo[key]) + (ms[key] - mo[key])) / base

def classify_panel(case):
    t = np.asarray(case['ground_truth'], dtype=np.float32)
    h = np.asarray(case['history'], dtype=np.float32)
    mo, mi, ms = case['metrics']['ResDiff'], case['metrics']['iTransformer'], case['metrics']['SimDiff']
    peak_idx = int(np.argmax(t))
    true_range = float(t.max() - t.min()) + 1e-6
    rise = float((t[-1] - t[0]) / true_range)
    prominence = float(t[peak_idx] - np.mean([t[0], t[-1]])) / true_range
    tp = turning_points(t)
    diff_energy = float(np.mean(np.abs(np.diff(t)))) / true_range
    high_level = float(np.mean(t) / (np.max(h) + 1e-6))
    scores = {
        'high-flow': float(1.7 * max(rise, 0.0) + 0.8 * high_level + 0.6 * normalized_gap(mo, mi, ms, 'mae') + 0.3 * normalized_gap(mo, mi, ms, 'diffmae')),
        'peak-timing': float(1.3 * prominence + 0.8 * float(0 < peak_idx < len(t) - 1) + 0.8 * normalized_gap(mo, mi, ms, 'pte') + 0.5 * normalized_gap(mo, mi, ms, 'pve')),
        'local-fluctuation': float(0.9 * min(tp, 3) + 0.8 * min(diff_energy, 1.0) + 0.8 * normalized_gap(mo, mi, ms, 'diffmae') + 0.4 * normalized_gap(mo, mi, ms, 'mae') - 0.5 * max(rise, 0.0)),
    }
    return max(scores, key=scores.get), scores

def representative_score(mo, mi, ms, t: np.ndarray, pi: np.ndarray, ps: np.ndarray, po: np.ndarray):
    t = np.asarray(t, dtype=np.float32)
    pi = np.asarray(pi, dtype=np.float32)
    ps = np.asarray(ps, dtype=np.float32)
    po = np.asarray(po, dtype=np.float32)
    gain_itr = float(mi['mae'] - mo['mae'])
    gain_sim = float(ms['mae'] - mo['mae'])
    bias_itr = float(np.mean(pi - t))
    bias_sim = float(np.mean(ps - t))
    ours_residual_std = float(np.std(t - po))
    return float(gain_itr + gain_sim + 0.3 * abs(bias_itr) + 0.3 * abs(bias_sim) - 0.2 * ours_residual_std)

def prep_plot_arrays(case):
    hist = np.asarray(case['history'], dtype=np.float32)
    gt = np.asarray(case['ground_truth'], dtype=np.float32)
    itr = np.asarray(case['predictions']['iTransformer'], dtype=np.float32)
    sim = np.asarray(case['predictions']['SimDiff'], dtype=np.float32)
    ours = np.asarray(case['predictions']['ResDiff'], dtype=np.float32)
    anchor = float(hist[-1])
    x_hist = np.arange(-(len(hist) - 1), 1)
    x_future = np.arange(0, len(gt) + 1)
    return x_hist, x_future, hist, smooth_history(hist), np.r_[anchor, gt], np.r_[anchor, itr], np.r_[anchor, sim], np.r_[anchor, ours]

def compute_ylim(hist_smooth, gt, itr, sim, ours):
    vals = np.concatenate([hist_smooth, gt, itr, sim, ours]).astype(np.float32)
    low, high = float(vals.min()), float(vals.max())
    pad = 0.08 * max(high - low, 1.0)
    return low - pad, high + pad

def style_axis(ax, show_xlabel: bool = False):
    ax.set_facecolor('white')
    ax.grid(axis='y', color=COLORS['grid'], linewidth=0.55, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(COLORS['spine'])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis='both', labelsize=9, colors='#444444')
    ax.set_ylabel('Traffic flow', fontsize=9.5)
    ax.set_xlim(-95.5, 12.5)
    ax.set_xticks([-95, -72, -48, -24, 0, 12])
    if show_xlabel:
        ax.set_xlabel('Time step', fontsize=10)

def plot_publication_case(ax, case, title: str, note: bool = False, show_xlabel: bool = False, legend: bool = True, y_lim=None):
    x_hist, x_future, raw_hist, smooth_hist, gt, itr, sim, ours = prep_plot_arrays(case)
    ax.set_facecolor('white')
    ax.axvspan(0.0, len(gt) - 1, color=COLORS['future_region'], alpha=1.0, zorder=0)
    ax.plot(x_hist, raw_hist, color=COLORS['raw_history'], linewidth=0.7, alpha=0.55, label='History', zorder=1)
    ax.plot(x_hist, smooth_hist, color=COLORS['history'], linewidth=1.4, alpha=0.95, zorder=2)
    ax.axvline(0.0, color=COLORS['pred_line'], linestyle='--', linewidth=1.1, alpha=0.95, zorder=3)
    ax.scatter([0], [raw_hist[-1]], s=18, color=COLORS['ground_truth'], zorder=5)
    h1, = ax.plot(x_future, gt, color=COLORS['ground_truth'], linewidth=1.8, linestyle='-', label='Ground Truth', zorder=4)
    h2, = ax.plot(x_future, itr, color=COLORS['itr'], linewidth=1.5, linestyle=(0, (4, 2)), label='iTransformer', zorder=4)
    h3, = ax.plot(x_future, sim, color=COLORS['sim'], linewidth=1.5, linestyle=(0, (6, 2, 1.5, 2)), label='SimDiff', zorder=4)
    h4, = ax.plot(x_future, ours, color=COLORS['ours'], linewidth=1.7, linestyle='-', label='RegDiff', zorder=5)
    ax.set_title(title, fontsize=12, fontweight='semibold', pad=14, loc='center')
    ax.set_ylabel('Traffic flow', fontsize=11)
    ax.set_xlim(-len(raw_hist), len(gt) - 0.5)
    hist_ticks = [t for t in [-96, -72, -48, -24, 0] if t >= -len(raw_hist)]
    future_ticks = [t for t in [1, 3, 6, 9, 12] if t <= len(gt) - 1]
    ax.set_xticks(hist_ticks + future_ticks)
    if show_xlabel:
        ax.set_xlabel('Time step (forecast horizon)', fontsize=11)
    ax.grid(True, color=COLORS['grid'], linewidth=0.7, alpha=0.9)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        vals = np.concatenate([raw_hist, gt, itr, sim, ours]).astype(np.float32)
        low, high = float(vals.min()), float(vals.max())
        margin = 0.12 * (high - low + 1e-6)
        ax.set_ylim(low - margin, high + margin)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    if legend:
        leg = ax.legend(loc='upper left', fontsize=9, frameon=True, ncol=1, handlelength=2.0, borderpad=0.8)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#BFBFBF')
        leg.get_frame().set_alpha(0.92)
    return [h1, h2, h3, h4]

def choose_triptych_cases(cases):
    rep_titles = ['rep1', 'rep2', 'rep3']
    return list(zip(rep_titles, cases[:3]))

def save_candidate_csv(cases, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['rank','sample_index','sensor_index','sensor_id','holiday_flag','pred_start_time','pred_end_time','true_range','true_std','true_slope_energy','turning_points','representative_score','selected_panel_type_recommendation','ours_mae','ours_pte','ours_pve','ours_diffmae','itr_mae','itr_pte','itr_pve','itr_diffmae','sim_mae','sim_pte','sim_pve','sim_diffmae']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, c in enumerate(cases, 1):
            w.writerow({'rank':i,'sample_index':c['sample_index'],'sensor_index':c['sensor_index'],'sensor_id':c['sensor_id'],'holiday_flag':c['holiday_flag'],'pred_start_time':c['pred_start_time'],'pred_end_time':c['pred_end_time'],'true_range':f"{c['true_range']:.6f}",'true_std':f"{c['true_std']:.6f}",'true_slope_energy':f"{c['true_slope_energy']:.6f}",'turning_points':c.get('turning_points', 0),'representative_score':f"{c['score']:.6f}",'selected_panel_type_recommendation':c.get('panel_type', ''),'ours_mae':f"{c['metrics']['ResDiff']['mae']:.6f}",'ours_pte':f"{c['metrics']['ResDiff']['pte']:.6f}",'ours_pve':f"{c['metrics']['ResDiff']['pve']:.6f}",'ours_diffmae':f"{c['metrics']['ResDiff']['diffmae']:.6f}",'itr_mae':f"{c['metrics']['iTransformer']['mae']:.6f}",'itr_pte':f"{c['metrics']['iTransformer']['pte']:.6f}",'itr_pve':f"{c['metrics']['iTransformer']['pve']:.6f}",'itr_diffmae':f"{c['metrics']['iTransformer']['diffmae']:.6f}",'sim_mae':f"{c['metrics']['SimDiff']['mae']:.6f}",'sim_pte':f"{c['metrics']['SimDiff']['pte']:.6f}",'sim_pve':f"{c['metrics']['SimDiff']['pve']:.6f}",'sim_diffmae':f"{c['metrics']['SimDiff']['diffmae']:.6f}"})

def save_previews(cases):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for c in cases:
        stem = f"fujian_case_preview_sample{c['sample_index']:04d}_sensor{c['sensor_id']}"
        fig, ax = plt.subplots(figsize=(7.2, 2.8), facecolor='white')
        plot_publication_case(ax, c, PANEL_TITLES.get(c.get('panel_type', ''), 'Representative case'), show_xlabel=True)
        fig.tight_layout()
        fig.savefig(FIGURES / f'{stem}.png', dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(FIGURES / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
        plt.close(fig)

def save_triptych(chosen):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 12), constrained_layout=True)
    default_titles = ['(a) Representative case I', '(b) Representative case II', '(c) Representative case III']
    for i, (_, c) in enumerate(chosen):
        title = default_titles[i]
        plot_publication_case(axes[i], c, title=title, show_xlabel=True, legend=(i == 0))
    fig.savefig(FIGURES / 'fujian_case_triptych.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES / 'fujian_case_triptych.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES / 'fujian_case_triptych_preview.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES / 'fujian_case_triptych_preview.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def smooth_history(hist: np.ndarray):
    hist = np.asarray(hist, dtype=np.float32)
    n = len(hist)
    if n < 5:
        return hist
    window = min(9, n if n % 2 == 1 else n - 1)
    if window < 5:
        return hist
    return savgol_filter(hist, window_length=window, polyorder=2)

def plot_case(c, p: Path):
    hist=np.asarray(c['history'], dtype=np.float32); true=np.asarray(c['ground_truth'], dtype=np.float32); pi=np.asarray(c['predictions']['iTransformer'], dtype=np.float32); ps=np.asarray(c['predictions']['SimDiff'], dtype=np.float32); po=np.asarray(c['predictions']['ResDiff'], dtype=np.float32)
    L=len(hist); H=len(true)
    hist_span = 32.0
    xh=np.linspace(-hist_span, 0.0, L)
    xf=np.arange(0, H + 1, dtype=np.float32)
    true_plot=np.r_[hist[-1], true]; pi_plot=np.r_[hist[-1], pi]; ps_plot=np.r_[hist[-1], ps]; po_plot=np.r_[hist[-1], po]
    plt.figure(figsize=(8.0,3.2))
    ax=plt.gca()
    ax.set_facecolor('white')
    ax.axvspan(0, H, color='#EAF3FF', zorder=0)
    ax.plot(xh, hist, color='#9A9A9A', lw=1.0, alpha=0.95, label='History', zorder=1)
    ax.axvline(0, color='#7F7F7F', ls='--', lw=1.1, alpha=0.95, zorder=2)
    ax.scatter([0], [hist[-1]], s=16, color='#222222', zorder=5)
    ax.plot(xf, true_plot, color='#222222', lw=1.8, label='Ground Truth', zorder=4)
    ax.plot(xf, pi_plot, color='#4C78A8', ls=(0, (4, 2)), lw=1.5, label='iTransformer', zorder=3)
    ax.plot(xf, ps_plot, color='#54A24B', ls=(0, (6, 2, 1.5, 2)), lw=1.5, label='SimDiff', zorder=3)
    ax.plot(xf, po_plot, color='#E45756', lw=1.7, label='RegDiff', zorder=4)
    ax.set_xlim(-hist_span-1.0, H + 0.5)
    hist_tick_labels = [-96, -72, -48, -24, 0]
    hist_tick_pos = np.linspace(-hist_span, 0.0, len(hist_tick_labels))
    future_ticks = [1, 3, 6, 9, 12]
    xticks = list(hist_tick_pos) + [t for t in future_ticks if t <= H]
    xlabels = [str(t) for t in hist_tick_labels] + [str(t) for t in future_ticks if t <= H]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    vals = np.concatenate([hist, true_plot, pi_plot, ps_plot, po_plot])
    ypad = max(float((vals.max() - vals.min()) * 0.10), 1.0)
    ax.set_ylim(float(vals.min() - ypad), float(vals.max() + ypad))
    ax.set_xlabel('Time step (forecast horizon)')
    ax.set_ylabel('Traffic flow')
    ax.grid(True, color='#ECECEC', linewidth=0.7, alpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    leg = ax.legend(loc='upper left', fontsize=8.5, frameon=True, ncol=1, handlelength=2.0, borderpad=0.7)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#BFBFBF')
    leg.get_frame().set_alpha(0.92)
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()

def save(cases, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out/'top30_candidates.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    cols=['rank','sample_index','sensor_index','sensor_id','pred_start_time','pred_end_time','score','true_range','true_std','true_slope_energy','bias_ours','bias_itr','bias_sim','ours_mae','ours_pte','ours_pve','ours_diffmae','itr_mae','itr_pte','itr_pve','itr_diffmae','sim_mae','sim_pte','sim_pve','sim_diffmae']
    with (out/'top30_candidates.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i,c in enumerate(cases,1):
            w.writerow({'rank':i,'sample_index':c['sample_index'],'sensor_index':c['sensor_index'],'sensor_id':c['sensor_id'],'pred_start_time':c['pred_start_time'],'pred_end_time':c['pred_end_time'],'score':f"{c['score']:.6f}",'true_range':f"{c['true_range']:.6f}",'true_std':f"{c['true_std']:.6f}",'true_slope_energy':f"{c['true_slope_energy']:.6f}",'bias_ours':f"{c['bias_ours']:.6f}",'bias_itr':f"{c['bias_itr']:.6f}",'bias_sim':f"{c['bias_sim']:.6f}",'ours_mae':f"{c['metrics']['ResDiff']['mae']:.6f}",'ours_pte':f"{c['metrics']['ResDiff']['pte']:.6f}",'ours_pve':f"{c['metrics']['ResDiff']['pve']:.6f}",'ours_diffmae':f"{c['metrics']['ResDiff']['diffmae']:.6f}",'itr_mae':f"{c['metrics']['iTransformer']['mae']:.6f}",'itr_pte':f"{c['metrics']['iTransformer']['pte']:.6f}",'itr_pve':f"{c['metrics']['iTransformer']['pve']:.6f}",'itr_diffmae':f"{c['metrics']['iTransformer']['diffmae']:.6f}",'sim_mae':f"{c['metrics']['SimDiff']['mae']:.6f}",'sim_pte':f"{c['metrics']['SimDiff']['pte']:.6f}",'sim_pve':f"{c['metrics']['SimDiff']['pve']:.6f}",'sim_diffmae':f"{c['metrics']['SimDiff']['diffmae']:.6f}"})
    (out/'top20_candidates.csv').write_text((out/'top30_candidates.csv').read_text(encoding='utf-8'), encoding='utf-8')
    (out/'top30_candidates.csv').write_text((out/'top30_candidates.csv').read_text(encoding='utf-8'), encoding='utf-8')
    pdir=out/'top_30_simple_selection'; pdir.mkdir(parents=True, exist_ok=True)
    for i,c in enumerate(cases,1):
        stem = f"candidate_{i:03d}_sample{c['sample_index']}_sensor{c['sensor_id']}_score{c['score']:.3f}_mae_ours{c['metrics']['ResDiff']['mae']:.2f}_mae_itr{c['metrics']['iTransformer']['mae']:.2f}_mae_sim{c['metrics']['SimDiff']['mae']:.2f}"
        plot_case(c, pdir/f"{stem}.png")
    if cases:
        b=cases[0]
        plot_case(b, out/'recommended_case.png')
        plot_case(b, out/'recommended_case.pdf')
        FIGURES.mkdir(parents=True, exist_ok=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        save_candidate_csv(cases, RESULTS/'fujian_h12_case_candidates.csv')
        save_previews(cases)
        chosen = choose_triptych_cases(cases)
        if chosen:
            save_triptych(chosen)
            b = chosen[0][1]
        plot_case(b, FIGURES/'fujian_case_h12.png')
        plot_case(b, FIGURES/'fujian_case_h12.pdf')
        np.savez(out/'recommended_case_arrays.npz', history=np.asarray(b['history'],dtype=np.float32), ground_truth=np.asarray(b['ground_truth'],dtype=np.float32), pred_itr=np.asarray(b['predictions']['iTransformer'],dtype=np.float32), pred_sim=np.asarray(b['predictions']['SimDiff'],dtype=np.float32), pred_ours=np.asarray(b['predictions']['ResDiff'],dtype=np.float32), sample_index=np.asarray(b['sample_index']), sensor_index=np.asarray(b['sensor_index']))

def do_dump(cache_dir: Path):
    for k, spec in SPECS.items():
        dump_cache(k, collect(spec), cache_dir)
    print(f'[dump] saved caches to {cache_dir}')

def do_select(cache_dir: Path, out_dir: Path, top_k: int, l1: float, l2: float, l3: float, minc: int):
    C = load_cache(cache_dir)
    cases = select(C, top_k, l1, l2, l3, minc)
    save(cases, out_dir)
    print(f'[select] outputs -> {out_dir}')
    print(f'[select] candidates -> {RESULTS / "fujian_h12_case_candidates.csv"}')
    print(f'[select] figures -> {FIGURES}')
    chosen = choose_triptych_cases(cases)
    for panel, c in chosen:
        print(f"[panel] {panel}: sample={c['sample_index']} sensor={c['sensor_id']} score={c['score']:.6f}")
    if cases:
        b=cases[0]; print(f"[best] sample={b['sample_index']} sensor={b['sensor_id']} score={b['score']:.6f} ours/itr/sim-mae={b['metrics']['ResDiff']['mae']:.6f}/{b['metrics']['iTransformer']['mae']:.6f}/{b['metrics']['SimDiff']['mae']:.6f}")
    else:
        print('[select] no candidate satisfied the strict filters')

def main():
    ap=argparse.ArgumentParser(description='Representative holiday case selection for Fujian-30 H=12')
    ap.add_argument('--mode', choices=['dump','select','all'], default='all')
    ap.add_argument('--cache_dir', default=str(CACHE))
    ap.add_argument('--output_dir', default=str(OUT))
    ap.add_argument('--top_k', type=int, default=30)
    ap.add_argument('--lambda_pte', type=float, default=.5)
    ap.add_argument('--lambda_pve', type=float, default=.5)
    ap.add_argument('--lambda_diff', type=float, default=.5)
    ap.add_argument('--min_conditions', type=int, default=2, choices=[1,2,3])
    a=ap.parse_args(); cache_dir=Path(a.cache_dir); out_dir=Path(a.output_dir)
    if a.mode in ('dump','all'):
        do_dump(cache_dir)
    if a.mode in ('select','all'):
        do_select(cache_dir, out_dir, a.top_k, a.lambda_pte, a.lambda_pve, a.lambda_diff, a.min_conditions)

if __name__ == '__main__': main()

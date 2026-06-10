from __future__ import annotations
import argparse, csv, json, random, sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path: sys.path.insert(0, str(ROOT.parent))
from Holidiff.exp import Exp_Long_Term_Forecast
try:
    import yaml
except Exception as exc:
    yaml = None; _yaml_import_error = exc
INTERPRETATIONS = {1:'Baseline / trend',2:'Rhythm / phase',3:'Fluct. / transition',4:'Disturb. / sharp change'}
EPS = 1e-8

def set_seeds(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_yaml(path:Path)->dict:
    if yaml is None: raise RuntimeError(f'PyYAML import failed: {_yaml_import_error}')
    return dict(yaml.safe_load(path.read_text(encoding='utf-8')))

def build_args(config_path:Path, checkpoint:Path, overrides:argparse.Namespace)->SimpleNamespace:
    cfg = load_yaml(config_path)
    cfg.update({'config':str(config_path),'load_checkpoint':str(checkpoint),'version':str(cfg.get('version','')),'is_training':0,'data':'fujian30','pred_len':int(overrides.horizon),'seq_len':int(getattr(cfg,'seq_len',96)),'batch_size':int(overrides.batch_size),'eval_batch_size':int(overrides.batch_size),'test_times':int(overrides.num_samples),'vs_times':int(overrides.num_samples),'sample_times':int(overrides.num_samples),'seed':int(overrides.seed)})
    device = str(overrides.device)
    if device.startswith('cuda') and torch.cuda.is_available():
        cfg['use_gpu'] = True
        try: cfg['gpu'] = int(device.split(':',1)[1])
        except Exception: cfg['gpu'] = 0
    else:
        cfg['use_gpu'] = False; cfg['gpu'] = 0
    if cfg.get('use_multi_gpu', False):
        d = str(cfg.get('devices','0')).replace(' ',''); cfg['devices'] = d; cfg['device_ids'] = [int(x) for x in d.split(',') if x]; cfg['gpu'] = cfg['device_ids'][0]
    return SimpleNamespace(**cfg)

def masked_mae(pred,true,mask=None)->float:
    if mask is None: return float(np.mean(np.abs(pred-true)))
    return float((np.abs(pred-true)*mask).sum()/max(float(mask.sum()),1.0))

def get_subset_mask(holidays, subset:str):
    if subset == 'all' or holidays is None: return None
    return (holidays.sum(axis=1) > 0).astype(bool)

def build_band_masks(freq_len:int, num_bands:int):
    edges = torch.linspace(0, freq_len, num_bands + 1); out = []
    for i in range(num_bands):
        s, e = int(edges[i].item()), int(edges[i+1].item())
        out.append((s, freq_len if i == num_bands - 1 else e))
    return out

def perturb_history_frequency(batch_x:torch.Tensor, band_idx:int|None, mode:str, num_bands:int)->torch.Tensor:
    if band_idx is None: return batch_x
    x = batch_x.detach().clone().permute(0,2,1); xf = torch.fft.rfft(x, dim=-1); s,e = build_band_masks(xf.size(-1), num_bands)[band_idx]
    if mode == 'remove': xf[..., s:e] = 0
    elif mode == 'keep':
        keep = torch.zeros_like(xf); keep[..., s:e] = xf[..., s:e]; xf = keep
    else: raise ValueError(f'Unsupported perturbation mode: {mode}')
    return torch.fft.irfft(xf, n=x.size(-1), dim=-1).permute(0,2,1).to(dtype=batch_x.dtype)

def build_shared_noise_banks(loader, sample_times:int, enc_in:int, pred_len:int):
    banks = []
    for batch in loader:
        bs = batch[0].shape[0]
        banks.append([torch.randn((bs*enc_in, pred_len), dtype=torch.float32) for _ in range(sample_times)])
    return banks

def run_prediction(exp, loader, sample_times:int, perturb_band:int|None, perturb_mode:str, num_bands:int, stage_name:str, shared_noise_banks=None):
    preds,trues,masks,hols = [],[],[],[]; exp._core_model().eval(); exp.model.eval(); total = len(loader)
    print(f'[stage] {stage_name} | sample_times={sample_times} | total_batches={total}', flush=True)
    with torch.no_grad():
        for bi,batch in enumerate(loader):
            if bi == 0 or (bi+1)%10 == 0 or (bi+1) == total: print(f'[progress] {stage_name}: batch {bi+1}/{total}', flush=True)
            bx,by,bxm,bym = batch[0],batch[1],batch[2],batch[3]
            bm = batch[4] if len(batch)>4 else None; bh = batch[5] if len(batch)>5 else None
            bx = perturb_history_frequency(bx.float(), perturb_band, perturb_mode, num_bands).to(exp.device)
            by = by.float(); bxm = bxm.float().to(exp.device); bym = bym.float().to(exp.device)
            dec = torch.zeros_like(by[:, -exp.args.pred_len:, :]).float(); dec = torch.cat([by[:, :exp.args.label_len, :], dec], 1).float().to(exp.device)
            if shared_noise_banks is not None: noise_bank = [t.to(exp.device) for t in shared_noise_banks[bi]]
            else:
                bs = bx.shape[0]; nnodes = int(exp.args.enc_in); noise_bank = [torch.randn((bs*nnodes, int(exp.args.pred_len)), device=exp.device) for _ in range(sample_times)]
            out = exp.model(bx,bxm,dec,bym,sample_times=sample_times,holiday_flag=bh.float().to(exp.device) if bh is not None else None,future_target=by[:, -exp.args.pred_len:, :].to(exp.device),preset_noises=noise_bank)
            out = out[0] if exp.args.is_diff else out; out = exp._process_model_output(out, is_diff=exp.args.is_diff)
            preds.append(out.detach().cpu().numpy()); trues.append(by[:, -exp.args.pred_len:, :].numpy())
            if bm is not None: masks.append(bm.detach().cpu().numpy())
            if bh is not None: hols.append(bh.detach().cpu().numpy())
    P = np.concatenate(preds,0); Y = np.concatenate(trues,0); M = np.concatenate(masks,0) if masks else None; H = np.concatenate(hols,0) if hols else None
    print(f'[done] {stage_name}: preds_shape={P.shape}, trues_shape={Y.shape}', flush=True)
    return P,Y,M,H

def subset_arrays(preds,trues,masks,holidays,subset:str):
    sm = get_subset_mask(holidays, subset)
    if sm is None: return preds,trues,masks,holidays
    return preds[sm],trues[sm],(masks[sm] if masks is not None else None),holidays[sm]

def save_csv(path:Path, mae_full:float, rows:list[dict])->None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['band','interpretation','mae_full','mae_removed','delta','contribution_all','contribution_dynamic','is_dynamic_band']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({'band':r['band'],'interpretation':r['interpretation'],'mae_full':f'{mae_full:.6f}','mae_removed':f"{r['mae_removed']:.6f}",'delta':f"{r['delta']:.6f}",'contribution_all':f"{r['contribution_all']:.6f}",'contribution_dynamic':'' if r['contribution_dynamic'] is None else f"{r['contribution_dynamic']:.6f}",'is_dynamic_band':str(bool(r['is_dynamic_band']))})

def print_summary(mae_full:float, rows:list[dict], dynamic_band_sum_delta:float)->None:
    print(f'Full model MAE: {mae_full:.6f}'); print(f'dynamic_band_sum_delta: {dynamic_band_sum_delta:.6f}')
    for r in rows:
        dyn = '--' if r['contribution_dynamic'] is None else f"{r['contribution_dynamic']:.6f}"
        print(f"Band k={r['band']} removed MAE={r['mae_removed']:.6f}, Delta={r['delta']:.6f}, C_all={r['contribution_all']:.6f}, C_dyn={dyn}")
    print('\nLaTeX rows (full bands):')
    for r in rows:
        dyn = '--' if r['contribution_dynamic'] is None else f"{r['contribution_dynamic']:.4f}"
        line = f"k={r['band']} & {r['interpretation']} & {r['delta']:.5f} & {r['contribution_all']:.4f} & {dyn}"
        print(line + ' \\\\')
    print('\nLaTeX rows (dynamic bands only):')
    for r in rows:
        if not r['is_dynamic_band']: continue
        line = f"k={r['band']} & {r['interpretation']} & {r['contribution_dynamic']:.4f}"
        print(line + ' \\\\')

def main():
    p = argparse.ArgumentParser(description='Evaluate inference-time input spectral perturbation sensitivity on trained HoliDiff.')
    p.add_argument('--config', type=str, required=True); p.add_argument('--checkpoint', type=str, required=True); p.add_argument('--dataset', type=str, default='Fujian-30'); p.add_argument('--horizon', type=int, default=12); p.add_argument('--split', type=str, default='test'); p.add_argument('--batch_size', type=int, default=32); p.add_argument('--device', type=str, default='cuda:0'); p.add_argument('--num_samples', type=int, default=10); p.add_argument('--save_dir', type=str, default='results/band_contribution/'); p.add_argument('--subset', type=str, choices=['all','holiday'], default='all'); p.add_argument('--seed', type=int, default=2021); p.add_argument('--num_bands', type=int, default=4); p.add_argument('--perturbation_mode', type=str, choices=['remove','keep'], default='remove'); p.add_argument('--save_numpy', action='store_true'); args = p.parse_args()
    set_seeds(args.seed)
    config_path = Path(args.config).resolve(); checkpoint_path = Path(args.checkpoint).resolve(); save_dir = (ROOT / args.save_dir).resolve() if not Path(args.save_dir).is_absolute() else Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    print(f'[setup] dataset={args.dataset} horizon={args.horizon} split={args.split} subset={args.subset} num_samples={args.num_samples} num_bands={args.num_bands} perturbation_mode={args.perturbation_mode}', flush=True)
    print(f'[setup] config={config_path}', flush=True); print(f'[setup] checkpoint={checkpoint_path}', flush=True)
    exp_args = build_args(config_path, checkpoint_path, args); exp = Exp_Long_Term_Forecast(exp_args); dataset, loader = exp._get_data(args.split)
    print(f'[setup] dataloader ready: num_batches={len(loader)} batch_size={args.batch_size}', flush=True)
    shared_noise_banks = build_shared_noise_banks(loader, args.num_samples, int(exp_args.enc_in), int(args.horizon)); print(f'[setup] shared noise banks prepared for {len(shared_noise_banks)} batches', flush=True)
    exp._load_checkpoint_compat(str(checkpoint_path)); prev = exp._set_model_aggregation_mode(getattr(exp.args,'test_aggregation_mode',getattr(exp.args,'aggregation_mode',None)))
    P,Y,M,H = run_prediction(exp, loader, args.num_samples, perturb_band=None, perturb_mode='remove', num_bands=args.num_bands, stage_name='full', shared_noise_banks=shared_noise_banks)
    P,Y,M,H = subset_arrays(P,Y,M,H,args.subset); P = exp._inverse_transform(dataset,P); Y = exp._inverse_transform(dataset,Y); mae_full = masked_mae(P,Y,M)
    rows=[]; pos=[]; pred_store={'full':P}
    for band_idx in range(args.num_bands):
        Pk,Yk,Mk,Hk = run_prediction(exp, loader, args.num_samples, perturb_band=band_idx, perturb_mode='remove', num_bands=args.num_bands, stage_name=f'remove_k{band_idx+1}', shared_noise_banks=shared_noise_banks)
        Pk,Yk,Mk,Hk = subset_arrays(Pk,Yk,Mk,Hk,args.subset); Pk = exp._inverse_transform(dataset,Pk); Yk = exp._inverse_transform(dataset,Yk)
        mae_removed = masked_mae(Pk,Yk,Mk); delta = float(mae_removed - mae_full); pos.append(max(delta,0.0)); pred_store[f'remove_k{band_idx+1}'] = Pk
        rows.append({'band':band_idx+1,'interpretation':INTERPRETATIONS[band_idx+1],'mae_removed':mae_removed,'delta':delta,'contribution_all':0.0,'contribution_dynamic':None,'is_dynamic_band':band_idx+1>=2})
    pos_sum = float(sum(pos)) + EPS; dyn_sum = float(sum(max(r['delta'],0.0) for r in rows if r['band']>=2)); dyn_den = dyn_sum + EPS
    for r,pd in zip(rows,pos):
        r['contribution_all'] = float(pd / pos_sum)
        if r['band'] >= 2: r['contribution_dynamic'] = float(max(r['delta'],0.0) / dyn_den)
    save_csv(save_dir / f'band_contribution_H{args.horizon}.csv', mae_full, rows)
    payload = {'dataset':args.dataset,'horizon':args.horizon,'split':args.split,'subset':args.subset,'seed':args.seed,'num_samples':args.num_samples,'num_bands':args.num_bands,'mae_full':mae_full,'bands':rows,'dynamic_band_sum_delta':dyn_sum,'timestamp':datetime.now().isoformat(timespec='seconds')}
    (save_dir / f'band_contribution_H{args.horizon}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.save_numpy:
        np.save(save_dir / f'pred_full_H{args.horizon}.npy', P); np.save(save_dir / f'true_H{args.horizon}.npy', Y)
        for band_idx in range(args.num_bands): np.save(save_dir / f'pred_remove_k{band_idx+1}_H{args.horizon}.npy', pred_store[f'remove_k{band_idx+1}'])
    exp._restore_model_aggregation_mode(prev); print_summary(mae_full, rows, dyn_sum)

if __name__ == '__main__':
    main()

"""
采样解剖脚本：对 worst-case 样本收集 10 条原始 DPM 轨迹 + MoM 中间结果。

用法：
  python diagnosis/macro/scripts/anatomy_sampling.py \
      --config Holidiff/configs/compare_fujian30_standard_30epoch_base.yaml \
      --checkpoint <ckpt_path> \
      --output_dir diagnosis/macro/output/anatomy \
      --sample_ids 3817,2475,3819,2474 \
      --sample_times 10 \
      --n_blocks 5 \
      --rmom 5
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Holidiff.data_provider.fujian30_loader import Fujian30CsvDataset
from Holidiff.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

try:
    import yaml
except ImportError as e:
    raise RuntimeError(f"PyYAML required: {e}")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_args(cfg):
    cfg = dict(cfg)
    cfg["use_gpu"] = bool(torch.cuda.is_available() and cfg.get("use_gpu", True))
    if cfg.get("use_multi_gpu", False):
        ids = str(cfg.get("devices", "0")).replace(" ", "").split(",")
        cfg["device_ids"] = [int(x) for x in ids]
        cfg["gpu"] = cfg["device_ids"][0]
    return SimpleNamespace(**cfg)


def build_setting(a):
    return (
        "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}"
        .format(
            a.task_name, a.model_id, a.model, a.data, a.features,
            a.seq_len, a.label_len, a.pred_len, a.d_model, a.n_heads,
            a.e_layers, a.d_layers, a.d_ff, a.expand, a.d_conv,
            a.factor, a.embed, a.distil, a.des, 0,
        )
    )


def inv_transform(ds, arr):
    flat = arr.reshape(-1, arr.shape[-1])
    return ds.scaler.inverse_transform(flat).reshape(arr.shape)


def manual_sample_once(model, x_past_norm, x_mark_enc, pred_len, enc_in, seed):
    """执行一次 DPM 采样，返回 raw-space (B, pred_len, N)。"""
    torch.manual_seed(seed)
    B_N = x_past_norm.shape[0]
    B = B_N // enc_in
    device = model.betas.device

    start_code = torch.randn((B_N, pred_len), device=device)
    samples, _ = model.sampler.sample(
        S=model.configs.s_steps,
        conditioning=x_past_norm,
        x_mark_enc=x_mark_enc,
        batch_size=B_N,
        shape=[enc_in, pred_len],
        verbose=False,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        eta=0.0,
        x_T=start_code,
    )
    samples = samples.reshape(B, enc_in, pred_len)
    return samples


def mom_aggregate(all_samples_np, n_blocks, rmom, rng):
    """
    all_samples_np: (sample_times, B, pred_len, N)
    返回:
      mom_out:      (B, pred_len, N)
      all_block_means: list of (n_blocks, B, pred_len, N) — 每次 rmom 的 block 均值
    """
    S, B, T, N = all_samples_np.shape
    mom_results = []
    all_block_means = []

    for _ in range(rmom):
        idx = rng.permutation(S)
        shuffled = all_samples_np[idx]
        block_size = S // n_blocks
        block_means = []
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size if b < n_blocks - 1 else S
            block_means.append(shuffled[start:end].mean(axis=0))
        block_means = np.stack(block_means, axis=0)  # (n_blocks, B, T, N)
        all_block_means.append(block_means)
        mom_results.append(np.median(block_means, axis=0))

    mom_out = np.stack(mom_results, axis=0).mean(axis=0)
    return mom_out, all_block_means


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output_dir", default=str(ROOT / "diagnosis" / "macro" / "output" / "anatomy"))
    p.add_argument("--sample_ids", default="3817,2475,3819,2474",
                   help="comma-separated sample_ids to dissect")
    p.add_argument("--sample_times", type=int, default=10)
    p.add_argument("--n_blocks", type=int, default=5)
    p.add_argument("--rmom", type=int, default=5)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    c = p.parse_args()

    target_ids = set(int(x) for x in c.sample_ids.split(","))
    out_dir = Path(c.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(Path(c.config))
    a = make_args(cfg)
    if c.batch_size is not None:
        a.batch_size = c.batch_size

    exp = Exp_Long_Term_Forecast(a)
    s = build_setting(a)
    ckpt = Path(c.checkpoint) if c.checkpoint else Path(a.checkpoints) / s / "checkpoint.pth"
    if ckpt.is_dir():
        ckpt = ckpt / "checkpoint.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    csv_path = Path(a.root_path) / a.data_path
    ds = Fujian30CsvDataset(
        str(csv_path),
        str(Path(a.root_path) / "adjacent_gantry.csv"),
        "test",
        getattr(a, "mode", "standard"),
        a.seq_len, a.pred_len,
        getattr(a, "stride", 1),
        getattr(a, "scale", True),
    )
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    drop_last=False, num_workers=c.num_workers)

    m = exp.model.module if hasattr(exp.model, "module") else exp.model
    device = m.betas.device
    enc_in = a.enc_in
    pred_len = a.pred_len
    rng = np.random.default_rng(c.seed_base)

    global_sample_idx = 0
    collected = {}

    with torch.no_grad():
        for batch_i, batch in enumerate(dl):
            bx, by, bxm, bym = batch[0], batch[1], batch[2], batch[3]
            B = bx.shape[0]

            batch_target_local = []
            for local_i in range(B):
                gidx = global_sample_idx + local_i
                if gidx in target_ids:
                    batch_target_local.append((gidx, local_i))

            if not batch_target_local:
                global_sample_idx += B
                continue

            print(f"[batch {batch_i}] processing samples: {[g for g, _ in batch_target_local]}")

            # Process each target sample individually to avoid running the full batch
            for gidx, local_i in batch_target_local:
                print(f"  sampling {c.sample_times}x for sample {gidx} (local {local_i})...")

                # Slice out just this one sample: shape (1, seq_len, N)
                bx_one = bx[local_i:local_i+1].float().to(device)
                bxm_one = bxm[local_i:local_i+1].float().to(device)
                by_one = by[local_i:local_i+1]

                # Normalize exactly as forward_consensus_inference does.
                # For new_norm, capture RevIN mean/stdev immediately after norm so
                # we can apply a stateless denorm for each of the 10 samples.
                if a.new_norm:
                    x_past = m.nda_layer(bx_one, "norm")   # (1, seq_len, N)
                    revin_mean = m.nda_layer.mean.clone()   # (1, 1, N)
                    revin_stdev = m.nda_layer.stdev.clone() # (1, 1, N)
                    revin_affine_w = m.nda_layer.affine_weight.clone()  # (N,)
                    revin_affine_b = m.nda_layer.affine_bias.clone()    # (N,)
                    x_past = x_past.permute(0, 2, 1)        # (1, N, seq_len)
                else:
                    x_past = bx_one.permute(0, 2, 1)        # (1, N, seq_len)
                    mean_ = x_past.mean(dim=-1, keepdim=True)
                    std_ = x_past.std(dim=-1, keepdim=True)
                    x_past = (x_past - mean_) / (std_ + 1e-5)

                x_past_flat = x_past.reshape(enc_in, -1)    # (N, seq_len) — B=1 so B*N=N

                # Move captured RevIN stats to CPU for numpy-based denorm
                revin_mean_np = revin_mean.cpu().numpy()     # (1, 1, N)
                revin_stdev_np = revin_stdev.cpu().numpy()   # (1, 1, N)
                revin_w_np = revin_affine_w.cpu().numpy()    # (N,)
                revin_b_np = revin_affine_b.cpu().numpy()    # (N,)
                eps2 = m.nda_layer.eps ** 2

                def revin_denorm_np(arr):
                    """Stateless RevIN denorm in numpy. arr: (1, T, N)"""
                    arr = arr - revin_b_np
                    arr = arr / (revin_w_np + eps2)
                    arr = arr * revin_stdev_np
                    arr = arr + revin_mean_np
                    return arr

                # Collect sample_times raw trajectories
                all_scaled = []
                for s_idx in range(c.sample_times):
                    seed = c.seed_base * 1000 + gidx * 100 + s_idx
                    raw_norm = manual_sample_once(
                        m, x_past_flat, bxm_one, pred_len, enc_in, seed
                    )  # (1, N, pred_len)
                    raw_norm_np = raw_norm.permute(0, 2, 1).cpu().numpy()  # (1, pred_len, N)

                    if a.new_norm:
                        t_np = revin_denorm_np(raw_norm_np)
                    else:
                        t_np = raw_norm_np

                    all_scaled.append(t_np)

                all_scaled = np.stack(all_scaled, axis=0)    # (S, 1, pred_len, N)

                # Apply dataset scaler inverse to get raw flow values
                all_raw = np.stack(
                    [inv_transform(ds, all_scaled[s]) for s in range(c.sample_times)],
                    axis=0
                )  # (S, 1, pred_len, N)

                # MoM aggregation — squeeze out the B=1 dim for storage
                all_raw_sq = all_raw[:, 0, :, :]            # (S, T, N)
                all_raw_4d = all_raw_sq[:, np.newaxis, :, :]  # (S, 1, T, N) for mom_aggregate
                mom_out_4d, all_block_means = mom_aggregate(all_raw_4d, c.n_blocks, c.rmom, rng)
                mom_out = mom_out_4d[0]                      # (T, N)
                mean_out = all_raw_sq.mean(axis=0)           # (T, N)

                # Ground truth and history (raw space)
                true_raw = inv_transform(ds, by_one[:, -pred_len:, :].numpy())[0]  # (T, N)
                hist_raw = inv_transform(ds, bx[local_i:local_i+1].numpy())[0]     # (seq_len, N)

                last_block_means = all_block_means[-1][:, 0, :, :]  # (n_blocks, T, N)

                record = {
                    "sample_id": gidx,
                    "batch_idx": batch_i,
                    "local_idx": local_i,
                    "all_raw_samples": all_raw_sq.tolist(),          # (S, T, N)
                    "block_means": last_block_means.tolist(),         # (n_blocks, T, N)
                    "mom_pred": mom_out.tolist(),                     # (T, N)
                    "mean_pred": mean_out.tolist(),                   # (T, N)
                    "true_raw": true_raw.tolist(),                    # (T, N)
                    "history_raw": hist_raw.tolist(),                 # (seq_len, N)
                    "n_blocks": c.n_blocks,
                    "rmom": c.rmom,
                    "sample_times": c.sample_times,
                }
                out_path = out_dir / f"anatomy_sample{gidx}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False)
                print(f"  saved → {out_path}")
                collected[gidx] = True

            global_sample_idx += B

            if len(collected) == len(target_ids):
                print("All target samples collected, stopping early.")
                break

    missing = target_ids - set(collected.keys())
    if missing:
        print(f"WARNING: samples not found in test set: {missing}")
    print(f"Done. {len(collected)} anatomy files saved to {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None

try:
    import yaml
except Exception as exc:
    yaml = None
    _yaml_import_error = exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from Holidiff.exp import Exp_Long_Term_Forecast
from Holidiff.macro.aggregation_utils import split_history_regimes
from Holidiff.macro.dca_aggregator import DensityCentroidAggregator

TEXT_COLOR = "#202020"
GRID_COLOR = "#E9E9E9"
SPINE_COLOR = "#555555"
POINT_COLOR = "#AFCBFF"
DENSITY_COLOR = "#E6E6E6"
GT_COLOR = "#111111"
MEAN_COLOR = "#8FB4FF"
MEDIAN_COLOR = "#F7B7C3"
MOM_COLOR = "#B8E0A5"
KDE_COLOR = "#F7D38B"
DCA_COLOR = "#F0A562"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(f"PyYAML import failed: {_yaml_import_error}")
    cfg = dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    cfg["use_gpu"] = bool(cfg.get("use_gpu", True) and torch.cuda.is_available())
    cfg["train_val_aggregation_mode"] = cfg.get("train_val_aggregation_mode", "single")
    cfg["test_aggregation_mode"] = cfg.get("test_aggregation_mode", "dca")
    cfg["test_times"] = cfg.get("test_times", cfg.get("vs_times", cfg.get("sample_times", 1)))
    if cfg.get("use_multi_gpu", False):
        devices = str(cfg.get("devices", "0")).replace(" ", "")
        cfg["devices"] = devices
        cfg["device_ids"] = [int(x) for x in devices.split(",") if x]
        cfg["gpu"] = cfg["device_ids"][0]
    return cfg


def masked_mae_per_window(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    err = np.abs(pred - true)
    if mask is None:
        return err.mean(axis=1)
    denom = np.maximum(mask.sum(axis=1), 1e-8)
    return (err * mask).sum(axis=1) / denom


def masked_abs_errors(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true)


def mom_aggregate(samples_bkhn: np.ndarray, n_blocks: int, rmom_n: int, seed: int) -> np.ndarray:
    outputs = torch.from_numpy(samples_bkhn).permute(1, 0, 2, 3).float().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    def consensus_mean(seq: torch.Tensor) -> torch.Tensor:
        return torch.sum(seq, dim=0) / seq.size(0)

    def consensus_reduce(tensor: torch.Tensor) -> torch.Tensor:
        nb = int(n_blocks)
        if nb > tensor.size(0):
            nb = max(1, (tensor.size(0) + 1) // 2)
        indic = torch.randperm(tensor.size(0), generator=generator)
        tensor = tensor[indic]
        block_size = max(1, tensor.size(0) // nb)
        means = []
        for i in range(nb):
            start = i * block_size
            end = start + block_size if (i + 1) < nb else tensor.size(0)
            means.append(consensus_mean(tensor[start:end]))
        return torch.median(torch.stack(means), dim=0)[0]

    results = []
    for _ in range(int(rmom_n)):
        shuffled = outputs[torch.randperm(outputs.size(0), generator=generator)]
        results.append(consensus_reduce(shuffled))
    return torch.median(torch.stack(results), dim=0)[0].numpy()


def aggregate_density_family(
    samples_bkhn: np.ndarray,
    history_bln: np.ndarray,
    bandwidth_kde: float,
    bandwidth_hist: float,
    dca_mode: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bsz, ksz, horizon, nodes = samples_bkhn.shape
    kde_pred = np.zeros((bsz, horizon, nodes), dtype=np.float32)
    dca_pred = np.zeros((bsz, horizon, nodes), dtype=np.float32)
    dca_weights = np.zeros((bsz, ksz, horizon, nodes), dtype=np.float32)

    agg_kde = DensityCentroidAggregator(
        bandwidth_kde=bandwidth_kde, bandwidth_hist=bandwidth_hist, mode="kde_only", eps=eps
    )
    agg_dca = DensityCentroidAggregator(
        bandwidth_kde=bandwidth_kde, bandwidth_hist=bandwidth_hist, mode=dca_mode, eps=eps
    )

    for bi in range(bsz):
        for ni in range(nodes):
            _hist_all, hist_free, hist_cong = split_history_regimes(history_bln[bi, :, ni])
            samp = samples_bkhn[bi, :, :, ni]
            diff = samp[:, None, :] - samp[None, :, :]
            w_kde = np.exp(-0.5 * (diff / max(agg_kde.h_kde, agg_kde.eps)) ** 2).sum(axis=1)
            kde_norm = w_kde / np.maximum(w_kde.sum(axis=0, keepdims=True), agg_kde.eps)
            kde_pred[bi, :, ni] = (kde_norm * samp).sum(axis=0)

            if hist_free is None and hist_cong is None:
                w_joint = w_kde
            else:
                h = max(agg_dca.h_hist, agg_dca.eps)
                like_free = np.zeros((ksz, horizon), dtype=np.float32)
                like_cong = np.zeros((ksz, horizon), dtype=np.float32)
                if hist_free is not None and len(hist_free) > 0:
                    diff_free = samp[:, :, None] - hist_free[None, None, :]
                    like_free = np.exp(-0.5 * (diff_free / h) ** 2).sum(axis=2)
                if hist_cong is not None and len(hist_cong) > 0:
                    diff_cong = samp[:, :, None] - hist_cong[None, None, :]
                    like_cong = np.exp(-0.5 * (diff_cong / h) ** 2).sum(axis=2)
                if agg_dca.mode == "competitive":
                    w_hist = np.maximum(like_free, like_cong)
                else:
                    w_hist = like_free + like_cong
                if agg_dca.mode == "hist_only":
                    w_joint = w_hist
                else:
                    w_joint = w_kde * np.maximum(w_hist, agg_dca.eps)
            dca_norm = w_joint / np.maximum(w_joint.sum(axis=0, keepdims=True), agg_dca.eps)
            dca_pred[bi, :, ni] = (dca_norm * samp).sum(axis=0)
            dca_weights[bi, :, :, ni] = dca_norm
    return kde_pred, dca_pred, dca_weights


def style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.tick_params(labelsize=8.5, colors=TEXT_COLOR)


def inverse_transform(exp, dataset, arr: np.ndarray) -> np.ndarray:
    return exp._inverse_transform(dataset, arr)


def norm_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or float(np.max(x) - np.min(x)) < 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def count_hist_peaks(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float32)
    hist, _edges = np.histogram(values, bins=min(8, max(5, values.size // 2)))
    peaks = 0
    for idx, cnt in enumerate(hist):
        left = hist[idx - 1] if idx > 0 else -1
        right = hist[idx + 1] if idx + 1 < hist.size else -1
        if cnt >= 2 and cnt >= left and cnt >= right:
            peaks += 1
    return max(peaks, 1)


def compute_density_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 2:
        return None
    lo = float(values.min())
    hi = float(values.max())
    pad = max((hi - lo) * 0.15, 1e-3)
    grid = np.linspace(lo - pad, hi + pad, 256)
    if gaussian_kde is not None:
        try:
            return grid, gaussian_kde(values)(grid)
        except Exception:
            pass
    hist, edges = np.histogram(values, bins=min(20, max(5, values.size // 2)), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = np.interp(grid, centers, hist, left=0.0, right=0.0)
    return grid, density


def choose_selected_position(rows: list[dict]) -> tuple[dict, list[dict]]:
    if not rows:
        raise RuntimeError("No eligible point-wise cases found.")

    dca_errors = np.asarray([r["dca_abs_error"] for r in rows], dtype=np.float64)
    err_threshold = float(np.percentile(dca_errors, 40))
    filtered = [r for r in rows if r["dca_abs_error"] <= err_threshold]
    if len(filtered) < min(20, len(rows)):
        err_threshold = float(np.percentile(dca_errors, 55))
        filtered = [r for r in rows if r["dca_abs_error"] <= err_threshold]
    if not filtered:
        filtered = list(rows)

    gaps = np.asarray([r["dca_improvement_over_best_non_dca"] for r in filtered], dtype=np.float64)
    dca_err = np.asarray([r["dca_abs_error"] for r in filtered], dtype=np.float64)
    spreads = np.asarray([r["candidate_spread_at_h"] for r in filtered], dtype=np.float64)
    vars_ = np.asarray([r["future_variation"] for r in filtered], dtype=np.float64)
    peaks = np.asarray([r["local_peak_count"] for r in filtered], dtype=np.float64)
    window_gaps = np.asarray([r["window_gap_to_best_non_dca"] for r in filtered], dtype=np.float64)

    score = (
        0.40 * norm_array(gaps)
        + 0.25 * (1.0 - norm_array(dca_err))
        + 0.15 * norm_array(spreads)
        + 0.10 * norm_array(vars_)
        + 0.05 * norm_array(peaks)
        + 0.05 * norm_array(window_gaps)
    )

    ranked = []
    for row, sc in zip(filtered, score):
        rec = dict(row)
        rec["selection_score"] = float(sc)
        ranked.append(rec)
    ranked.sort(
        key=lambda r: (r["selection_score"], r["dca_improvement_over_best_non_dca"], -r["dca_abs_error"]),
        reverse=True,
    )
    return ranked[0], ranked[:20]


def make_plot(selected: dict, payload: dict, out_pdf: Path, out_png: Path) -> None:
    values = np.asarray(payload["candidate_values"], dtype=np.float32)
    weights = np.asarray(payload["candidate_weights"], dtype=np.float32)
    point_values = payload["point_values"]
    gt_value = float(point_values["ground_truth"])

    fig, ax = plt.subplots(figsize=(7.0, 3.25), facecolor="white")
    style_axis(ax)

    density = compute_density_curve(values)
    if density is not None:
        grid, dens = density
        dens = dens / max(float(dens.max()), 1e-8)
        dens = dens * 0.24
        ax.fill_between(grid, -dens, dens, color=DENSITY_COLOR, alpha=0.65, zorder=1)
        ax.plot(grid, dens, color="#D0D0D0", lw=1.0, zorder=1)
        ax.plot(grid, -dens, color="#D0D0D0", lw=1.0, zorder=1)

    rng = np.random.default_rng(2021)
    jitter = rng.uniform(-0.105, 0.105, size=values.shape[0])
    sizes = 34.0 + 235.0 * (weights / max(float(weights.max()), 1e-8))
    ax.scatter(values, jitter, s=sizes, facecolor=POINT_COLOR, edgecolor="white", linewidth=0.7, alpha=0.97, zorder=3)

    line_styles = {
        "ground_truth": (GT_COLOR, "-", 2.6, "GT"),
        "mean": (MEAN_COLOR, "--", 1.8, "Mean"),
        "median": (MEDIAN_COLOR, (0, (4, 1.4)), 1.8, "Median"),
        "mom": (MOM_COLOR, (0, (2.5, 1.1)), 1.8, "MoM"),
        "kde_mode": (KDE_COLOR, "-.", 1.9, "KDE Mode"),
        "dca": (DCA_COLOR, "-", 2.6, "DCA"),
    }
    order = ["ground_truth", "mean", "median", "mom", "kde_mode", "dca"]
    for key in order:
        color, ls, lw, _short = line_styles[key]
        ax.axvline(float(point_values[key]), color=color, ls=ls, lw=lw, zorder=2)

    x_lo = float(min(np.min(values), min(point_values.values())))
    x_hi = float(max(np.max(values), max(point_values.values())))
    x_rng = max(x_hi - x_lo, 1e-6)
    ordered = sorted([(key, float(point_values[key])) for key in order], key=lambda kv: kv[1])
    top_levels = [0.245, 0.215, 0.185, 0.155, 0.125, 0.095]
    last_x = None
    level = 0
    for key, value in ordered:
        if last_x is not None and abs(value - last_x) < 0.045 * x_rng:
            level = min(level + 1, len(top_levels) - 1)
        else:
            level = 0
        color, _ls, _lw, short = line_styles[key]
        ax.text(
            value,
            top_levels[level],
            short,
            color=color,
            fontsize=7.4,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.94),
            zorder=4,
        )
        last_x = value

    text_lines = [f"GT={gt_value:.2f}"]
    for key, label in [("mean", "Mean"), ("median", "Median"), ("mom", "MoM"), ("kde_mode", "KDE"), ("dca", "DCA")]:
        text_lines.append(f"{label} |e|={abs(point_values[key] - gt_value):.2f}")
    ax.text(
        0.015,
        0.97,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D8D8D8", linewidth=0.7),
    )

    handles = [
        Line2D([0], [0], color=GT_COLOR, lw=2.6, ls="-", label="Ground Truth"),
        Line2D([0], [0], color=MEAN_COLOR, lw=1.8, ls="--", label="Mean"),
        Line2D([0], [0], color=MEDIAN_COLOR, lw=1.8, ls=(0, (4, 1.4)), label="Median"),
        Line2D([0], [0], color=MOM_COLOR, lw=1.8, ls=(0, (2.5, 1.1)), label="MoM"),
        Line2D([0], [0], color=KDE_COLOR, lw=1.9, ls="-.", label="KDE Mode"),
        Line2D([0], [0], color=DCA_COLOR, lw=2.6, ls="-", label="DCA"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=POINT_COLOR,
            markeredgecolor="white",
            markersize=7.3,
            label="Candidate samples\n(size = DCA weight)",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.02, 1.01), frameon=False, fontsize=7.6, ncol=2, handlelength=2.3, columnspacing=1.25, handletextpad=0.8)

    pad = max(0.12 * x_rng, 1e-3)
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(-0.28, 0.28)
    ax.set_xlabel("Prediction value", fontsize=9.3)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.18)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_cache(exp, test_data, test_loader, checkpoint_path: Path, cache_path: Path, seed: int, k: int) -> dict:
    exp._load_checkpoint_compat(str(checkpoint_path))
    exp.model.eval()
    core_model = exp._core_model()

    bandwidth_kde = float(getattr(exp.args, "dca_bandwidth_kde", 15.0))
    bandwidth_hist = float(getattr(exp.args, "dca_bandwidth_hist", 20.0))
    dca_mode = str(getattr(exp.args, "dca_mode", "joint")).lower()
    rmom_n = int(getattr(exp.args, "rmom", 20))
    n_blocks = int(getattr(core_model, "n_blocks", getattr(exp.args, "n_b", 5)))
    eps = float(getattr(exp.args, "aggregation_eps", 1e-8))

    histories = []
    futures = []
    masks = []
    holidays = []
    candidates = []
    preds_single = []
    preds_mean = []
    preds_median = []
    preds_mom = []
    preds_kde = []
    preds_dca = []
    dca_weights_all = []

    total_batches = len(test_loader)
    print(f"[aggregation-case] total_batches={total_batches} K={k}", flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"[aggregation-case] batch {batch_idx}/{total_batches}", flush=True)
            batch_x = batch[0].float().to(exp.device)
            batch_y = batch[1].float()
            batch_x_mark = batch[2].float().to(exp.device)
            batch_y_mark = batch[3].float().to(exp.device)
            batch_mask = batch[4].float().numpy() if len(batch) > 4 else None
            batch_holiday = batch[5].float().numpy() if len(batch) > 5 else None
            dec = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
            dec = torch.cat([batch_y[:, :exp.args.label_len, :], dec], dim=1).float().to(exp.device)
            _, all_samples = exp.model(
                batch_x,
                batch_x_mark,
                dec,
                batch_y_mark,
                sample_times=k,
                holiday_flag=batch[5].float().to(exp.device) if len(batch) > 5 else None,
                future_target=batch_y[:, -exp.args.pred_len:, :].to(exp.device),
            )
            cand = all_samples.detach().cpu().numpy().astype(np.float32)
            hist = batch_x.detach().cpu().numpy().astype(np.float32)
            true = batch_y[:, -exp.args.pred_len:, :].numpy().astype(np.float32)

            cand = inverse_transform(exp, test_data, cand)
            hist = inverse_transform(exp, test_data, hist)
            true = inverse_transform(exp, test_data, true)

            pred_single = cand[:, 0, :, :]
            pred_mean = cand.mean(axis=1)
            pred_median = np.median(cand, axis=1)
            pred_mom = mom_aggregate(cand, n_blocks=n_blocks, rmom_n=rmom_n, seed=int(seed) + batch_idx)
            pred_kde, pred_dca, batch_dca_weights = aggregate_density_family(
                cand, hist, bandwidth_kde, bandwidth_hist, dca_mode, eps
            )

            histories.append(hist)
            futures.append(true)
            candidates.append(cand)
            preds_single.append(pred_single)
            preds_mean.append(pred_mean)
            preds_median.append(pred_median)
            preds_mom.append(pred_mom)
            preds_kde.append(pred_kde)
            preds_dca.append(pred_dca)
            dca_weights_all.append(batch_dca_weights)
            if batch_mask is not None:
                masks.append(batch_mask.astype(np.float32))
            if batch_holiday is not None:
                holidays.append(batch_holiday.astype(np.float32))

    payload = {
        "history": np.concatenate(histories, axis=0),
        "truth": np.concatenate(futures, axis=0),
        "candidates": np.concatenate(candidates, axis=0),
        "pred_single": np.concatenate(preds_single, axis=0),
        "pred_mean": np.concatenate(preds_mean, axis=0),
        "pred_median": np.concatenate(preds_median, axis=0),
        "pred_mom": np.concatenate(preds_mom, axis=0),
        "pred_kde": np.concatenate(preds_kde, axis=0),
        "pred_dca": np.concatenate(preds_dca, axis=0),
        "dca_weights": np.concatenate(dca_weights_all, axis=0),
        "mask": np.empty((0,), dtype=np.float32) if not masks else np.concatenate(masks, axis=0),
        "holiday": np.empty((0,), dtype=np.float32) if not holidays else np.concatenate(holidays, axis=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def load_cache(cache_path: Path) -> dict:
    arrs = np.load(cache_path)
    return {k: arrs[k] for k in arrs.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic representative aggregation case selection and plotting.")
    parser.add_argument("--config", type=str, default="/root/yanyijin/STdiff/Holidiff/configs/fujian30/holidiff_h12.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/STdiff_runs/checkpoints/fujian30/final_holidiff__h12/checkpoint.pth",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--cache_npz",
        type=str,
        default="/root/yanyijin/STdiff/Holidiff/results/aggregation_case_cache_h12_seed2021.npz",
    )
    parser.add_argument(
        "--save_csv", type=str, default="/root/yanyijin/STdiff/Holidiff/results/aggregation_case_selection.csv"
    )
    parser.add_argument(
        "--save_json", type=str, default="/root/yanyijin/STdiff/Holidiff/results/aggregation_case_metadata.json"
    )
    parser.add_argument("--save_pdf", type=str, default="/root/yanyijin/STdiff/Holidiff/figures/aggcase.pdf")
    parser.add_argument("--save_png", type=str, default="/root/yanyijin/STdiff/Holidiff/figures/aggcase.png")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    cache_path = Path(args.cache_npz)
    save_csv = Path(args.save_csv)
    save_json = Path(args.save_json)
    save_pdf = Path(args.save_pdf)
    save_png = Path(args.save_png)

    cfg = load_cfg(config_path)
    cfg.update({"config": str(config_path), "version": "aggregation_case_h12", "load_checkpoint": str(checkpoint_path)})
    set_seeds(int(args.seed))
    exp = Exp_Long_Term_Forecast(SimpleNamespace(**cfg))
    test_data, test_loader = exp._get_data("test")

    if cache_path.exists():
        print(f"[aggregation-case] reuse cache: {cache_path}", flush=True)
        cached = load_cache(cache_path)
        if int(cached["candidates"].shape[1]) != int(args.k):
            print("[aggregation-case] cache K mismatch, rebuilding cache.", flush=True)
            cached = build_cache(exp, test_data, test_loader, checkpoint_path, cache_path, int(args.seed), int(args.k))
    else:
        cached = build_cache(exp, test_data, test_loader, checkpoint_path, cache_path, int(args.seed), int(args.k))

    true_all = cached["truth"]
    cand_all = cached["candidates"]
    pred_single_all = cached["pred_single"]
    pred_mean_all = cached["pred_mean"]
    pred_median_all = cached["pred_median"]
    pred_mom_all = cached["pred_mom"]
    pred_kde_all = cached["pred_kde"]
    pred_dca_all = cached["pred_dca"]
    dca_weights = cached["dca_weights"]
    mask_all = cached["mask"] if "mask" in cached and cached["mask"].size > 0 else None

    meta = exp._load_fujian30_meta(test_data)
    total_samples = true_all.shape[0]
    for key in ("starts", "pred_starts", "pred_ends"):
        meta[key] = np.asarray(meta[key])[:total_samples]

    mae_mean = masked_mae_per_window(pred_mean_all, true_all, mask_all)
    mae_median = masked_mae_per_window(pred_median_all, true_all, mask_all)
    mae_mom = masked_mae_per_window(pred_mom_all, true_all, mask_all)
    mae_kde = masked_mae_per_window(pred_kde_all, true_all, mask_all)
    mae_dca = masked_mae_per_window(pred_dca_all, true_all, mask_all)

    abs_single = masked_abs_errors(pred_single_all, true_all)
    abs_mean = masked_abs_errors(pred_mean_all, true_all)
    abs_median = masked_abs_errors(pred_median_all, true_all)
    abs_mom = masked_abs_errors(pred_mom_all, true_all)
    abs_kde = masked_abs_errors(pred_kde_all, true_all)
    abs_dca = masked_abs_errors(pred_dca_all, true_all)

    future_range_all = true_all.max(axis=1) - true_all.min(axis=1)
    future_variation_all = future_range_all + np.abs(np.diff(true_all, axis=1)).sum(axis=1)
    candidate_spread_mean = cand_all.std(axis=1).mean(axis=1)
    candidate_spread_peak = cand_all.std(axis=1).max(axis=1)
    step_spread = cand_all.std(axis=1)
    turning_points = np.sign(np.diff(true_all, axis=1))
    turning_count = np.sum((turning_points[:, 1:, :] * turning_points[:, :-1, :]) < 0, axis=1)

    best_alt_step_stack = np.stack([abs_mean, abs_median, abs_mom, abs_kde], axis=-1)
    best_alt_step_err = np.min(best_alt_step_stack, axis=-1)
    best_alt_step_idx = np.argmin(best_alt_step_stack, axis=-1)
    best_alt_step_names = np.array(["mean", "median", "mom", "kde_mode"], dtype=object)[best_alt_step_idx]

    best_alt_window_stack = np.stack([mae_mean, mae_median, mae_mom, mae_kde], axis=-1)
    best_alt_window_mae = np.min(best_alt_window_stack, axis=-1)
    window_gap = best_alt_window_mae - mae_dca

    dca_step_best = abs_dca < best_alt_step_err
    dyn_threshold = float(np.percentile(future_variation_all.reshape(-1), 35))
    spread_threshold = float(np.percentile(step_spread.reshape(-1), 45))
    point_gap = best_alt_step_err - abs_dca
    coarse_mask = dca_step_best & (window_gap[:, None, :] > 0) & (future_variation_all[:, None, :] > dyn_threshold) & (
        step_spread > spread_threshold
    )

    rows = []
    for bi, hi, ni in np.argwhere(coarse_mask):
        values = cand_all[bi, :, hi, ni]
        rows.append(
            {
                "sample_index": int(bi),
                "node_index": int(ni),
                "node_id": str(meta["station_ids"][ni]),
                "prediction_step": int(hi + 1),
                "prediction_step_index": int(hi),
                "point_time": str(meta["time_index"][int(meta["pred_starts"][bi]) + int(hi)]),
                "pred_start_time": str(meta["time_index"][int(meta["pred_starts"][bi])]),
                "pred_end_time": str(meta["time_index"][int(meta["pred_ends"][bi])]),
                "ground_truth": float(true_all[bi, hi, ni]),
                "single": float(pred_single_all[bi, hi, ni]),
                "mean": float(pred_mean_all[bi, hi, ni]),
                "median": float(pred_median_all[bi, hi, ni]),
                "mom": float(pred_mom_all[bi, hi, ni]),
                "kde_mode": float(pred_kde_all[bi, hi, ni]),
                "dca": float(pred_dca_all[bi, hi, ni]),
                "single_abs_error": float(abs_single[bi, hi, ni]),
                "mean_abs_error": float(abs_mean[bi, hi, ni]),
                "median_abs_error": float(abs_median[bi, hi, ni]),
                "mom_abs_error": float(abs_mom[bi, hi, ni]),
                "kde_mode_abs_error": float(abs_kde[bi, hi, ni]),
                "dca_abs_error": float(abs_dca[bi, hi, ni]),
                "best_non_dca_method": str(best_alt_step_names[bi, hi, ni]),
                "best_non_dca_abs_error": float(best_alt_step_err[bi, hi, ni]),
                "dca_improvement_over_best_non_dca": float(point_gap[bi, hi, ni]),
                "window_gap_to_best_non_dca": float(window_gap[bi, ni]),
                "mae_mean": float(mae_mean[bi, ni]),
                "mae_median": float(mae_median[bi, ni]),
                "mae_mom": float(mae_mom[bi, ni]),
                "mae_kde": float(mae_kde[bi, ni]),
                "mae_dca": float(mae_dca[bi, ni]),
                "future_range": float(future_range_all[bi, ni]),
                "future_variation": float(future_variation_all[bi, ni]),
                "candidate_spread_mean": float(candidate_spread_mean[bi, ni]),
                "candidate_spread_peak": float(candidate_spread_peak[bi, ni]),
                "candidate_spread_at_h": float(step_spread[bi, hi, ni]),
                "local_peak_count": int(count_hist_peaks(values)),
                "turning_points": int(turning_count[bi, ni]),
            }
        )

    selected, top20 = choose_selected_position(rows)
    bi = int(selected["sample_index"])
    ni = int(selected["node_index"])
    hi = int(selected["prediction_step_index"])

    payload = {
        "candidate_values": cand_all[bi, :, hi, ni],
        "candidate_weights": dca_weights[bi, :, hi, ni],
        "point_values": {
            "ground_truth": float(true_all[bi, hi, ni]),
            "mean": float(pred_mean_all[bi, hi, ni]),
            "median": float(pred_median_all[bi, hi, ni]),
            "mom": float(pred_mom_all[bi, hi, ni]),
            "kde_mode": float(pred_kde_all[bi, hi, ni]),
            "dca": float(pred_dca_all[bi, hi, ni]),
        },
    }
    make_plot(selected, payload, save_pdf, save_png)

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "sample_index",
        "node_index",
        "node_id",
        "prediction_step",
        "point_time",
        "ground_truth",
        "mean",
        "median",
        "mom",
        "kde_mode",
        "dca",
        "mean_abs_error",
        "median_abs_error",
        "mom_abs_error",
        "kde_mode_abs_error",
        "dca_abs_error",
        "best_non_dca_method",
        "best_non_dca_abs_error",
        "dca_improvement_over_best_non_dca",
        "window_gap_to_best_non_dca",
        "future_variation",
        "candidate_spread_at_h",
        "candidate_spread_peak",
        "local_peak_count",
        "selection_score",
        "pred_start_time",
        "pred_end_time",
    ]
    with save_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in top20:
            writer.writerow({k: row[k] for k in csv_fields})

    point_predictions = {
        "single": float(pred_single_all[bi, hi, ni]),
        "mean": float(pred_mean_all[bi, hi, ni]),
        "median": float(pred_median_all[bi, hi, ni]),
        "mom": float(pred_mom_all[bi, hi, ni]),
        "kde_mode": float(pred_kde_all[bi, hi, ni]),
        "dca": float(pred_dca_all[bi, hi, ni]),
    }
    point_errors = {
        "single": float(abs_single[bi, hi, ni]),
        "mean": float(abs_mean[bi, hi, ni]),
        "median": float(abs_median[bi, hi, ni]),
        "mom": float(abs_mom[bi, hi, ni]),
        "kde_mode": float(abs_kde[bi, hi, ni]),
        "dca": float(abs_dca[bi, hi, ni]),
    }

    metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "seed": int(args.seed),
        "K": int(args.k),
        "cache_npz": str(cache_path),
        "selection_csv": str(save_csv),
        "figure_pdf": str(save_pdf),
        "figure_png": str(save_png),
        "selected_case": selected,
        "point_metrics": {
            "sample_index": bi,
            "node_index": ni,
            "node_id": str(meta["station_ids"][ni]),
            "prediction_step": int(hi + 1),
            "point_time": str(meta["time_index"][int(meta["pred_starts"][bi]) + int(hi)]),
            "ground_truth": float(true_all[bi, hi, ni]),
            "predictions": point_predictions,
            "absolute_errors": point_errors,
            "best_non_dca_method": selected["best_non_dca_method"],
            "best_non_dca_abs_error": float(selected["best_non_dca_abs_error"]),
            "dca_improvement_over_best_non_dca": float(selected["dca_improvement_over_best_non_dca"]),
        },
        "candidate_values": [float(x) for x in cand_all[bi, :, hi, ni]],
        "candidate_dca_weights": [float(x) for x in dca_weights[bi, :, hi, ni]],
        "top20_candidates": top20,
    }
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "selected_sample_index": bi,
                "selected_node_index": ni,
                "selected_prediction_step_h": int(hi + 1),
                "ground_truth": float(true_all[bi, hi, ni]),
                "best_non_dca_method": selected["best_non_dca_method"],
                "dca_improvement_over_best_non_dca": float(selected["dca_improvement_over_best_non_dca"]),
                "absolute_errors": point_errors,
                "number_of_candidate_realizations": int(args.k),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


"""
论文级采样解剖可视化。

输出两类图：
1. Figure A: trend_annihilation_*  —— 精简的趋势湮灭全景图
2. Figure C: block_bias_*         —— Block 级偏差诊断图
"""
import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

FREQ_MIN = 15


def load_anatomy(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["all_raw_samples"] = np.array(d["all_raw_samples"])
    d["block_means"] = np.array(d["block_means"])
    d["mom_pred"] = np.array(d["mom_pred"])
    d["mean_pred"] = np.array(d["mean_pred"])
    d["true_raw"] = np.array(d["true_raw"])
    d["history_raw"] = np.array(d["history_raw"])
    return d


def station_view(d, col_idx):
    return {
        "samples": d["all_raw_samples"][:, :, col_idx],
        "block_means": d["block_means"][:, :, col_idx],
        "mom_pred": d["mom_pred"][:, col_idx],
        "mean_pred": d["mean_pred"][:, col_idx],
        "true_raw": d["true_raw"][:, col_idx],
        "history_raw": d["history_raw"][:, col_idx],
    }


def make_timestamps(pred_start_time_str, seq_len, pred_len):
    pred_start = pd.Timestamp(pred_start_time_str)
    freq = pd.Timedelta(minutes=FREQ_MIN)
    hist_times = pd.date_range(end=pred_start - freq, periods=seq_len, freq=freq)
    fut_times = pd.date_range(start=pred_start, periods=pred_len, freq=freq)
    return hist_times, fut_times


def pick_representative_upward_sample(samples, true_future):
    """
    从 10 条采样中挑一条“最能跟随上升趋势”的代表轨迹：
    取和 true_future 相关性最高的那条。
    """
    best_idx, best_score = 0, -1e18
    true_centered = true_future - true_future.mean()
    true_norm = np.linalg.norm(true_centered) + 1e-8
    for i in range(samples.shape[0]):
        s = samples[i]
        s_centered = s - s.mean()
        score = np.dot(s_centered, true_centered) / ((np.linalg.norm(s_centered) + 1e-8) * true_norm)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def format_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.tick_params(axis="x", rotation=0)


def plot_trend_annihilation(sc, hist_times, fut_times, sample_id, station_id, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    rep_idx = pick_representative_upward_sample(sc["samples"], sc["true_raw"])
    rep_sample = sc["samples"][rep_idx]

    bridge_x = [hist_times[-1], fut_times[0]]
    split_time = hist_times[-1]

    # prediction region shading
    ax.axvspan(fut_times[0], fut_times[-1], color="#D9EAF7", alpha=0.18, zorder=0)

    # history and future gt
    ax.plot(hist_times, sc["history_raw"], color="#1F4E79", linewidth=2.5, label="History", zorder=4)
    ax.plot(bridge_x, [sc["history_raw"][-1], sc["true_raw"][0]], color="#1F4E79", linestyle="--", linewidth=2.5, zorder=4)
    ax.plot(fut_times, sc["true_raw"], color="#1F4E79", linestyle="--", linewidth=2.5, label="Ground Truth", zorder=5)

    # representative good sample
    ax.plot(bridge_x, [sc["history_raw"][-1], rep_sample[0]], color="#6FA8DC", linewidth=1.8, alpha=0.8, zorder=2)
    ax.plot(fut_times, rep_sample, color="#6FA8DC", linewidth=1.8, alpha=0.8, label="Representative sample", zorder=2)

    # MoM pred
    ax.plot(bridge_x, [sc["history_raw"][-1], sc["mom_pred"][0]], color="#C00000", linewidth=3.0, zorder=6)
    ax.plot(fut_times, sc["mom_pred"], color="#C00000", linewidth=3.0, label="MoM (median)", zorder=6)

    # optional simple mean as subtle reference
    ax.plot(fut_times, sc["mean_pred"], color="#548235", linewidth=1.8, linestyle="-.", alpha=0.85, label="Simple Mean", zorder=3)

    # underestimation area
    ax.fill_between(
        fut_times,
        sc["mom_pred"],
        sc["true_raw"],
        where=(sc["true_raw"] > sc["mom_pred"]),
        interpolate=True,
        color="#C00000",
        alpha=0.14,
        label="Underestimation",
        zorder=1,
    )

    # annotate largest gap
    gap = sc["true_raw"] - sc["mom_pred"]
    max_gap_idx = int(np.argmax(gap))
    ax.annotate(
        f"Gap: {gap[max_gap_idx]:.0f}",
        xy=(fut_times[max_gap_idx], sc["mom_pred"][max_gap_idx]),
        xytext=(fut_times[max_gap_idx], sc["mom_pred"][max_gap_idx] + max(40.0, gap[max_gap_idx] * 0.35)),
        arrowprops=dict(arrowstyle="->", color="#C00000", lw=1.2),
        fontsize=10,
        color="#C00000",
        zorder=7,
    )

    ax.axvline(split_time, color="black", linestyle=":", linewidth=1.4, alpha=0.75, label="Prediction start")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow (veh/15min)")
    ax.set_title(f"Trend Annihilation: MoM Loses Upward Momentum\nSample {sample_id} | Station {station_id}", pad=10)
    ax.legend(loc="upper left", framealpha=0.92, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    format_time_axis(ax)
    fig.tight_layout()

    out_png = fig_dir / f"trend_annihilation_sample{sample_id}_station{station_id}.png"
    out_pdf = fig_dir / f"trend_annihilation_sample{sample_id}_station{station_id}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_png}")


def plot_block_bias(sc, fut_times, sample_id, station_id, fig_dir):
    n_blocks = sc["block_means"].shape[0]
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.10},
    )

    # top: GT context
    ax = axes[0]
    ax.plot(fut_times, sc["true_raw"], color="#1F4E79", linewidth=2.5, linestyle="--")
    ax.set_ylabel("GT\nFlow")
    ax.set_title(f"Sample {sample_id} | Station {station_id} | Congested | Holiday", pad=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    # bottom: block bias
    ax = axes[1]
    colors = ["#4472C4", "#70AD47", "#C55A11", "#7F7F7F", "#2E75B6"]
    true_future = sc["true_raw"]
    for b in range(n_blocks):
        bias = sc["block_means"][b] - true_future
        ax.plot(fut_times, bias, color=colors[b % len(colors)], linewidth=1.7, alpha=0.88, label=f"Block {b+1} bias")

    median_bias = sc["mom_pred"] - true_future
    ax.plot(fut_times, median_bias, color="#C00000", linewidth=3.0, label="MoM bias")
    mean_bias = sc["mean_pred"] - true_future
    ax.plot(fut_times, mean_bias, color="#548235", linewidth=2.0, linestyle="-.", label="Mean bias")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.55)
    ax.fill_between(fut_times, 0, median_bias, where=(median_bias < 0), alpha=0.18, color="#C00000")

    ax.set_xlabel("Time")
    ax.set_ylabel("Bias (Pred - True)")
    ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    format_time_axis(ax)
    fig.tight_layout()

    out_png = fig_dir / f"block_bias_sample{sample_id}_station{station_id}.png"
    out_pdf = fig_dir / f"block_bias_sample{sample_id}_station{station_id}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=str(ROOT / "diagnosis" / "macro" / "output" / "anatomy"))
    p.add_argument("--figures_dir", default=str(ROOT / "diagnosis" / "macro" / "figures" / "anatomy"))
    p.add_argument("--sample_ids", default="3817,2475,3819,2474")
    p.add_argument("--overlays_csv", default=str(ROOT / "diagnosis" / "macro" / "output" / "masked" / "step1_worst_case_overlays.csv"))
    c = p.parse_args()

    in_dir = Path(c.input_dir)
    fig_dir = Path(c.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [int(x) for x in c.sample_ids.split(",")]
    overlays = pd.read_csv(c.overlays_csv) if Path(c.overlays_csv).exists() else pd.DataFrame()

    for sid in sample_ids:
        json_path = in_dir / f"anatomy_sample{sid}.json"
        if not json_path.exists():
            print(f"[skip] {json_path} not found")
            continue

        d = load_anatomy(json_path)
        sub = overlays[overlays["sample_id"] == sid].sort_values("node_rank") if len(overlays) else pd.DataFrame()
        if not len(sub):
            print(f"[skip] no overlay mapping for sample {sid}")
            continue

        pred_start = sub["pred_start_time"].iloc[0]
        seq_len = d["history_raw"].shape[0]
        pred_len = d["true_raw"].shape[0]
        hist_times, fut_times = make_timestamps(pred_start, seq_len, pred_len)

        # 只生成论文最需要的每个 sample 的最差站点图
        col_idx = int(sub["node_index_pos"].iloc[0])
        station_id = int(sub["station_index"].iloc[0])
        sc = station_view(d, col_idx)

        plot_trend_annihilation(sc, hist_times, fut_times, sid, station_id, fig_dir)
        plot_block_bias(sc, fut_times, sid, station_id, fig_dir)

    print(f"Done. Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()

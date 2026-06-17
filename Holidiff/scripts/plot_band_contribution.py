from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# ============================================================
# Academic colors
# ============================================================
BAND_COLORS = {
    1: "#5F88C2",   # blue
    2: "#6F9660",   # sage green
    3: "#D89436",   # orange
    4: "#7557B2",   # purple
}

METRIC_COLORS = {
    "all": "#5F88C2",
    "dyn": "#D89436",
}

TEXT_COLOR = "#202020"
GRID_COLOR = "#E9E9E9"
SPINE_COLOR = "#555555"
NA_FACE = "#F6F6F6"
NA_EDGE = "#BFBFBF"


# ============================================================
# Data loading
# ============================================================
def load_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    parsed = []

    for r in rows:
        dyn_raw = r.get("contribution_dynamic", "")
        contribution_dynamic = (
            None
            if dyn_raw in ("", "None", "none", "null", None)
            else float(dyn_raw)
        )

        mae_full_raw = r.get("mae_full", "")
        mae_removed_raw = r.get("mae_removed", "")
        delta_raw = r.get("delta", "") or r.get("delta_mae", "")

        parsed.append(
            {
                "band": int(r["band"]),
                "interpretation": r["interpretation"],
                "mae_full": None if mae_full_raw in ("", None) else float(mae_full_raw),
                "mae_removed": None if mae_removed_raw in ("", None) else float(mae_removed_raw),
                "delta": float(delta_raw),
                "contribution_all": float(r["contribution_all"]),
                "contribution_dynamic": contribution_dynamic,
                "is_dynamic_band": str(r.get("is_dynamic_band", "")).lower() == "true",
            }
        )

    return sorted(parsed, key=lambda x: x["band"])


# ============================================================
# Helpers
# ============================================================
def style_axis(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.tick_params(labelsize=8.5, colors=TEXT_COLOR)


def band_role(band: int) -> str:
    return {
        1: "Baseline",
        2: "Rhythm",
        3: "Transition",
        4: "Disturbance",
    }[band]


# ============================================================
# Panel (a): Delta_k
# ============================================================
def draw_delta_panel(ax, rows: list[dict]) -> None:
    style_axis(ax, grid_axis="y")

    x = np.arange(len(rows))
    bands = [r["band"] for r in rows]
    deltas = [max(r["delta"], 1e-8) for r in rows]
    colors = [BAND_COLORS[b] for b in bands]

    bars = ax.bar(
        x,
        deltas,
        width=0.58,
        color=colors,
        alpha=0.84,
        edgecolor="white",
        linewidth=1.0,
        zorder=2,
    )

    ax.scatter(
        x,
        deltas,
        s=28,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_ylim(0.035, max(deltas) * 1.85)

    ax.set_title(
        r"(a) Error increment",
        fontsize=10.7,
        fontweight="semibold",
        color=TEXT_COLOR,
        pad=8,
        loc="center",
    )

    ax.set_ylabel(r"Error increase $\Delta_k$ (log scale)", fontsize=9.4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [rf"$k={b}$" + "\n" + band_role(b) for b in bands],
        fontsize=8.0,
    )

    for xi in x:
        ax.axvline(xi, color="#F4F4F4", lw=0.6, zorder=0)

    for rect, r in zip(bars, rows):
        if r["band"] == 1:
            label = f"{r['delta']:.2f}"
            y = r["delta"] * 1.15
        else:
            label = f"{r['delta']:.3f}"
            y = max(r["delta"], 1e-8) * 1.25

        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=TEXT_COLOR,
        )


# ============================================================
# Panel (b): C_all and C_dyn together
# ============================================================
def draw_contribution_panel(ax, rows: list[dict]) -> None:
    style_axis(ax, grid_axis="y")

    x = np.arange(len(rows))
    width = 0.28

    bands = [r["band"] for r in rows]
    c_all = [r["contribution_all"] for r in rows]
    c_dyn = [
        np.nan if r["contribution_dynamic"] is None else r["contribution_dynamic"]
        for r in rows
    ]

    # C_all bars
    bars_all = ax.bar(
        x - width / 2,
        c_all,
        width=width,
        color=METRIC_COLORS["all"],
        alpha=0.84,
        edgecolor="white",
        linewidth=1.0,
        label=r"$C_k^{\mathrm{all}}$",
        zorder=2,
    )

    # C_dyn bars
    dyn_heights = []
    dyn_faces = []
    dyn_edges = []
    dyn_hatches = []

    for v in c_dyn:
        if np.isnan(v):
            dyn_heights.append(0.025)  # visual placeholder
            dyn_faces.append(NA_FACE)
            dyn_edges.append(NA_EDGE)
            dyn_hatches.append("//")
        else:
            dyn_heights.append(v)
            dyn_faces.append(METRIC_COLORS["dyn"])
            dyn_edges.append("white")
            dyn_hatches.append(None)

    bars_dyn = ax.bar(
        x + width / 2,
        dyn_heights,
        width=width,
        color=dyn_faces,
        alpha=0.88,
        edgecolor=dyn_edges,
        linewidth=1.0,
        label=r"$C_k^{\mathrm{dyn}}$",
        zorder=2,
    )

    for bar, hatch in zip(bars_dyn, dyn_hatches):
        if hatch is not None:
            bar.set_hatch(hatch)

    ax.set_title(
        r"(b) Normalized contributions",
        fontsize=10.7,
        fontweight="semibold",
        color=TEXT_COLOR,
        pad=8,
        loc="center",
    )

    ax.set_ylabel("Contribution ratio", fontsize=9.4)
    ax.set_ylim(0, 1.08)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [rf"$k={b}$" + "\n" + band_role(b) for b in bands],
        fontsize=8.0,
    )

    # Labels for C_all
    for rect, r in zip(bars_all, rows):
        val = r["contribution_all"]
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.025,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=TEXT_COLOR,
        )

    # Labels for C_dyn
    for rect, r in zip(bars_dyn, rows):
        if r["contribution_dynamic"] is None:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                0.060,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="semibold",
                color="#777777",
            )
        else:
            val = r["contribution_dynamic"]
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                val + 0.025,
                f"{val * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=TEXT_COLOR,
            )

    legend_handles = [
        Patch(
            facecolor=METRIC_COLORS["all"],
            edgecolor="white",
            label=r"$C_k^{\mathrm{all}}$",
            alpha=0.84,
        ),
        Patch(
            facecolor=METRIC_COLORS["dyn"],
            edgecolor="white",
            label=r"$C_k^{\mathrm{dyn}}$",
            alpha=0.88,
        ),
        Patch(
            facecolor=NA_FACE,
            edgecolor=NA_EDGE,
            hatch="//",
            label="N/A",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=8.3,
        handlelength=1.4,
        handletextpad=0.5,
        borderaxespad=0.3,
    )


# ============================================================
# Main plot
# ============================================================
def plot(rows: list[dict], save_pdf: Path, save_png: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.6, 3.45),
        facecolor="white",
    )

    draw_delta_panel(axes[0], rows)
    draw_contribution_panel(axes[1], rows)

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.23,
        wspace=0.14,
    )

    save_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(save_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# Caption helper
# ============================================================
def print_caption() -> None:
    print(
        r"""\caption{Frequency-band sensitivity analysis on Fujian-30. 
Subfigure (a) reports the error increment $\Delta_k$ after removing each frequency band. 
Subfigure (b) compares the all-band normalized contribution $C_k^{\mathrm{all}}$ and the dynamic-band normalized contribution $C_k^{\mathrm{dyn}}$ under each frequency band. 
The value of $C_1^{\mathrm{dyn}}$ is marked as N/A because dynamic-band normalization is defined only within $k=2,3,4$.}"""
    )


# ============================================================
# Entry
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot frequency-band sensitivity metrics."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/band_contribution/band_contribution_H12.csv",
    )
    parser.add_argument(
        "--save_pdf",
        type=str,
        default="figures/band.pdf",
    )
    parser.add_argument(
        "--save_png",
        type=str,
        default="figures/band.png",
    )

    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    csv_path = csv_path if csv_path.is_absolute() else Path.cwd() / csv_path
    rows = load_rows(csv_path)

    save_pdf = Path(args.save_pdf)
    save_pdf = save_pdf if save_pdf.is_absolute() else Path.cwd() / save_pdf

    save_png = Path(args.save_png)
    save_png = save_png if save_png.is_absolute() else Path.cwd() / save_png

    plot(rows, save_pdf, save_png)
    print_caption()


if __name__ == "__main__":
    main()
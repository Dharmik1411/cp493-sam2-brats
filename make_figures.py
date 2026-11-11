"""Figures for the report, from the results files (150 test patients, whole tumour).

python make_figures.py      writes figures/fig_boxes.pdf and figures/fig_paired.pdf (plus .png previews)
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ZERO_SHOT, LORA = "#eb6834", "#2a78d6"  # validated pair (dataviz reference palette, slots 2 and 1)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def dice(name):
    return {r["case"]: float(r["dice"]) for r in csv.DictReader(open(f"results/{name}.csv")) if r["region"] == "WT"}


def boxes(name):
    return np.mean([float(r["n_boxes"]) for r in csv.DictReader(open(f"results/{name}.csv")) if r["region"] == "WT"])


def ci(x, rng):
    m = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def fig_boxes(rng):
    """Mean Dice against boxes drawn per patient: what the adapter is worth in boxes."""
    runs = [("rp30_150", "Zero-shot, re-prompt every 30", ZERO_SHOT),
            ("ceiling_150", "Zero-shot, box on every slice", ZERO_SHOT),
            ("lora_encoder_rp30_150", "LoRA v1, re-prompt every 30", LORA),
            ("lora_encoder_v2_rp30_150", "LoRA v2, re-prompt every 30", LORA)]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for name, label, color in runs:
        d = np.array(list(dice(name).values()))
        x, (lo, hi) = boxes(name), ci(d, rng)
        ax.errorbar(x, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", ms=5, color=color,
                    ecolor=color, elinewidth=1.2, capsize=2.5, markeredgecolor="white", markeredgewidth=0.8)
        right = x < 10
        ax.annotate(f"{label}\n{d.mean():.3f}", (x, d.mean()), xytext=(8 if right else -8, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1.4, 110)
    ax.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Boxes drawn per patient (log scale)")
    ax.set_ylabel("Whole-tumour Dice (mean, 95% CI)")
    ax.set_ylim(0.78, 0.91)
    fig.tight_layout()
    return fig


def fig_paired(rng):
    """Per-patient Dice with and without the v2 adapter, same 2.3 boxes per patient."""
    a, b = dice("rp30_150"), dice("lora_encoder_v2_rp30_150")
    cases = sorted(set(a) & set(b))
    x, y = np.array([a[c] for c in cases]), np.array([b[c] for c in cases])
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.plot([0.3, 1], [0.3, 1], color=MUTED, linewidth=0.8, linestyle="--")
    ax.scatter(x, y, s=14, color=LORA, edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(0.33, 0.97, f"Adapter better on {(y > x).sum()} of {len(x)} patients", fontsize=7, color=INK, va="top")
    ax.text(0.645, 0.66, "no change", fontsize=6.5, color=MUTED, ha="center", va="center", rotation=45)
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0.3, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Dice, zero-shot (re-prompt every 30)")
    ax.set_ylabel("Dice, LoRA v2 (re-prompt every 30)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    rng = np.random.default_rng(0)
    for name, make in [("fig_boxes", fig_boxes), ("fig_paired", fig_paired)]:
        fig = make(rng)
        for ext in ["pdf", "png"]:
            fig.savefig(f"figures/{name}.{ext}", dpi=200)
        print("wrote", f"figures/{name}.pdf")

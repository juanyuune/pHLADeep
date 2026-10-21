"""
plot_loao.py  –  LOAO evaluation bar chart (Nature journal style)
=================================================================
Visualises Leave-One-Allele-Out (LOAO) generalisation results for
pHLADeep-Attn on B*40:01 and B*51:01.

Usage:
  python plot_loao.py \
      --csv  /srv/jupyterlab/workspace/HLA1/results/loao/loao_results_stratC_ankh.csv \
      --output /srv/jupyterlab/workspace/HLA1/results/figures/plot_loao.png
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",    type=str,
    default="/srv/jupyterlab/workspace/HLA1/results/loao/loao_results_stratC_ankh.csv")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--double_col", action="store_true")
args = parser.parse_args()

if args.output is None:
    args.output = os.path.join(os.path.dirname(args.csv),
                               "../figures/plot_loao.png")
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

# ── Nature typography ─────────────────────────────────────────────────────────
MM = 1 / 25.4
FIG_W = (180 if args.double_col else 88) * MM
FIG_H = FIG_W * 0.82

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         7,
    "axes.titlesize":    8,
    "axes.labelsize":    7,
    "xtick.labelsize":   6.5,
    "ytick.labelsize":   6,
    "legend.fontsize":   6.5,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "savefig.dpi":       300,
})

# ── Load CSV ──────────────────────────────────────────────────────────────────
rows = []
with open(args.csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "allele": row["held_out_allele"].replace("HLA-B", "HLA-B*").replace("4001","40:01").replace("5101","51:01"),
            "AUC":  float(row["AUC"]),
            "Sens": float(row["Sens"]),
            "Spec": float(row["Spec"]),
            "MCC":  float(row["MCC"]),
            "F1":   float(row["F1"]),
            "Acc":  float(row["Acc"]),
        })

alleles  = [r["allele"] for r in rows]
metrics  = ["AUC", "Sens", "Spec", "MCC"]
m_labels = ["AUC", "Sensitivity", "Specificity", "MCC"]

# ── CVD-safe palette: blue, orange (2 alleles) ───────────────────────────────
COLORS = ["#2a78d6", "#eb6834"]

n_alleles = len(alleles)
n_metrics = len(metrics)
x         = np.arange(n_metrics)
bar_w     = 0.30
offsets   = np.linspace(-(n_alleles-1)/2, (n_alleles-1)/2, n_alleles) * bar_w

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for i, (row, color) in enumerate(zip(rows, COLORS)):
    vals = [row[m] for m in metrics]
    bars = ax.bar(x + offsets[i], vals, width=bar_w,
                  color=color, alpha=0.88,
                  label=row["allele"],
                  zorder=3,
                  linewidth=0)

    # Value labels on top of each bar
    for xi, v in zip(x + offsets[i], vals):
        va   = "bottom" if v >= 0 else "top"
        yoff = 0.012 if v >= 0 else -0.012
        ax.text(xi, v + yoff, f"{v:.3f}",
                ha="center", va=va, fontsize=5.2,
                color="#333333")

# Random-chance reference line at 0.5 for AUC / 0 for MCC
ax.axhline(0.5, color="#aaaaaa", lw=0.6, linestyle="--", zorder=1,
           label="AUC = 0.5 (random)")
ax.axhline(0.0, color="#cccccc", lw=0.5, linestyle="-",  zorder=1)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(m_labels)
ax.set_ylabel("Score", labelpad=4)
ax.set_ylim(-0.25, 1.05)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")
ax.tick_params(axis="both", which="major", color="#444444",
               labelcolor="#222222", width=0.5, length=3)
ax.tick_params(axis="y", which="minor", color="#cccccc", width=0.4, length=2)

ax.grid(True, which="major", axis="y", color="#ebebeb",
        linewidth=0.5, linestyle="-", zorder=0)
ax.set_axisbelow(True)

ax.set_title(
    "LOAO generalisation — pHLADeep-Attn (Ankh-Large)",
    fontsize=8, fontweight="bold", color="#111111", pad=6, loc="left")

# ── Legend ────────────────────────────────────────────────────────────────────
leg = ax.legend(loc="upper right", fontsize=6.5,
                framealpha=0.95, edgecolor="#dddddd",
                frameon=True, handlelength=1.2,
                handletextpad=0.5, borderpad=0.6,
                labelspacing=0.4)
leg.get_frame().set_linewidth(0.5)

fig.tight_layout(pad=0.8)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
pdf_path = args.output.replace(".png", ".pdf")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved PNG : {args.output}")
print(f"Saved PDF : {pdf_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  {'Allele':<14}  {'AUC':>7}  {'Sens':>7}  {'Spec':>7}  {'MCC':>7}")
print("-"*60)
for r in rows:
    print(f"  {r['allele']:<14}  {r['AUC']:>7.4f}  {r['Sens']:>7.4f}"
          f"  {r['Spec']:>7.4f}  {r['MCC']:>7.4f}")
print("="*60)
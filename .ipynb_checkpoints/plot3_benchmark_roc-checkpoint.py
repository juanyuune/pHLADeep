"""
plot3_benchmark_roc.py  –  ROC curves: benchmark comparison
============================================================
High-impact journal style (Nature family). Clean, modern, uncluttered.

Usage:
  python plot3_benchmark_roc.py \\
      --results_dir /path/to/results \\
      --output      /path/to/figures/plot3_benchmark_roc.png \\
      --plm_name    ankh \\
      [--double_col]
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str,
                    default="/srv/jupyterlab/workspace/HLA1/results")
parser.add_argument("--output",      type=str, default=None)
parser.add_argument("--plm_name",    type=str, default="ankh")
parser.add_argument("--double_col",  action="store_true")
args = parser.parse_args()

if args.output is None:
    args.output = os.path.join(args.results_dir, "benchmark",
                               "plot3_benchmark_roc.png")
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED STYLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MM_TO_IN = 1 / 25.4
FIG_W    = 180 * MM_TO_IN if args.double_col else 88 * MM_TO_IN
FIG_H    = FIG_W * 0.92

RAND_COLOR = "#BBBBBB"
RAND_LW    = 1.0
RAND_DASH  = (4, 3)

GRID_COLOR = "#F0F0F0"
GRID_LW    = 0.6

matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "lines.linewidth":    2.2,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "savefig.dpi":        300,
    "figure.dpi":         150,
})

# ── Curve definitions ──────────────────────────────────────────────────────────
# Baselines: muted tones (grey-adjacent), solid lines
# pHLADeep: vivid palette, solid lines — best model slightly thicker
# All solid — colour alone differentiates; no dashes (cleaner, more modern)
CURVES = [
    {
        "label": "NetMHCpan 4.2",
        "pkl":   os.path.join(args.results_dir, "benchmark",
                              f"roc_data_NetMHCpan_42_{args.plm_name}.pkl"),
        "color": "#888888",     # neutral grey — clearly a baseline
        "lw":    1.8,
    },
    {
        "label": "MHCflurry 2.0",
        "pkl":   os.path.join(args.results_dir, "benchmark",
                              f"roc_data_MHCflurry_20_{args.plm_name}.pkl"),
        "color": "#AAAAAA",     # lighter grey — second baseline
        "lw":    1.8,
    },
    {
        "label":     "pHLADeep-Pep",
        "subfolder": "stratA",
        "color":     "#2E6FAB",  # slate blue
        "lw":        2.2,
    },
    {
        "label":     "pHLADeep-Attn",
        "subfolder": "stratC",
        "color":     "#1A9E76",  # teal
        "lw":        2.2,
    },
    {
        "label":     "pHLADeep-Concat",
        "subfolder": "stratB_concat",
        "color":     "#C85A2A",  # burnt sienna — best model, stands out
        "lw":        2.6,        # slightly thicker to highlight best model
    },
]

PLM_DISPLAY = {
    "ankh":    "Ankh-Large",
    "esm2":    "ESM-2",
    "prostt5": "ProstT5",
    "prot5xl": "ProtT5-XL",
}

def find_pkl_in_folder(folder, plm_name):
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".pkl") and plm_name.lower() in fname.lower():
            return os.path.join(folder, fname)
    pkls = [f for f in os.listdir(folder) if f.endswith(".pkl")]
    if pkls:
        print(f"  [warn] no '{plm_name}' match in {folder}, using {pkls[0]}")
        return os.path.join(folder, pkls[0])
    return None

roc_loaded = {}
print(f"\nLoading ROC pkls (plm={args.plm_name}):")
for c in CURVES:
    label = c["label"]
    if "pkl" in c:
        path = c["pkl"]
        if not os.path.exists(path):
            print(f"  SKIP {label}: pkl not found ({path})")
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
    else:
        folder = os.path.join(args.results_dir, c["subfolder"])
        if not os.path.isdir(folder):
            print(f"  SKIP {label}: folder not found ({folder})")
            continue
        path = find_pkl_in_folder(folder, args.plm_name)
        if path is None:
            print(f"  SKIP {label}: no pkl in {folder}")
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)

    fpr     = np.array(d["fpr"])
    tpr     = np.array(d["tpr"])
    auc_val = float(d["auc"])
    roc_loaded[label] = (fpr, tpr, auc_val)
    print(f"  {label:<22}  AUC={auc_val:.4f}  ({os.path.basename(path)})")

if not roc_loaded:
    raise RuntimeError("No ROC data loaded — check paths.")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

legend_lines  = []
legend_labels = []

for c in CURVES:
    label = c["label"]
    if label not in roc_loaded:
        continue
    fpr, tpr, auc_val = roc_loaded[label]
    line, = ax.plot(fpr, tpr, color=c["color"], lw=c["lw"],
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)
    legend_lines.append(line)
    legend_labels.append(f"{label}   AUC = {auc_val:.4f}")

# Random diagonal
diag, = ax.plot([0, 1], [0, 1], color=RAND_COLOR, lw=RAND_LW,
                dashes=RAND_DASH, zorder=1)
legend_lines.append(diag)
legend_labels.append("Random classifier   AUC = 0.5000")

# ── Axes ───────────────────────────────────────────────────────────────────────
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_xlabel("False positive rate (1 – specificity)", labelpad=5)
ax.set_ylabel("True positive rate (sensitivity)",           labelpad=5)

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["left"].set_color("#333333")
ax.spines["bottom"].set_color("#333333")
ax.tick_params(axis="both", which="major", color="#333333",
               labelcolor="#222222", width=0.8, length=4, direction="out")
ax.tick_params(axis="both", which="minor", color="#CCCCCC",
               width=0.5, length=2.5, direction="out")

ax.yaxis.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LW,
              linestyle="-", zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

plm_display = PLM_DISPLAY.get(args.plm_name.lower(), args.plm_name)
ax.set_title(f"Benchmark comparison ({plm_display})",
             fontsize=11, fontweight="bold", color="#111111",
             pad=8, loc="left")

# ── Legend — frameless ─────────────────────────────────────────────────────────
leg = ax.legend(legend_lines, legend_labels,
                loc="lower right",
                fontsize=9,
                frameon=False,
                handlelength=2.0,
                handletextpad=0.6,
                borderpad=0,
                labelspacing=0.5,
                labelcolor="#222222")

fig.tight_layout(pad=1.0)

# ── Save ───────────────────────────────────────────────────────────────────────
fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
pdf_path = args.output.replace(".png", ".pdf")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nSaved PNG : {args.output}")
print(f"Saved PDF : {pdf_path}")
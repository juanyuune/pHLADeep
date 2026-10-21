"""
plot2_model_comparison.py  –  ROC curves: pHLADeep model comparison
=====================================================================
High-impact journal style (Nature family). Clean, modern, uncluttered.

Usage:
  python plot2_model_comparison.py \\
      --results_dir /path/to/results \\
      --plm_name    ankh \\
      --output      /path/to/figures/plot2_model_comparison.png \\
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
from matplotlib.lines import Line2D

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str,
                    default="/srv/jupyterlab/workspace/HLA1/results")
parser.add_argument("--plm_name",   type=str, default="ankh")
parser.add_argument("--output",     type=str, default=None)
parser.add_argument("--double_col", action="store_true")
args = parser.parse_args()

if args.output is None:
    args.output = os.path.join(args.results_dir, "plot2_model_comparison.png")
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

# Slate blue, Burnt sienna, Teal — same palette as plot1
MODELS = [
    {
        "label":     "pHLADeep-Pep",
        "subfolder": "stratA",
        "color":     "#2E6FAB",
        "lw":        2.2,
    },
    {
        "label":     "pHLADeep-Concat",
        "subfolder": "stratB_concat",
        "color":     "#C85A2A",
        "lw":        2.2,
    },
    {
        "label":     "pHLADeep-Attn",
        "subfolder": "stratC",
        "color":     "#1A9E76",
        "lw":        2.2,
    },
]

PLM_DISPLAY = {
    "ankh":    "Ankh-Large",
    "esm2":    "ESM-2",
    "prostt5": "ProstT5",
    "prot5xl": "ProtT5-XL",
}

def find_pkl(folder, plm_name):
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".pkl") and plm_name.lower() in fname.lower():
            return os.path.join(folder, fname)
    pkls = [f for f in os.listdir(folder) if f.endswith(".pkl")]
    if pkls:
        print(f"  [warn] no '{plm_name}' match in {folder}, using {pkls[0]}")
        return os.path.join(folder, pkls[0])
    raise FileNotFoundError(f"No pkl found in {folder}")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

plm_display = PLM_DISPLAY.get(args.plm_name.lower(), args.plm_name)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

legend_lines  = []
legend_labels = []
summary       = []

print(f"\nLoading pkl files (plm={args.plm_name}):")

for m in MODELS:
    folder = os.path.join(args.results_dir, m["subfolder"])
    if not os.path.isdir(folder):
        print(f"  SKIP {m['label']}: folder not found ({folder})")
        continue
    try:
        pkl_path = find_pkl(folder, args.plm_name)
    except FileNotFoundError as e:
        print(f"  SKIP {m['label']}: {e}")
        continue

    d      = load_pkl(pkl_path)
    fpr    = np.array(d["fpr"])
    tpr    = np.array(d["tpr"])
    auc    = float(d["auc"])
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix     = int(d.get("gmean_ix", np.argmax(gmeans)))
    gmean  = float(gmeans[ix])

    line, = ax.plot(fpr, tpr, color=m["color"], lw=m["lw"],
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)

    ax.scatter(fpr[ix], tpr[ix], s=60, color=m["color"], zorder=5,
               edgecolors="white", linewidths=1.8)

    legend_lines.append(line)
    legend_labels.append(f"{m['label']}   AUC = {auc:.4f}")
    summary.append((m["label"], auc, gmean))
    print(f"  {m['label']:<22}  AUC={auc:.4f}  G-mean={gmean:.4f}")

if not summary:
    raise RuntimeError("No data loaded — check --results_dir and pkl files.")

# Random diagonal
diag, = ax.plot([0, 1], [0, 1], color=RAND_COLOR, lw=RAND_LW,
                dashes=RAND_DASH, zorder=1)
legend_lines.append(diag)
legend_labels.append("Random classifier   AUC = 0.5000")

# G-mean legend proxy
gmean_handle = Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#555555", markeredgecolor="white",
                      markeredgewidth=1.8, markersize=7)
legend_lines.append(gmean_handle)
legend_labels.append("G-mean optimal threshold")

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

ax.set_title(f"pHLADeep model comparison ({plm_display})",
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

print("\n" + "=" * 65)
print(f"  {'Model':<22}  {'AUC':>7}  {'G-mean':>7}")
print("-" * 65)
for label, auc, gmean in summary:
    print(f"  {label:<22}  {auc:>7.4f}  {gmean:>7.4f}")
print("=" * 65)
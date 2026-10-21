"""
plot1_plm_comparison.py  –  ROC curves: PLM embedding comparison
=================================================================
High-impact journal style (Nature family). Clean, modern, uncluttered.

Usage:
  python plot1_plm_comparison.py \\
      --pkl_dir /path/to/stratA \\
      --output  /path/to/figures/plot1_plm_comparison.png \\
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
parser.add_argument("--pkl_dir",    type=str,
                    default="/srv/jupyterlab/workspace/HLA1/results/stratA")
parser.add_argument("--output",     type=str, default=None)
parser.add_argument("--double_col", action="store_true")
args = parser.parse_args()

if args.output is None:
    args.output = os.path.join(args.pkl_dir, "plot1_plm_comparison.png")
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED STYLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MM_TO_IN = 1 / 25.4
FIG_W    = 180 * MM_TO_IN if args.double_col else 88 * MM_TO_IN
FIG_H    = FIG_W * 0.92

# Curated palette — distinct in both color and greyscale print
# Slate blue, Burnt sienna, Teal, Plum
COLORS = ["#2E6FAB", "#C85A2A", "#1A9E76", "#7B4FA6"]

# Lines: solid only — differentiated by color + marker, no dashes
# (dashes look dated in modern high-impact figures)
LINEWIDTHS = [2.2, 2.2, 2.2, 2.2]

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

PLM_MAP = [
    ("ankh",    "Ankh-Large"),
    ("esm2",    "ESM-2"),
    ("prostt5", "ProstT5"),
    ("prot5xl", "ProtT5-XL"),
    ("prott5",  "ProtT5"),
    ("prot_",   "ProtTrans"),
]

def find_pkls(folder):
    found = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".pkl"):
            continue
        for key, label in PLM_MAP:
            if key in fname.lower() and label not in found:
                found[label] = os.path.join(folder, fname)
                break
    return [(label, found[label]) for _, label in PLM_MAP if label in found]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

entries = find_pkls(args.pkl_dir)
if not entries:
    raise FileNotFoundError(f"No recognised pkl files in {args.pkl_dir}")

print(f"\nFound {len(entries)} PLM result(s):")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

legend_lines  = []
legend_labels = []
summary       = []

for i, (label, pkl_path) in enumerate(entries):
    d      = load_pkl(pkl_path)
    fpr    = np.array(d["fpr"])
    tpr    = np.array(d["tpr"])
    auc    = float(d["auc"])
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix     = int(d.get("gmean_ix", np.argmax(gmeans)))
    gmean  = float(gmeans[ix])

    color = COLORS[i % len(COLORS)]

    line, = ax.plot(fpr, tpr, color=color, lw=2.2,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)

    # G-mean marker: clean filled circle, thin white ring
    ax.scatter(fpr[ix], tpr[ix], s=60, color=color, zorder=5,
               edgecolors="white", linewidths=1.8)

    legend_lines.append(line)
    legend_labels.append(f"{label}   AUC = {auc:.4f}")
    summary.append((label, auc, gmean, os.path.basename(pkl_path)))
    print(f"  {label:<16}  AUC={auc:.4f}  G-mean={gmean:.4f}")

# Random diagonal — light, clearly secondary
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

# Clean open box — only left and bottom spines
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

# Minimal recessive grid — horizontal only, very light
ax.yaxis.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LW,
              linestyle="-", zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Bold left-aligned title
ax.set_title("PLM embedding comparison (pHLADeep-Pep)",
             fontsize=11, fontweight="bold", color="#111111",
             pad=8, loc="left")

# ── Legend — frameless, inside plot ────────────────────────────────────────────
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

print("\n" + "=" * 55)
print(f"  {'PLM':<16}  {'AUC':>7}  {'G-mean':>7}")
print("-" * 55)
for row in sorted(summary, key=lambda x: -x[1]):
    print(f"  {row[0]:<16}  {row[1]:>7.4f}  {row[2]:>7.4f}")
print("=" * 55)
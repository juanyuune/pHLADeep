import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PKL_PATH = "/srv/jupyterlab/workspace/HLA1/results/casestudy_ms/casestudy_scores.pkl"

with open(PKL_PATH, "rb") as f:
    d = pickle.load(f)

df      = d["df"].copy()
y_true  = np.array(d["y_true"])
scores  = d["scores"]   # dict: method_name -> np.array shape (220,)

METHODS = list(scores.keys())
print("Methods found:", METHODS)
print(f"Total rows: {len(df)}  |  Positives: {y_true.sum()}  |  Negatives: {(y_true==0).sum()}")
print()

df["y_true"] = y_true
for m in METHODS:
    df[m] = scores[m]


def safe_auc(y, s):
    # convert to plain numpy to avoid pandas index misalignment
    y = np.array(y, dtype=float)
    s = np.array(s, dtype=float)
    mask = ~np.isnan(s)
    y, s = y[mask], s[mask]
    if y.sum() < 2 or (y == 0).sum() < 1:
        return np.nan
    return roc_auc_score(y, s)


print("=" * 70)
print("TABLE 10b — LENGTH-STRATIFIED AUC")
print("=" * 70)

lengths = sorted(df["length"].unique())
header  = f"{'Length':<10} {'N_pos':>6} {'N_neg':>6}  " + \
          "  ".join(f"{m[:18]:>18}" for m in METHODS)
print(header)
print("-" * len(header))

length_rows = []
for L in lengths:
    sub   = df[df["length"] == L]
    n_pos = int(sub["y_true"].sum())
    n_neg = int((sub["y_true"] == 0).sum())
    aucs  = []
    for m in METHODS:
        a = safe_auc(sub["y_true"], sub[m])
        aucs.append(a)
    row = f"{str(L)+'-mer':<10} {n_pos:>6} {n_neg:>6}  " + \
          "  ".join(f"{'N/A':>18}" if np.isnan(a) else f"{a:>18.4f}" for a in aucs)
    print(row)
    length_rows.append((L, n_pos, n_neg, aucs))

print("-" * len(header))
all_aucs = []
for m in METHODS:
    a = safe_auc(df["y_true"], df[m])
    all_aucs.append(a)
row = f"{'All':10} {int(y_true.sum()):>6} {int((y_true==0).sum()):>6}  " + \
      "  ".join(f"{'N/A':>18}" if np.isnan(a) else f"{a:>18.4f}" for a in all_aucs)
print(row)
print()

print("=" * 70)
print("TABLE — ALLELE-STRATIFIED AUC")
print("=" * 70)

alleles = sorted(df["allele"].unique())
header2 = f"{'Allele':<16} {'N_pos':>6} {'N_neg':>6}  " + \
          "  ".join(f"{m[:18]:>18}" for m in METHODS)
print(header2)
print("-" * len(header2))

for allele in alleles:
    sub   = df[df["allele"] == allele]
    n_pos = int(sub["y_true"].sum())
    n_neg = int((sub["y_true"] == 0).sum())
    aucs  = []
    for m in METHODS:
        a = safe_auc(sub["y_true"], sub[m])
        aucs.append(a)
    row = f"{allele:<16} {n_pos:>6} {n_neg:>6}  " + \
          "  ".join(f"{'N/A':>18}" if np.isnan(a) else f"{a:>18.4f}" for a in aucs)
    print(row)
print()

print("=" * 70)
print("ALLELE × LENGTH AUC (for A*03:01 breakdown)")
print("=" * 70)

for allele in alleles:
    for L in lengths:
        sub   = df[(df["allele"] == allele) & (df["length"] == L)]
        n_pos = int(sub["y_true"].sum())
        n_neg = int((sub["y_true"] == 0).sum())
        if n_pos == 0:
            continue
        aucs = []
        for m in METHODS:
            a = safe_auc(sub["y_true"], sub[m])
            aucs.append(a)
        row = f"{allele} {L}-mer  n_pos={n_pos} n_neg={n_neg}  " + \
              "  ".join(f"{'N/A':>10}" if np.isnan(a) else f"{a:>10.4f}" for a in aucs)
        print(row)
print()

print("=" * 70)
print("ALL POSITIVE PEPTIDES — per-peptide scores (sorted by Concat score)")
print("=" * 70)

pos = df[df["y_true"] == 1].copy()
concat_col = [m for m in METHODS if "concat" in m.lower() or "Concat" in m]
concat_col = concat_col[0] if concat_col else METHODS[0]
pos = pos.sort_values(concat_col)

base_cols = ["peptide", "allele", "length"]
for extra in ["neoantigen", "source", "label"]:
    if extra in pos.columns and extra not in base_cols:
        base_cols.append(extra)
cols_to_show = base_cols + METHODS

with pd.option_context("display.max_columns", None, "display.width", 200,
                        "display.float_format", "{:.3f}".format):
    print(pos[cols_to_show].to_string(index=False))
print()

print("=" * 70)
print("COUNTERFACTUAL: A*03:01 AUC excluding KRAS 10-mers")
print("=" * 70)

kras_10mer = (df["length"] == 10) & (df["allele"].str.contains("03:01"))
sub_no_kras = df[(df["allele"].str.contains("03:01")) & ~kras_10mer]
n_pos = int(sub_no_kras["y_true"].sum())
n_neg = int((sub_no_kras["y_true"] == 0).sum())
print(f"A*03:01 excluding 10-mers: N_pos={n_pos}, N_neg={n_neg}")
for m in METHODS:
    a = safe_auc(sub_no_kras["y_true"], sub_no_kras[m])
    tag = "N/A" if np.isnan(a) else f"{a:.4f}"
    print(f"  {m}: {tag}")
print()

print("Done. Copy the values above into Tables 10b and 10c.")

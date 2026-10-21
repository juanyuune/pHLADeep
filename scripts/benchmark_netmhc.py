"""
benchmark_netmhc.py  –  Benchmark vs NetMHCpan 4.2 and MHCflurry 2.0
======================================================================
Compares pHLADeep-Pep / pHLADeep-Concat / pHLADeep-Attn (pkl results)
against NetMHCpan 4.2 and MHCflurry 2.0 on the same held-out test sets.

Peptide sequences are extracted from FASTA filenames in the expanded
data directory, e.g.:
  .../HLA-A0201/pos_test/HLA-A0201_GILGFVFTL.fasta  → peptide GILGFVFTL, label 1
  .../HLA-A0201/neg_test/HLA-A0201_AAAPRTFLL.fasta  → peptide AAAPRTFLL, label 0

NOW SAVES ROC pkls for NetMHCpan and MHCflurry so they can be plotted
with full fpr/tpr curves (not just summary AUC).

Usage:
  conda activate pytorch-env
  python benchmark_netmhc.py \
      --data_path   /srv/jupyterlab/workspace/HLA1/data/expanded \
      --results_dir /srv/jupyterlab/workspace/HLA1/results \
      --output      /srv/jupyterlab/workspace/HLA1/results/benchmark \
      --plm_name    ankh \
      [--netmhcpan_path /path/to/netMHCpan] \
      [--skip_netmhcpan] \
      [--skip_mhcflurry]
"""

import os
import sys
import csv
import math
import pickle
import logging
import argparse
import tempfile
import subprocess

import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.metrics import roc_curve

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",      type=str, required=True)
parser.add_argument("--results_dir",    type=str, required=True)
parser.add_argument("--output",         type=str, required=True)
parser.add_argument("--plm_name",       type=str, default="ankh")
parser.add_argument("--netmhcpan_path", type=str, default="netMHCpan")
parser.add_argument("--skip_netmhcpan", action="store_true")
parser.add_argument("--skip_mhcflurry", action="store_true")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

ALLELE_MAP = {
    "HLA-A0101": ("HLA-A01:01",  "HLA-A*01:01"),
    "HLA-A0201": ("HLA-A02:01",  "HLA-A*02:01"),
    "HLA-A0301": ("HLA-A03:01",  "HLA-A*03:01"),
    "HLA-A1101": ("HLA-A11:01",  "HLA-A*11:01"),
    "HLA-A2402": ("HLA-A24:02",  "HLA-A*24:02"),
    "HLA-B0702": ("HLA-B07:02",  "HLA-B*07:02"),
    "HLA-B0801": ("HLA-B08:01",  "HLA-B*08:01"),
    "HLA-B3501": ("HLA-B35:01",  "HLA-B*35:01"),
    "HLA-B4001": ("HLA-B40:01",  "HLA-B*40:01"),
    "HLA-B5101": ("HLA-B51:01",  "HLA-B*51:01"),
    "HLA-B5701": ("HLA-B57:01",  "HLA-B*57:01"),
}


def load_sequences_from_fasta_dir(allele_dir, split="test"):
    seqs, labels = [], []
    for label, subdir in [(1, f"pos_{split}"), (0, f"neg_{split}")]:
        folder = os.path.join(allele_dir, subdir)
        if not os.path.isdir(folder):
            logging.warning("  Missing folder: %s", folder)
            continue
        for fname in os.listdir(folder):
            if not fname.endswith(".fasta"):
                continue
            stem = fname[:-6]
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            peptide = parts[-1]
            if len(peptide) == 9 and peptide.isalpha():
                seqs.append(peptide.upper())
                labels.append(label)
    logging.info("  Loaded %d seqs (pos=%d neg=%d) from %s",
                 len(seqs), sum(l==1 for l in labels),
                 sum(l==0 for l in labels), allele_dir)
    return seqs, labels


def run_netmhcpan(peptides, allele_nmhc, netmhcpan_path="netMHCpan"):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
        for i, pep in enumerate(peptides):
            f.write(f">p{i}\n{pep}\n")
        fasta_path = f.name
    out_path = fasta_path.replace(".fa", "_out.txt")
    cmd = [netmhcpan_path, "-f", fasta_path, "-a", allele_nmhc,
           "-l", "9", "-BA", "-xls", "-xlsfile", out_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logging.error("netMHCpan error:\n%s", result.stderr[:500])
            return None
    except FileNotFoundError:
        logging.error("netMHCpan not found at '%s'", netmhcpan_path)
        return None
    except subprocess.TimeoutExpired:
        logging.error("netMHCpan timed out for %s", allele_nmhc)
        return None
    ranks = []
    try:
        with open(out_path) as f:
            lines = f.readlines()
        header = None
        for line in lines:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if any(c.strip() in ("EL_rank", "Rank_EL", "%Rank_EL", "%Rank") for c in cols):
                header = [c.strip() for c in cols]
                continue
            if header and len(cols) >= 2:
                try:
                    ri = next(i for i, h in enumerate(header)
                              if h in ("EL_rank", "Rank_EL", "%Rank_EL", "%Rank"))
                    ranks.append(float(cols[ri]))
                except (StopIteration, ValueError):
                    continue
    except Exception as e:
        logging.error("Failed to parse netMHCpan output: %s", e)
        return None
    finally:
        for p in [fasta_path, out_path]:
            try: os.unlink(p)
            except OSError: pass
    if len(ranks) != len(peptides):
        logging.warning("  netMHCpan: expected %d scores, got %d", len(peptides), len(ranks))
    return np.array(ranks) if ranks else None


def run_mhcflurry(peptides, allele_mhcf):
    try:
        from mhcflurry import Class1PresentationPredictor
    except ImportError:
        logging.error("mhcflurry not installed.")
        return None
    try:
        predictor = Class1PresentationPredictor.load()
        df = predictor.predict(peptides=peptides, alleles=[allele_mhcf], verbose=0)
        return df["presentation_score"].values
    except Exception as e:
        logging.error("MHCflurry error for %s: %s", allele_mhcf, e)
        return None


def compute_metrics(y_true, scores, higher_is_better=True):
    if not higher_is_better:
        scores = -np.array(scores, dtype=float)
    y_true = np.array(y_true)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc_val = sk_metrics.auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = int(np.argmax(gmeans))
    threshold = thresholds[ix]
    y_pred = (scores >= threshold).astype(int)
    cm = sk_metrics.confusion_matrix(y_true, y_pred).ravel()
    TN, FP, FN, TP = cm if len(cm) == 4 else (0, 0, 0, 0)
    Sens  = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec  = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc   = (TP + TN) / len(y_true)
    denom = math.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
    MCC   = (TP*TN - FP*FN) / denom if denom > 0 else 0.0
    F1    = 2*TP / (2*TP + FP + FN) if 2*TP + FP + FN > 0 else 0.0
    return auc_val, Sens, Spec, Acc, MCC, F1, fpr, tpr, thresholds, ix, scores


def save_roc_pkl(output_dir, method_key, plm_name, y_true, y_score,
                 fpr, tpr, thresholds, auc_val, ix):
    """Save full ROC data pkl for external tools (mirrors pHLADeep pkl format)."""
    roc_data = {
        "plm":        plm_name,
        "method":     method_key,
        "fpr":        fpr,
        "tpr":        tpr,
        "thresholds": thresholds,
        "auc":        auc_val,
        "gmean_ix":   ix,
        "threshold":  float(thresholds[ix]),
        "y_true":     np.array(y_true),
        "y_score":    np.array(y_score),
    }
    safe_key = method_key.replace(" ", "_").replace(".", "")
    pkl_path = os.path.join(output_dir, f"roc_data_{safe_key}_{plm_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(roc_data, f)
    logging.info("  Saved ROC pkl → %s", pkl_path)
    return pkl_path


def load_strategy_pkl(strat_folder, plm_name):
    for fname in sorted(os.listdir(strat_folder)):
        if fname.endswith(".pkl") and plm_name.lower() in fname.lower():
            path = os.path.join(strat_folder, fname)
            with open(path, "rb") as f:
                return pickle.load(f), fname
    pkls = [f for f in os.listdir(strat_folder) if f.endswith(".pkl")]
    if pkls:
        logging.warning("No '%s' match in %s; using %s", plm_name, strat_folder, pkls[0])
        path = os.path.join(strat_folder, pkls[0])
        with open(path, "rb") as f:
            return pickle.load(f), pkls[0]
    return None, None


def strategy_metrics_from_pkl(d):
    y_true  = np.array(d["y_true"])
    y_score = np.array(d["y_score"])
    fpr     = np.array(d["fpr"])
    tpr     = np.array(d["tpr"])
    auc_val = float(d["auc"])
    ix      = int(d.get("gmean_ix", np.argmax(np.sqrt(tpr * (1 - fpr)))))
    if "threshold" in d:
        threshold = float(d["threshold"])
    elif "thresholds" in d:
        threshold = float(np.array(d["thresholds"])[ix])
    else:
        _, _, thresholds = roc_curve(y_true, y_score)
        threshold = float(thresholds[ix])
    y_pred = (y_score >= threshold).astype(int)
    cm = sk_metrics.confusion_matrix(y_true, y_pred).ravel()
    TN, FP, FN, TP = cm if len(cm) == 4 else (0, 0, 0, 0)
    Sens  = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec  = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc   = (TP + TN) / len(y_true)
    denom = math.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
    MCC   = (TP*TN - FP*FN) / denom if denom > 0 else 0.0
    F1    = 2*TP / (2*TP + FP + FN) if 2*TP + FP + FN > 0 else 0.0
    return auc_val, Sens, Spec, Acc, MCC, F1


def main():
    logging.info("=== Benchmark: pHLADeep vs NetMHCpan 4.2 & MHCflurry 2.0 ===")
    rows = []

    # ── 1. pHLADeep pkls ──────────────────────────────────────────────────────
    MODELS = [
        ("pHLADeep-Pep",    "stratA"),
        ("pHLADeep-Concat", "stratB_concat"),
        ("pHLADeep-Attn",   "stratC"),
    ]
    logging.info("\n--- pHLADeep models (from pkl) ---")
    for model_name, subfolder in MODELS:
        folder = os.path.join(args.results_dir, subfolder)
        if not os.path.isdir(folder):
            logging.warning("Skipping %s: folder not found (%s)", model_name, folder)
            continue
        d, fname = load_strategy_pkl(folder, args.plm_name)
        if d is None:
            logging.warning("No pkl found for %s in %s", model_name, folder)
            continue
        auc_val, Sens, Spec, Acc, MCC, F1 = strategy_metrics_from_pkl(d)
        rows.append({"Method": model_name,
                     "AUC": auc_val, "Sens": Sens, "Spec": Spec,
                     "Acc": Acc, "MCC": MCC, "F1": F1})
        logging.info("  %-22s  AUC=%.4f  Sens=%.4f  Spec=%.4f  Acc=%.4f  MCC=%.4f  (pkl: %s)",
                     model_name, auc_val, Sens, Spec, Acc, MCC, fname)

    # ── 2. Collect test peptides ───────────────────────────────────────────────
    all_peptides, all_labels, allele_per_pep = [], [], []
    need_peptides = (not args.skip_netmhcpan) or (not args.skip_mhcflurry)
    if need_peptides:
        logging.info("\n--- Extracting test peptides from: %s ---", args.data_path)
        allele_tags = sorted(
            e for e in os.listdir(args.data_path)
            if os.path.isdir(os.path.join(args.data_path, e))
            and e.startswith("HLA-") and e in ALLELE_MAP
        )
        if not allele_tags:
            logging.error("No HLA-* subdirectories found in %s matching ALLELE_MAP",
                          args.data_path)
        for allele_tag in allele_tags:
            seqs, labels = load_sequences_from_fasta_dir(
                os.path.join(args.data_path, allele_tag), split="test")
            if not seqs:
                continue
            all_peptides.extend(seqs)
            all_labels.extend(labels)
            allele_per_pep.extend([allele_tag] * len(seqs))
        logging.info("Total test peptides: %d  (pos=%d  neg=%d)",
                     len(all_labels),
                     sum(l==1 for l in all_labels),
                     sum(l==0 for l in all_labels))

    # ── 3. NetMHCpan 4.2 ──────────────────────────────────────────────────────
    if not args.skip_netmhcpan and all_peptides:
        logging.info("\n--- NetMHCpan 4.2 ---")
        nmhc_scores = np.full(len(all_peptides), np.nan)
        for allele_tag in sorted(set(allele_per_pep)):
            nmhc_name = ALLELE_MAP[allele_tag][0]
            idxs = [i for i, a in enumerate(allele_per_pep) if a == allele_tag]
            peps = [all_peptides[i] for i in idxs]
            logging.info("  Running NetMHCpan for %s (%d peptides)…", allele_tag, len(peps))
            scores = run_netmhcpan(peps, nmhc_name, args.netmhcpan_path)
            if scores is not None and len(scores) == len(idxs):
                for i, s in zip(idxs, scores):
                    nmhc_scores[i] = s
            else:
                logging.warning("  Incomplete scores for %s — skipping allele", allele_tag)
        valid = ~np.isnan(nmhc_scores)
        if valid.sum() > 100:
            y_true_v  = np.array(all_labels)[valid]
            y_score_v = nmhc_scores[valid]
            auc_val, Sens, Spec, Acc, MCC, F1, fpr, tpr, thresholds, ix, y_score_v = \
                compute_metrics(y_true_v, y_score_v, higher_is_better=False)
            rows.append({"Method": "NetMHCpan 4.2",
                         "AUC": auc_val, "Sens": Sens, "Spec": Spec,
                         "Acc": Acc, "MCC": MCC, "F1": F1})
            logging.info("  NetMHCpan 4.2  AUC=%.4f  Sens=%.4f  Spec=%.4f  Acc=%.4f  MCC=%.4f",
                         auc_val, Sens, Spec, Acc, MCC)
            save_roc_pkl(args.output, "NetMHCpan_42", args.plm_name,
                         y_true_v, y_score_v, fpr, tpr, thresholds, auc_val, ix)
        else:
            logging.warning("Not enough valid NetMHCpan scores (%d) — skipping", valid.sum())
    else:
        logging.info("Skipping NetMHCpan (--skip_netmhcpan)")

    # ── 4. MHCflurry 2.0 ──────────────────────────────────────────────────────
    if not args.skip_mhcflurry and all_peptides:
        logging.info("\n--- MHCflurry 2.0 ---")
        mhcf_scores = np.full(len(all_peptides), np.nan)
        for allele_tag in sorted(set(allele_per_pep)):
            mhcf_name = ALLELE_MAP[allele_tag][1]
            idxs = [i for i, a in enumerate(allele_per_pep) if a == allele_tag]
            peps = [all_peptides[i] for i in idxs]
            logging.info("  Running MHCflurry for %s (%d peptides)…", allele_tag, len(peps))
            scores = run_mhcflurry(peps, mhcf_name)
            if scores is not None and len(scores) == len(idxs):
                for i, s in zip(idxs, scores):
                    mhcf_scores[i] = s
            else:
                logging.warning("  Incomplete scores for %s — skipping allele", allele_tag)
        valid = ~np.isnan(mhcf_scores)
        if valid.sum() > 100:
            y_true_v  = np.array(all_labels)[valid]
            y_score_v = mhcf_scores[valid]
            auc_val, Sens, Spec, Acc, MCC, F1, fpr, tpr, thresholds, ix, y_score_v = \
                compute_metrics(y_true_v, y_score_v, higher_is_better=True)
            rows.append({"Method": "MHCflurry 2.0",
                         "AUC": auc_val, "Sens": Sens, "Spec": Spec,
                         "Acc": Acc, "MCC": MCC, "F1": F1})
            logging.info("  MHCflurry 2.0  AUC=%.4f  Sens=%.4f  Spec=%.4f  Acc=%.4f  MCC=%.4f",
                         auc_val, Sens, Spec, Acc, MCC)
            save_roc_pkl(args.output, "MHCflurry_20", args.plm_name,
                         y_true_v, y_score_v, fpr, tpr, thresholds, auc_val, ix)
        else:
            logging.warning("Not enough valid MHCflurry scores (%d) — skipping", valid.sum())
    else:
        logging.info("Skipping MHCflurry (--skip_mhcflurry)")

    # ── 5. Save CSV ────────────────────────────────────────────────────────────
    if not rows:
        logging.error("No results to save.")
        return
    csv_path = os.path.join(args.output, f"benchmark_results_{args.plm_name}.csv")
    fields = ["Method", "AUC", "Sens", "Spec", "Acc", "MCC", "F1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                             for k, v in r.items()})

    print("\n" + "="*80)
    print(f"  Benchmark Comparison  |  PLM: {args.plm_name.upper()}")
    print("="*80)
    print(f"  {'Method':<22}  {'AUC':>7}  {'Sens':>7}  {'Spec':>7}  "
          f"{'Acc':>7}  {'MCC':>7}  {'F1':>7}")
    print("-"*80)
    for r in rows:
        print(f"  {r['Method']:<22}  {r['AUC']:>7.4f}  {r['Sens']:>7.4f}  "
              f"{r['Spec']:>7.4f}  {r['Acc']:>7.4f}  {r['MCC']:>7.4f}  {r['F1']:>7.4f}")
    print("="*80)
    print(f"\nResults saved → {csv_path}")


if __name__ == "__main__":
    main()
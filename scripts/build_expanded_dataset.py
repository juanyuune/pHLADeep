"""
build_expanded_dataset.py
=========================
Builds an expanded multi-allele dataset from:

  SOURCE A — IEDB file (your original 4 alleles, cleaned):
      IEDB_peptide_data.xlsx  →  A02-01, A03-01, B08-01, B57-01

  SOURCE B — S_variant_screen (new alleles from SARS-CoV-2 screen):
      S_variant_screen_data.xlsx
      Binder columns: A1, A24:02, B07:02, B40, B51
      Non-binder pool: all 29,689 screened peptides minus binders

  SOURCE C — Random_9mer_screen (independent validation set):
      Random_9-mer_screen_peptides.xlsx
      For your original 4 alleles — used as VALIDATION ONLY, not training

  SOURCE D — IEDB MHC Ligand Assay export (new alleles, 112-col format):
      One xlsx file per allele, downloaded from iedb.org/mhc_ligand_home.php
      col 11  = "Epitope - Name"  (peptide sequence)
      col 94  = "Assay - Qualitative Measurement"  (Positive* / Negative)
      Allele tag inferred from "MHC Restriction Type" in Query Parameters sheet
      --iedb_new  accepts one or more file paths (space-separated)

      Files confirmed (October 2026):
        HLA-A*01:01  f51a8a5b  rows=29007  9mer_pos=23883  9mer_neg=3547
        HLA-A*24:02  82c1013d  rows=30491  9mer_pos=26465  9mer_neg=2243
        HLA-B*07:02  a3629f66  rows=43367  9mer_pos=35938  9mer_neg=3442
        HLA-A*11:01  33124224  rows=34608  9mer_pos=31094  9mer_neg=3128
        HLA-B*35:01  0a7c586c  rows=21438  9mer_pos=17769  9mer_neg=753

      For alleles also present in Source B (A0101, A2402, B0702), the two
      sources are MERGED: IEDB is authoritative for labels; B only adds
      positives that IEDB lacks (never adds negatives, avoiding pool bias).

Output directory structure (mirrors your existing layout):
  <out_dir>/
    HLA-A0101/  pos_train/ neg_train/ pos_test/ neg_test/
    HLA-A0201/  ...
    HLA-A0301/  ...
    HLA-A1101/  ...   <- new (Source D only)
    HLA-A2402/  ...   <- merged (Source B + D)
    HLA-B0702/  ...   <- merged (Source B + D)
    HLA-B0801/  ...
    HLA-B3501/  ...   <- new (Source D only)
    HLA-B4001/  ...
    HLA-B5101/  ...
    HLA-B5701/  ...
    validation/          <- Random 9-mer screen (independent, held-out)
      HLA-A0201/ pos/ neg/
      HLA-A0301/ pos/ neg/
      HLA-B0801/ pos/ neg/
      HLA-B5701/ pos/ neg/

Each peptide is saved as <ALLELE>_<PEPTIDE>.fasta  (to be embedded by ESM-2)

Usage:
  python build_expanded_dataset.py \
      --iedb      /path/to/IEDB_peptide_data.xlsx \
      --svariant  /path/to/S_variant_screen_data.xlsx \
      --random9   /path/to/Random_9-mer_screen_peptides.xlsx \
      --iedb_new  /path/to/mhc_ligand_A0101.xlsx \
                  /path/to/mhc_ligand_A2402.xlsx \
                  /path/to/mhc_ligand_B0702.xlsx \
                  /path/to/mhc_ligand_A1101.xlsx \
                  /path/to/mhc_ligand_B3501.xlsx \
      --out       /srv/jupyterlab/workspace/HLA1/data/expanded

After this script, run your ESM-2 embedding pipeline on the new allele folders.
"""

import os
import re
import random
import argparse
import logging

from openpyxl import load_workbook
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--iedb",     type=str, required=True,
                    help="Path to IEDB_peptide_data.xlsx (original 4 alleles)")
parser.add_argument("--svariant", type=str, required=True,
                    help="Path to S_variant_screen_data.xlsx")
parser.add_argument("--random9",  type=str, required=True,
                    help="Path to Random_9-mer_screen_peptides.xlsx")
parser.add_argument("--iedb_new", type=str, nargs="*", default=[],
                    help="One or more IEDB MHC Ligand Assay export xlsx files "
                         "(112-col format, one allele per file)")
parser.add_argument("--out",      type=str, required=True,
                    help="Output root directory")
parser.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction held out as test set (default 0.2)")
parser.add_argument("--min_pos",   type=int, default=20,
                    help="Minimum positives required to include an allele")
parser.add_argument("--max_neg_ratio", type=int, default=10,
                    help="Cap negatives at max_neg_ratio × positives (0=no cap, default 10). "
                         "Useful to reduce severe class imbalance in S_variant alleles.")
args = parser.parse_args()

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MAXSEQ = 9


def is_valid_9mer(seq):
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    return len(seq) == MAXSEQ and all(c in AMINO_ACIDS for c in seq)


def write_fasta(folder, allele_tag, peptide):
    """Write peptide as FASTA: <allele>_<peptide>.fasta"""
    fname = f"{allele_tag}_{peptide}.fasta"
    fpath = os.path.join(folder, fname)
    if not os.path.exists(fpath):
        with open(fpath, "w") as f:
            f.write(f">{allele_tag}_{peptide}\n{peptide}\n")


def save_split(allele_tag, positives, negatives, out_root):
    """80/20 train/test split → pos_train, neg_train, pos_test, neg_test.
    If --max_neg_ratio > 0 and negatives >> positives, subsample negatives."""
    allele_dir = os.path.join(out_root, allele_tag)
    for sub in ["pos_train", "neg_train", "pos_test", "neg_test"]:
        os.makedirs(os.path.join(allele_dir, sub), exist_ok=True)

    # Optional negative subsampling to reduce class imbalance
    neg_list = list(negatives)
    if args.max_neg_ratio > 0 and len(neg_list) > args.max_neg_ratio * len(positives):
        random.seed(42)
        cap = args.max_neg_ratio * len(positives)
        neg_list = random.sample(neg_list, cap)
        logging.info("  %-12s  neg subsampled %d → %d (ratio 1:%d)",
                     allele_tag, len(negatives), cap, args.max_neg_ratio)

    if len(neg_list) < 2:
        logging.warning("  %-12s  WARNING: only %d negatives after dedup — "
                        "allele will be skipped (needs more IEDB data).",
                        allele_tag, len(neg_list))
        return 0, 0, 0, 0

    pos_tr, pos_te = train_test_split(
        list(positives), test_size=args.test_size, random_state=42)
    neg_tr, neg_te = train_test_split(
        neg_list, test_size=args.test_size, random_state=42)

    for pep in pos_tr:
        write_fasta(os.path.join(allele_dir, "pos_train"), allele_tag, pep)
    for pep in neg_tr:
        write_fasta(os.path.join(allele_dir, "neg_train"), allele_tag, pep)
    for pep in pos_te:
        write_fasta(os.path.join(allele_dir, "pos_test"),  allele_tag, pep)
    for pep in neg_te:
        write_fasta(os.path.join(allele_dir, "neg_test"),  allele_tag, pep)

    logging.info("  %-12s  pos=%4d  neg=%4d  (tr=%d/%d  te=%d/%d)",
                 allele_tag,
                 len(positives), len(neg_list),
                 len(pos_tr), len(neg_tr),
                 len(pos_te), len(neg_te))
    return len(pos_tr), len(neg_tr), len(pos_te), len(neg_te)


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE A — IEDB (original 4 alleles)
# ═══════════════════════════════════════════════════════════════════════════════
def load_iedb(fpath):
    """
    Returns dict: allele_tag → (set of positives, set of negatives)
    Label logic (same as your original pipeline):
      Positive-High / Positive-Intermediate / Positive / Positive-Low → 1
      Negative → 0
      NaN / unknown → drop
    Deduplication: if a peptide appears as both pos and neg, keep positive.
    """
    SHEET_TO_TAG = {
        "A02-01": "HLA-A0201",
        "A03-01": "HLA-A0301",
        "B08-01": "HLA-B0801",
        "B57-01": "HLA-B5701",
    }
    result = {}
    wb = load_workbook(fpath, read_only=True)

    for sheet, tag in SHEET_TO_TAG.items():
        ws = wb[sheet]
        rows = list(ws.iter_rows(min_row=1, values_only=True))
        header = rows[0]
        # B57-01 has extra 'Object Type' column — use name lookup, not index
        desc_col = header.index("Description")
        qual_col = header.index("Qualitative Measure")
        n_cols   = len(header)

        pos, neg = set(), set()
        for row in rows[1:]:
            if len(row) < n_cols:
                continue   # skip malformed rows
            pep = row[desc_col]
            qual = row[qual_col]
            if not is_valid_9mer(pep):
                continue
            pep = pep.strip().upper()
            if isinstance(qual, str) and qual.startswith("Positive"):
                pos.add(pep)
            elif qual == "Negative":
                neg.add(pep)

        # Dedup: peptides in both → keep positive
        neg -= pos
        result[tag] = (pos, neg)
        logging.info("IEDB %-12s: pos=%d  neg=%d", tag, len(pos), len(neg))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE B — S_variant screen (new alleles)
# ═══════════════════════════════════════════════════════════════════════════════
def load_svariant(fpath):
    """
    Column layout (row 2 = header, row 3+ = data):
      col 0 : all peptides (screened pool — used as negatives)
      col 1 : A1 binders
      col 2 : A2 SPP WT binders
      col 3 : A2 SPP KO binders
      col 4 : A3 binders
      col 5 : A24:02 binders
      col 6 : B07:02 binders
      col 7 : B8 binders
      col 8 : B40 binders
      col 9 : B51 binders
      col 10: C4 binders        ← skip (C4 is ambiguous/rare)
      col 11: C07:01 binders    ← skip (only 7 binders)
      col 12: C07:02 binders    ← skip (only 6 binders)

    We use ONLY 9-mers. Non-binder pool = all_peptides minus binders for that allele.
    We SKIP A2, A3, B8 here — those come from IEDB (more data).
    """
    COL_TO_TAG = {
        1:  "HLA-A0101",   # A1
        5:  "HLA-A2402",   # A24:02
        6:  "HLA-B0702",   # B07:02
        8:  "HLA-B4001",   # B40
        9:  "HLA-B5101",   # B51
    }
    wb = load_workbook(fpath, read_only=True)
    ws = wb['Sheet1']
    all_rows = list(ws.iter_rows(min_row=3, values_only=True))

    # All 9-mer peptides in the screened pool
    pool_9mer = set()
    for row in all_rows:
        pep = row[0]
        if is_valid_9mer(pep):
            pool_9mer.add(pep.strip().upper())
    logging.info("S_variant pool: %d unique 9-mers", len(pool_9mer))

    result = {}
    for col, tag in COL_TO_TAG.items():
        binders = set()
        for row in all_rows:
            pep = row[col]
            if is_valid_9mer(pep):
                binders.add(pep.strip().upper())
        # Non-binders = pool minus binders
        non_binders = pool_9mer - binders
        result[tag] = (binders, non_binders)
        logging.info("S_variant %-12s: pos=%d  neg=%d",
                     tag, len(binders), len(non_binders))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE C — Random 9-mer screen (validation only)
# ═══════════════════════════════════════════════════════════════════════════════
def load_random9mer(fpath):
    """
    Column layout (row 2 = header, row 3+ = data):
      col 0 : all random 9mers (screened pool)
      col 1 : A2 binders
      col 2 : A2 SPP KO binders
      col 3 : A3 binders
      col 4 : B8 binders
      col 5 : B57 binders
      col 6 : B57 abacavir binders

    Returns dict: allele_tag → (pos_set, neg_set)
    This is saved to validation/ folder — never mixed with training data.
    """
    COL_TO_TAG = {
        1: "HLA-A0201",
        3: "HLA-A0301",
        4: "HLA-B0801",
        5: "HLA-B5701",
    }
    wb = load_workbook(fpath, read_only=True)
    ws = wb['Table S1']
    all_rows = list(ws.iter_rows(min_row=3, values_only=True))

    pool = set()
    for row in all_rows:
        pep = row[0]
        if is_valid_9mer(pep):
            pool.add(pep.strip().upper())
    logging.info("Random9mer pool: %d unique 9-mers", len(pool))

    result = {}
    for col, tag in COL_TO_TAG.items():
        binders = set()
        for row in all_rows:
            pep = row[col]
            if is_valid_9mer(pep):
                binders.add(pep.strip().upper())
        non_binders = pool - binders
        result[tag] = (binders, non_binders)
        logging.info("Random9mer %-12s: pos=%d  neg=%d",
                     tag, len(binders), len(non_binders))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE D — IEDB MHC Ligand Assay export (new 112-col format)
# ═══════════════════════════════════════════════════════════════════════════════
# Allele → HLA tag mapping (add new alleles here as needed)
IEDB_MHC_TAG = {
    "HLA-A*01:01": "HLA-A0101",
    "HLA-A*02:01": "HLA-A0201",
    "HLA-A*03:01": "HLA-A0301",
    "HLA-A*11:01": "HLA-A1101",
    "HLA-A*24:02": "HLA-A2402",
    "HLA-B*07:02": "HLA-B0702",
    "HLA-B*08:01": "HLA-B0801",
    "HLA-B*35:01": "HLA-B3501",
    "HLA-B*40:01": "HLA-B4001",
    "HLA-B*51:01": "HLA-B5101",
    "HLA-B*57:01": "HLA-B5701",
}

def load_iedb_mhc_ligand(fpath):
    """
    Parse one IEDB MHC Ligand Assay export xlsx (112-col format).
    Returns: (allele_tag str, pos set, neg set)

    Column indices (0-based, confirmed October 2026):
      11  = Epitope - Name        (peptide sequence)
      94  = Assay - Qualitative Measurement  (Positive* / Negative)

    Allele is read from the 'Query Parameters' sheet:
      rows containing "MHC Restriction Type: HLA-X*xx:xx ..."
    """
    wb = load_workbook(fpath, read_only=True, data_only=True)

    # --- infer allele from Query Parameters ---
    allele_tag = None
    qp = wb["Query Parameters"]
    for row in qp.iter_rows(values_only=True):
        v = row[0] if row and len(row) > 0 else None
        if not v:
            continue
        m = re.search(r"HLA-[A-Z]\*\d+:\d+", str(v))
        if m:
            raw = m.group(0)
            allele_tag = IEDB_MHC_TAG.get(raw)
            if allele_tag is None:
                # Auto-generate tag: HLA-A*01:01 → HLA-A0101
                allele_tag = "HLA-" + raw[4:].replace("*", "").replace(":", "")
            break

    if allele_tag is None:
        logging.warning("Could not detect allele from %s — skipping", fpath)
        wb.close()
        return None, set(), set()

    # --- read peptides & labels ---
    ws = wb["Sheet1"]
    pos, neg = set(), set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        pep  = row[11]
        qual = row[94]
        if not is_valid_9mer(pep):
            continue
        pep = pep.strip().upper()
        if isinstance(qual, str) and qual.startswith("Positive"):
            pos.add(pep)
        elif qual == "Negative":
            neg.add(pep)

    # Dedup: if in both, keep positive
    neg -= pos
    wb.close()
    logging.info("IEDB-MHC %-12s: pos=%d  neg=%d  [%s]",
                 allele_tag, len(pos), len(neg), os.path.basename(fpath))
    return allele_tag, pos, neg


def save_validation(allele_tag, positives, negatives, out_root):
    """Save validation peptides to validation/<allele>/pos/ and neg/."""
    val_dir = os.path.join(out_root, "validation", allele_tag)
    os.makedirs(os.path.join(val_dir, "pos"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "neg"), exist_ok=True)
    for pep in positives:
        write_fasta(os.path.join(val_dir, "pos"), allele_tag, pep)
    for pep in negatives:
        write_fasta(os.path.join(val_dir, "neg"), allele_tag, pep)
    logging.info("  Validation %-12s: pos=%d  neg=%d",
                 allele_tag, len(positives), len(negatives))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(args.out, exist_ok=True)
    summary_rows = []

    # ── Source A: IEDB original 4 alleles ──
    logging.info("=== Loading IEDB original (Source A) ===")
    iedb = load_iedb(args.iedb)

    # ── Source B: S_variant (new alleles) ──
    logging.info("=== Loading S_variant screen (Source B) ===")
    svar = load_svariant(args.svariant)

    # ── Source D: New IEDB MHC Ligand Assay files ──
    iedb_new = {}  # tag → (pos set, neg set)
    if args.iedb_new:
        logging.info("=== Loading new IEDB MHC Ligand files (Source D) ===")
        for fpath in args.iedb_new:
            tag, pos, neg = load_iedb_mhc_ligand(fpath)
            if tag is None:
                continue
            if tag in iedb_new:
                # Merge with existing (shouldn't happen if one file per allele)
                p0, n0 = iedb_new[tag]
                p0.update(pos); n0.update(neg); n0 -= p0
                iedb_new[tag] = (p0, n0)
            else:
                iedb_new[tag] = (pos, neg)

    # ── Build per-allele combined dataset ──
    # Precedence: Source A > Source D > Source B
    # For labels, IEDB sources (A+D) are authoritative.
    # S_variant positives supplement where IEDB lacks data.
    # S_variant negatives are NOT used when IEDB has the same allele
    # (pool-screen negatives are lower confidence than IEDB negatives).

    all_tags = set(iedb.keys()) | set(svar.keys()) | set(iedb_new.keys())

    for tag in sorted(all_tags):
        pos_final, neg_final = set(), set()
        sources_used = []

        if tag in iedb:
            p, n = iedb[tag]
            pos_final.update(p); neg_final.update(n)
            sources_used.append("A")

        if tag in iedb_new:
            p, n = iedb_new[tag]
            pos_final.update(p); neg_final.update(n)
            sources_used.append("D")

        if tag in svar:
            p_sv, n_sv = svar[tag]
            if tag in iedb or tag in iedb_new:
                # IEDB data present: only add S_variant positives not in IEDB
                extra_pos = p_sv - pos_final
                if extra_pos:
                    pos_final.update(extra_pos)
                    logging.info("  %s: +%d S_variant positives not in IEDB",
                                 tag, len(extra_pos))
                sources_used.append("B(pos-only)")
            else:
                # No IEDB data: use both positives and negatives from S_variant
                pos_final.update(p_sv); neg_final.update(n_sv)
                sources_used.append("B")

        # Final dedup: anything positive stays positive
        neg_final -= pos_final
        src_str = "+".join(sources_used) if sources_used else "?"

        if len(pos_final) < args.min_pos:
            logging.warning("Skipping %s — only %d positives (sources: %s)",
                            tag, len(pos_final), src_str)
            continue

        counts = save_split(tag, pos_final, neg_final, args.out)
        if counts == (0, 0, 0, 0):
            continue   # skipped due to insufficient negatives
        summary_rows.append((tag, src_str, len(pos_final), len(neg_final), *counts))

    # ── Source C: Random 9-mer → validation only ──
    logging.info("=== Loading Random 9-mer screen (Source C — validation) ===")
    rand9 = load_random9mer(args.random9)
    for tag, (pos, neg) in rand9.items():
        if len(pos) < args.min_pos:
            logging.warning("Skipping validation %s — only %d pos", tag, len(pos))
            continue
        save_validation(tag, pos, neg, args.out)

    # ── Summary ──
    logging.info("")
    logging.info("=" * 75)
    logging.info("DATASET SUMMARY")
    logging.info("=" * 75)
    logging.info("%-14s %-18s %6s %6s %6s %6s %6s %6s",
                 "Allele", "Source(s)", "pos", "neg",
                 "pos_tr", "neg_tr", "pos_te", "neg_te")
    total_pos, total_neg = 0, 0
    for row in summary_rows:
        tag, src, pos, neg, ptr, ntr, pte, nte = row
        logging.info("%-14s %-18s %6d %6d %6d %6d %6d %6d",
                     tag, src, pos, neg, ptr, ntr, pte, nte)
        total_pos += pos
        total_neg += neg
    logging.info("-" * 75)
    logging.info("Total: %d positives, %d negatives across %d alleles",
                 total_pos, total_neg, len(summary_rows))
    logging.info("")
    logging.info("Output: %s", args.out)
    logging.info("Next step: run ESM-2 embedding on each allele folder")


if __name__ == "__main__":
    main()
"""
extract_iedb_sequences.py

Extracts 9-mer peptide sequences from the IEDB supplementary file
(41587_2022_1566_MOESM4_ESM.xlsx) from Bruno et al., Nature Biotechnology 2023.

Output: one CSV per allele + one combined CSV, with columns:
    sequence, allele, length, qualitative_label, binary_label,
    quantitative_nM, assay_group, method, pubmed_id

Usage:
    python extract_iedb_sequences.py
    python extract_iedb_sequences.py --input path/to/file.xlsx --outdir ./output
"""

import argparse
import os
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

# Map sheet name → HLA allele label
SHEET_TO_ALLELE = {
    "A02-01": "HLA-A*02:01",
    "A03-01": "HLA-A*03:01",
    "B08-01": "HLA-B*08:01",
    "B57-01": "HLA-B*57:01",
}

# Qualitative labels that count as binding positives
POSITIVE_LABELS = {"Positive-High", "Positive-Intermediate", "Positive", "Positive-Low"}
NEGATIVE_LABELS = {"Negative"}
# Positive-Low is included but flagged as weak — see binary_label below

# Strong positives only (for strict training)
STRONG_POSITIVE_LABELS = {"Positive-High", "Positive-Intermediate", "Positive"}

# Standard amino acid alphabet — filter out anything with non-standard characters
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Label strength ranking — used for deduplication (keep strongest label per sequence-allele)
LABEL_ORDER = {
    "Positive-High"        : 4,
    "Positive-Intermediate": 3,
    "Positive"             : 2,
    "Positive-Low"         : 1,
    "Negative"             : 0,
}


# ── Helper functions ───────────────────────────────────────────────────────────

def is_valid_sequence(seq: str) -> bool:
    """Return True if the sequence is a non-empty string of standard amino acids."""
    if not isinstance(seq, str) or len(seq) == 0:
        return False
    return all(aa in STANDARD_AA for aa in seq.upper())


def assign_binary_label(qualitative: str) -> int | None:
    """
    Convert qualitative label to binary.
    Returns 1 (positive), 0 (negative), or None (ambiguous — exclude).

    Design choice documented here:
    - Positive-High / Positive-Intermediate / Positive → 1
    - Negative → 0
    - Positive-Low → 1 by default, but flagged as weak (see weak_binder column)
    - Anything else (NaN, unknown) → None (dropped)
    """
    if qualitative in STRONG_POSITIVE_LABELS:
        return 1
    if qualitative == "Positive-Low":
        return 1  # included but weak — use weak_binder column to filter if needed
    if qualitative in NEGATIVE_LABELS:
        return 0
    return None  # ambiguous — will be dropped


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by sequence + allele pair.
    When a sequence appears multiple times (measured in different papers
    or assays), keep only the row with the strongest qualitative label.

    Priority: Positive-High > Positive-Intermediate > Positive >
              Positive-Low > Negative

    This ensures:
    - One embedding per unique peptide-allele combination
    - The most informative label is kept, not an arbitrary one
    - Dataset size is explicit and reproducible
    """
    before = len(df)

    df["label_rank"] = df["qualitative_label"].map(LABEL_ORDER).fillna(0)
    df = df.sort_values("label_rank", ascending=False)
    df = df.drop_duplicates(subset=["sequence", "allele"], keep="first")
    df = df.drop(columns=["label_rank"])
    df = df.reset_index(drop=True)

    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"    Deduplicated     : {before:,} → {after:,} rows "
              f"({removed:,} duplicates removed, strongest label kept)")
    return df


def load_sheet(filepath: str, sheet_name: str, allele: str) -> pd.DataFrame:
    """Load one sheet, clean it, and return a standardised DataFrame."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # B08-01 and B57-01 use 'len' instead of 'length'
    if "len" in df.columns and "length" not in df.columns:
        df = df.rename(columns={"len": "length"})

    # Keep 9-mers only
    df = df[df["length"] == 9].copy()

    # Rename sequence column
    df = df.rename(columns={"Description": "sequence"})

    # Uppercase and strip whitespace
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    # Drop invalid sequences
    df = df[df["sequence"].apply(is_valid_sequence)].copy()

    # Add allele label
    df["allele"] = allele

    # Binary label
    df["binary_label"] = df["Qualitative Measure"].apply(assign_binary_label)

    # Drop ambiguous labels
    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)

    # Flag weak binders separately
    df["weak_binder"] = (df["Qualitative Measure"] == "Positive-Low").astype(int)

    # Rename columns for clarity
    df = df.rename(columns={
        "Qualitative Measure":       "qualitative_label",
        "Quantitative measurement":  "quantitative_nM",
        "Assay Group":               "assay_group",
        "Method/Technique":          "method",
        "PubMed ID":                 "pubmed_id",
    })

    # Select and order final columns
    keep = [
        "sequence",
        "allele",
        "length",
        "qualitative_label",
        "binary_label",
        "weak_binder",
        "quantitative_nM",
        "assay_group",
        "method",
        "pubmed_id",
    ]
    # Only keep columns that exist
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # ── Deduplicate by sequence + allele (keep strongest label) ──
    df = deduplicate(df)

    df = df.reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame, allele: str) -> None:
    total     = len(df)
    pos       = (df["binary_label"] == 1).sum()
    neg       = (df["binary_label"] == 0).sum()
    weak      = df["weak_binder"].sum()
    has_quant = df["quantitative_nM"].notna().sum()

    print(f"\n  {allele}")
    print(f"    Total rows      : {total:>6,}")
    print(f"    Positives       : {pos:>6,}  (of which weak binders: {weak:,})")
    print(f"    Negatives       : {neg:>6,}")
    print(f"    With IC50/KD    : {has_quant:>6,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract IEDB 9-mer sequences.")
    parser.add_argument(
        "--input", default="IEDB_peptide_data_41587_2022_1566_MOESM4_ESM.xlsx",
        help="Path to the IEDB Excel file"
    )
    parser.add_argument(
        "--outdir", default="iedb_sequences",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exclude Positive-Low from positives (strict mode)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nReading: {args.input}")
    print(f"Output:  {args.outdir}/")
    print(f"Mode:    {'strict (Positive-Low excluded)' if args.strict else 'standard (Positive-Low included as weak)'}")

    all_dfs = []

    for sheet, allele in SHEET_TO_ALLELE.items():
        df = load_sheet(args.input, sheet, allele)

        # Strict mode: drop weak binders from positives
        if args.strict:
            df = df[~((df["binary_label"] == 1) & (df["weak_binder"] == 1))].copy()

        print_summary(df, allele)

        # Save per-allele CSV
        allele_slug = allele.replace("*", "").replace(":", "")
        out_path = os.path.join(args.outdir, f"{allele_slug}_9mer.csv")
        df.to_csv(out_path, index=False)
        print(f"    Saved → {out_path}")

        all_dfs.append(df)

    # Combined CSV
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(args.outdir, "all_alleles_9mer.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\nCombined: {len(combined):,} rows → {combined_path}")
    print(f"  Positives : {(combined['binary_label'] == 1).sum():,}")
    print(f"  Negatives : {(combined['binary_label'] == 0).sum():,}")
    print(f"  Weak binders included : {combined['weak_binder'].sum():,}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
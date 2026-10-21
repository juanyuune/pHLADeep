"""
csv_to_fasta.py

Converts IEDB 9-mer CSV into individual FASTA files.
Use --label to filter by binary_label (0 or 1).

Each FASTA file is named:  <allele>_<sequence>.fasta
Each FASTA header is:      ><allele>|<sequence>|<qualitative_label>

Usage:
    python csv_to_fasta.py --input train.csv --outdir pos_train --label 1
    python csv_to_fasta.py --input train.csv --outdir neg_train --label 0
    python csv_to_fasta.py --input test.csv  --outdir pos_test  --label 1
    python csv_to_fasta.py --input test.csv  --outdir neg_test  --label 0
"""

import os
import argparse
import pandas as pd


def sanitise(text: str) -> str:
    return str(text).replace("*", "").replace(":", "").replace("/", "_").strip()


def write_fasta(filepath: str, header: str, sequence: str) -> None:
    with open(filepath, "w") as f:
        f.write(f">{header}\n{sequence}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert IEDB 9-mer CSV to individual FASTA files"
    )
    parser.add_argument("--input",  required=True, help="Path to CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--label",  type=int, choices=[0, 1], default=None,
                        help="Filter by binary_label: 1=pos, 0=neg. Omit for all.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)

    required = ["sequence", "binary_label", "allele"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df = df[df["sequence"].str.len() == 9].copy()

    # Filter by label if specified
    if args.label is not None:
        df = df[df["binary_label"] == args.label].copy()

    n_written = 0
    n_skip    = 0

    for _, row in df.iterrows():
        seq        = row["sequence"]
        label      = int(row["binary_label"])
        allele     = row["allele"]
        qual_label = row.get("qualitative_label", "")

        if len(seq) != 9:
            n_skip += 1
            continue

        allele_safe = sanitise(allele)
        filename    = f"{allele_safe}_{seq}.fasta"
        header      = f"{allele}|{seq}|{qual_label}"
        filepath    = os.path.join(args.outdir, filename)

        write_fasta(filepath, header, seq)
        n_written += 1

    label_str = {1: "positive", 0: "negative", None: "all"}[args.label]
    print(f"\nDone.")
    print(f"  Label    : {label_str}")
    print(f"  Written  : {n_written:>7,} FASTA files  →  {args.outdir}")
    if n_skip > 0:
        print(f"  Skipped  : {n_skip:>7,}")


if __name__ == "__main__":
    main()
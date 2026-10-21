"""
embed_hla_pseudo.py
===================
Reads HLA pseudosequences from a CSV and generates ESM-2 embeddings.
Output: one .esm2 file per allele, shape (34, 1280), saved via pickle.

Usage:
  python embed_hla_pseudo.py \
      --csv  /srv/jupyterlab/workspace/HLA1/hla/hla_pseudosequences.csv \
      --out  /srv/jupyterlab/workspace/HLA1/hla/hla_embeddings

If --alleles is omitted, all alleles in the CSV are embedded.
Output filename format: HLA-A0101.esm2  (star and colon removed)
"""

import os
import gc
import csv
import pickle
import logging
import argparse

import torch
import esm
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Device: %s", device)

EXPECTED_LEN = 34   # MHCflurry pseudosequence length


def allele_to_tag(allele):
    """HLA-A*01:01 → HLA-A0101"""
    return allele.replace("*", "").replace(":", "")


def load_pseudosequences(csv_path, target_alleles=None):
    """Returns dict: allele_name → pseudosequence string"""
    seqs = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            allele = row["allele"].strip()
            seq    = row["pseudosequence"].strip().upper()
            if target_alleles and allele not in target_alleles:
                continue
            if len(seq) != EXPECTED_LEN:
                logging.warning("Unexpected length %d for %s — skipping",
                                len(seq), allele)
                continue
            seqs[allele] = seq
    return seqs


def load_esm2():
    logging.info("Loading ESM-2 650M model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    tokenizer = alphabet.get_batch_converter()
    logging.info("Model loaded.")
    return model, tokenizer


def embed_sequence(model, tokenizer, seq_id, seq):
    """Returns numpy array of shape (len(seq), 1280) — per-residue embeddings."""
    batch_data = [(seq_id, seq)]
    _, _, tokens = tokenizer(batch_data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)
        emb = out["representations"][33][0, 1:len(seq)+1].cpu().numpy()
    del tokens, out
    torch.cuda.empty_cache()
    gc.collect()
    return emb   # shape: (34, 1280)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True,
                        help="Path to hla_pseudosequences.csv")
    parser.add_argument("--out",     required=True,
                        help="Output directory for .esm2 files")
    parser.add_argument("--alleles", nargs="*", default=None,
                        help="Alleles to embed (e.g. HLA-A*01:01). "
                             "Default: all alleles in CSV.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    target_set = set(args.alleles) if args.alleles else None
    seqs = load_pseudosequences(args.csv, target_set)

    if not seqs:
        logging.error("No matching alleles found in CSV. Check allele names.")
        return

    if target_set:
        missing = target_set - set(seqs.keys())
        if missing:
            logging.warning("Not found in CSV: %s", missing)

    model, tokenizer = load_esm2()

    for allele, seq in sorted(seqs.items()):
        tag      = allele_to_tag(allele)
        out_path = os.path.join(args.out, f"{tag}.esm2")

        logging.info("Embedding %s  seq=%s", allele, seq)
        emb = embed_sequence(model, tokenizer, tag, seq)
        assert emb.shape == (EXPECTED_LEN, 1280), \
            f"Unexpected shape {emb.shape} for {allele}"

        with open(out_path, "wb") as f:
            pickle.dump(emb, f)
        logging.info("  → saved %s  shape=%s", out_path, emb.shape)

    logging.info("Done. %d alleles embedded.", len(seqs))


if __name__ == "__main__":
    main()
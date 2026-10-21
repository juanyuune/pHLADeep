"""
embed_hla_pseudo_ankh.py
========================
Reads HLA pseudosequences from a CSV and generates Ankh-Large embeddings.
Output: one .ankh file per allele, shape (34, 1536), saved via pickle.

Usage:
  python embed_hla_pseudo_ankh.py \
      --csv  /srv/jupyterlab/workspace/HLA1/hla/hla_pseudosequences.csv \
      --out  /srv/jupyterlab/workspace/HLA1/hla/hla_embeddings_ankh

If --alleles is omitted, all alleles in the CSV are embedded.
Output filename format: HLA-A0101.ankh  (star and colon removed)
"""

import os
import gc
import csv
import pickle
import logging
import argparse

import torch
import ankh
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


def load_ankh():
    logging.info("Loading Ankh-Large model...")
    model, tokenizer = ankh.load_large_model()
    model = model.to(device).eval()
    logging.info("Model loaded.")
    return model, tokenizer


def embed_sequence(model, tokenizer, seq):
    """Returns numpy array of shape (34, 1536) — per-residue embeddings."""
    seq_list = list(seq)
    ids = tokenizer.batch_encode_plus(
        [seq_list],
        add_special_tokens=True,
        padding=False,
        is_split_into_words=True,
        return_tensors='pt'
    )
    input_ids      = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # Remove CLS and EOS tokens — keep only residue positions
        emb = out.last_hidden_state[0, 1:len(seq)+1].cpu().numpy()

    del input_ids, attention_mask, out
    torch.cuda.empty_cache()
    gc.collect()

    return emb   # shape: (34, 1536)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True,
                        help="Path to hla_pseudosequences.csv")
    parser.add_argument("--out",     required=True,
                        help="Output directory for .ankh files")
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

    model, tokenizer = load_ankh()

    for allele, seq in sorted(seqs.items()):
        tag      = allele_to_tag(allele)
        out_path = os.path.join(args.out, f"{tag}.ankh")

        logging.info("Embedding %s  seq=%s", allele, seq)
        emb = embed_sequence(model, tokenizer, seq)
        assert emb.shape == (EXPECTED_LEN, 1536), \
            f"Unexpected shape {emb.shape} for {allele}"

        with open(out_path, "wb") as f:
            pickle.dump(emb, f)
        logging.info("  → saved %s  shape=%s", out_path, emb.shape)

    logging.info("Done. %d alleles embedded.", len(seqs))


if __name__ == "__main__":
    main()
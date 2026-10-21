import os
import gc
import csv
import pickle
import logging
import argparse

import torch
import ankh
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Device: %s", device)

EXPECTED_LEN = 34


def allele_to_tag(allele):
    return allele.replace("*", "").replace(":", "")


def load_pseudosequences(csv_path, target_alleles=None):
    seqs = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            allele = row["allele"].strip()
            seq    = row["pseudosequence"].strip().upper()
            if target_alleles and allele not in target_alleles:
                continue
            if len(seq) != EXPECTED_LEN:
                logging.warning("Unexpected length %d for %s — skipping", len(seq), allele)
                continue
            seqs[allele] = seq
    return seqs


def load_ankh():
    logging.info("Loading Ankh-Large model...")
    model, tokenizer = ankh.load_large_model()
    model = model.to(device).eval()
    return model, tokenizer


def embed_sequence(model, tokenizer, seq):
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
        emb = out.last_hidden_state[0, 1:len(seq)+1].cpu().numpy()

    del input_ids, attention_mask, out
    torch.cuda.empty_cache()
    gc.collect()

    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True)
    parser.add_argument("--out",     required=True)
    parser.add_argument("--alleles", nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    target_set = set(args.alleles) if args.alleles else None
    seqs = load_pseudosequences(args.csv, target_set)

    if not seqs:
        logging.error("No matching alleles found in CSV.")
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
        assert emb.shape == (EXPECTED_LEN, 1536), f"Unexpected shape {emb.shape} for {allele}"

        with open(out_path, "wb") as f:
            pickle.dump(emb, f)
        logging.info("  saved %s  shape=%s", out_path, emb.shape)

    logging.info("Done. %d alleles embedded.", len(seqs))


if __name__ == "__main__":
    main()

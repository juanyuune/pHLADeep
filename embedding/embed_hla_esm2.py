"""
embed_hla_esm2.py

Embeds HLA pseudosequences (34 residues) using ESM-2.
Saves one .esm2 pickle per allele with shape (34, 1280).

Usage:
    python embed_hla_esm2.py \
        --input  /srv/jupyterlab/workspace/HLA1/hla/hla_pseudosequences.csv \
        --outdir /srv/jupyterlab/workspace/HLA1/hla/hla_embeddings
"""

import os
import gc
import pickle
import logging
import argparse

import numpy as np
import pandas as pd
import torch
import esm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_esm2():
    logging.info("Loading ESM-2 650M model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    tokenizer = alphabet.get_batch_converter()
    logging.info(f"ESM-2 loaded on {device}")
    return model, tokenizer


def embed_sequence(model, tokenizer, seq_id, sequence, repr_layer=33):
    """Embed a single sequence. Returns (L, 1280) numpy array."""
    batch_data = [(seq_id, sequence)]
    _, _, batch_tokens = tokenizer(batch_data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        token_reps = out["representations"][repr_layer]

    # Slice off BOS/EOS — keep only residue positions
    emb = token_reps[0, 1:len(sequence)+1].detach().cpu().numpy()

    del token_reps, out, batch_tokens
    torch.cuda.empty_cache()
    gc.collect()

    return emb   # shape (34, 1280)


def sanitise(allele: str) -> str:
    return allele.replace("*", "").replace(":", "").replace("/", "_").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Embed HLA pseudosequences with ESM-2"
    )
    parser.add_argument("--input",  required=True,
                        help="CSV with columns: allele, pseudosequence")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for .esm2 pickle files")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if "allele" not in df.columns or "pseudosequence" not in df.columns:
        raise ValueError("CSV must have columns: allele, pseudosequence")

    model, tokenizer = load_esm2()

    for _, row in df.iterrows():
        allele = row["allele"]
        seq    = row["pseudosequence"].strip().upper()

        logging.info(f"Embedding {allele} (length {len(seq)})")
        emb = embed_sequence(model, tokenizer, allele, seq)

        # Save
        fname    = f"{sanitise(allele)}.esm2"
        out_path = os.path.join(args.outdir, fname)
        with open(out_path, "wb") as f:
            pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"  Saved {emb.shape} → {out_path}")

    logging.info(f"\nDone. {len(df)} HLA embeddings saved to {args.outdir}")


if __name__ == "__main__":
    main()
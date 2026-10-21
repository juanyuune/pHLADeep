import numpy as np
import torch
import argparse
import os
import gc
import logging
import pickle
from transformers import T5Tokenizer, T5EncoderModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-in",  "--path_input",  type=str, required=True, help="Path to input FASTA file directory")
parser.add_argument("-out", "--path_output", type=str, required=True, help="Path to output ProtT5 embeddings directory")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Load ProtT5-XL Model ─────────────────────────────────────────
def get_prot_t5_model():
    logging.info("Loading ProtT5-XL model...")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50")
    model = model.to(device).eval()
    logging.info(f"ProtT5-XL loaded on {device}.")
    return model, tokenizer


# ── Read FASTA sequence ──────────────────────────────────────────
def read_fasta(fasta_path):
    seq = ''
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if not line.startswith('>'):
                seq += line.strip()
    seq_id = os.path.splitext(os.path.basename(fasta_path))[0]
    return [(seq_id, seq)]


# ── Generate ProtT5 embeddings ───────────────────────────────────
def get_embeddings(model, tokenizer, seqs):
    """
    Returns per-residue embeddings from ProtT5-XL encoder hidden states.
    Output shape per sequence: (seq_len, 1024)
    Spaces are inserted between amino acids as required by ProtT5 tokenizer.
    EOS token is stripped so the embedding aligns 1-to-1 with the input
    amino acid sequence, matching the ESM-2 behaviour.
    """
    results = {}

    for seq_id, seq in seqs:
        # ProtT5 requires spaces between amino acids
        seq_spaced = " ".join(list(seq))

        ids = tokenizer(
            [seq_spaced],
            add_special_tokens=True,
            padding=False,
            return_tensors="pt"
        )

        input_ids      = ids["input_ids"].to(device)
        attention_mask = ids["attention_mask"].to(device)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Last hidden state: shape (1, seq_len+1, 1024)
        # Strip EOS token → (seq_len, 1024)
        hidden_states = output.last_hidden_state
        seq_len = len(seq)
        emb = hidden_states[0, :seq_len].detach().cpu().numpy()

        results[seq_id] = emb
        logging.info(f"  [{seq_id}] embedding shape: {emb.shape}")

        del input_ids, attention_mask, output, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ── Save embeddings as .pt5 file using pickle ────────────────────
def save_embeddings(data, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)


# ── Main processing function ─────────────────────────────────────
def process_fasta(fasta_file, output_file, model, tokenizer):
    try:
        filename   = os.path.splitext(os.path.basename(fasta_file))[0]
        seqs       = read_fasta(fasta_file)
        results    = get_embeddings(model, tokenizer, seqs)
        embeddings = results[filename]
        save_embeddings(embeddings, output_file)
        logging.info(f"Successfully processed {fasta_file} -> {output_file}")
    except Exception as e:
        logging.error(f"Error processing {fasta_file}: {e}")
        save_failed_entry(fasta_file, output_file)


# ── Save failed file names ────────────────────────────────────────
def save_failed_entry(path, opath):
    with open('NO_OK.txt', 'a') as f:
        f.write(f"{path} > {opath}\n")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.path_input):
        raise FileNotFoundError(f"Input directory '{args.path_input}' not found!")
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    model, tokenizer = get_prot_t5_model()

    input_files = [f for f in os.listdir(args.path_input) if f.endswith(".fasta")]
    count = 0

    for file in input_files:
        input_path  = os.path.join(args.path_input,  file)
        output_path = os.path.join(args.path_output,
                                   os.path.splitext(file)[0] + ".pt5")
        logging.info(f"Processing: {input_path}")
        process_fasta(input_path, output_path, model, tokenizer)
        count += 1
        gc.collect()

    logging.info(f"Processed {count} sequences successfully!")
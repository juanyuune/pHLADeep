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
parser.add_argument("-out", "--path_output", type=str, required=True, help="Path to output ProstT5 embeddings directory")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Load ProstT5 Model ───────────────────────────────────────────
def get_prostt5_model():
    logging.info("Loading ProstT5 model...")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/ProstT5", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
    model = model.to(device).eval()
    logging.info(f"ProstT5 loaded on {device}.")
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


# ── Generate ProstT5 embeddings ──────────────────────────────────
def get_embeddings(model, tokenizer, seqs):
    """
    Returns per-residue embeddings from ProstT5 encoder hidden states.
    Output shape per sequence: (seq_len, 1024)
    ProstT5 requires the prefix token <AA2fold> for amino acid sequence input
    (as opposed to 3Di structural token input which uses <fold2AA>).
    The prefix token and EOS token are stripped so the embedding aligns
    1-to-1 with the input amino acid sequence, matching the ESM-2 behaviour.
    """
    results = {}

    for seq_id, seq in seqs:
        # ProstT5 requires <AA2fold> prefix + spaces between amino acids
        seq_spaced = "<AA2fold> " + " ".join(list(seq))

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

        # Last hidden state: shape (1, seq_len+2, 1024)
        # Strip prefix token (position 0) and EOS token (last position)
        # → (seq_len, 1024)
        hidden_states = output.last_hidden_state
        seq_len = len(seq)
        emb = hidden_states[0, 1:seq_len + 1].detach().cpu().numpy()

        results[seq_id] = emb
        logging.info(f"  [{seq_id}] embedding shape: {emb.shape}")

        del input_ids, attention_mask, output, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ── Save embeddings as .pt5s file using pickle ───────────────────
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

    model, tokenizer = get_prostt5_model()

    input_files = [f for f in os.listdir(args.path_input) if f.endswith(".fasta")]
    count = 0

    for file in input_files:
        input_path  = os.path.join(args.path_input,  file)
        output_path = os.path.join(args.path_output,
                                   os.path.splitext(file)[0] + ".prostt5")
        logging.info(f"Processing: {input_path}")
        process_fasta(input_path, output_path, model, tokenizer)
        count += 1
        gc.collect()

    logging.info(f"Processed {count} sequences successfully!")
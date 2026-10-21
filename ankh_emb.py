import numpy as np
import torch
import argparse
import os
import gc
import ankh
import logging
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-in",  "--path_input",  type=str, required=True, help="Path to input FASTA file directory")
parser.add_argument("-out", "--path_output", type=str, required=True, help="Path to output Ankh embeddings directory")
parser.add_argument("-model", "--model_size", type=str, default="large",
                    choices=["base", "large"],
                    help="Ankh model size: 'base' (450M) or 'large' (1.5B). Default: large")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Load Ankh Model ──────────────────────────────────────────────
def get_ankh_model(model_size="large"):
    logging.info(f"Loading Ankh-{model_size} model...")
    if model_size == "large":
        model, tokenizer = ankh.load_large_model()
    else:
        model, tokenizer = ankh.load_base_model()
    model = model.to(device).eval()
    logging.info(f"Ankh-{model_size} loaded on {device}.")
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


# ── Generate Ankh embeddings ─────────────────────────────────────
def get_embeddings(model, tokenizer, seqs):
    """
    Returns per-residue embeddings from Ankh's encoder hidden states.
    Output shape per sequence: (seq_len, hidden_dim)
      - Ankh-base:  hidden_dim = 768
      - Ankh-large: hidden_dim = 1536
    Special tokens (BOS/EOS) are stripped so the embedding
    aligns 1-to-1 with the input amino acid sequence,
    matching the ESM-2 behaviour of excluding CLS/SEP tokens.
    """
    results = {}

    for seq_id, seq in seqs:
        # Ankh tokenizer expects a list of protein sequences as character lists
        protein_sequences = [list(seq)]

        outputs = tokenizer.batch_encode_plus(
            protein_sequences,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        )

        input_ids      = outputs["input_ids"].to(device)
        attention_mask = outputs["attention_mask"].to(device)

        with torch.no_grad():
            embedding_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Last hidden state: shape (1, seq_len+2, hidden_dim)
        # Strip first (BOS) and last (EOS) special tokens → (seq_len, hidden_dim)
        hidden_states = embedding_output.last_hidden_state  # (1, L+2, H)
        seq_len = len(seq)
        emb = hidden_states[0, 1:seq_len + 1].detach().cpu().numpy()  # (seq_len, H)

        results[seq_id] = emb
        logging.info(f"  [{seq_id}] embedding shape: {emb.shape}")

        # Free memory per sequence
        del input_ids, attention_mask, embedding_output, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ── Save embeddings as .ankh file using pickle ───────────────────
def save_ankh_embeddings(ankh_data, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(ankh_data, f)


# ── Main processing function ─────────────────────────────────────
def process_fasta(fasta_file, output_file, model, tokenizer):
    try:
        filename = os.path.splitext(os.path.basename(fasta_file))[0]
        seqs     = read_fasta(fasta_file)
        results  = get_embeddings(model, tokenizer, seqs)
        embeddings = results[filename]
        save_ankh_embeddings(embeddings, output_file)
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

    # Validate paths
    if not os.path.exists(args.path_input):
        raise FileNotFoundError(f"Input directory '{args.path_input}' not found!")
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    # Load model once — reused for all FASTA files
    model, tokenizer = get_ankh_model(args.model_size)

    input_files = [f for f in os.listdir(args.path_input) if f.endswith(".fasta")]
    count = 0

    for file in input_files:
        input_path  = os.path.join(args.path_input,  file)
        output_path = os.path.join(args.path_output,
                                   os.path.splitext(file)[0] + ".ankh")
        logging.info(f"Processing: {input_path}")
        process_fasta(input_path, output_path, model, tokenizer)
        count += 1
        gc.collect()

    logging.info(f"Processed {count} sequences successfully!")
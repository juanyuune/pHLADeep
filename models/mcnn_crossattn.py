"""
mcnn_crossattn.py  –  PLM + Peptide-HLA Cross-Attention + Position-Attention + MCNN
======================================================================================
Strategy C: Cross-attention between peptide and HLA embeddings.

Biological motivation:
  HLA class I binding is a lock-and-key interaction. Specific HLA pocket
  residues directly contact peptide anchor positions (P2→Pocket B, P9→Pocket F).
  Cross-attention lets each peptide position attend to all HLA pseudosequence
  positions, learning which HLA residues are relevant for each peptide position.

Architecture:
  1. ESM-2 peptide embedding  : (N, 9,  1280)   Q (queries)
  2. ESM-2 HLA embedding      : (N, 39, 1280)   K, V (keys, values)
  3. Cross-attention          : (N, 9,  1280)   peptide attends to HLA
  4. Residual + LayerNorm     : (N, 9,  1280)
  5. Position-wise attention  : (N, 1,  9, 1280) learned anchor weighting
  6. Multi-window CNN         : concat of MaxPool branches
  7. Dense(hidden) → Dense(2, softmax)

References:
  - TransPHLA  : Chu et al. 2022, Briefings in Bioinformatics
  - pMHCformer : Liu et al. 2023
  - ESM-2      : Lin et al. 2023, Science

Usage:
  python mcnn_crossattn.py \
      --data_path   /srv/jupyterlab/workspace/HLA1/emb/esm2 \
      --hla_emb_dir /srv/jupyterlab/workspace/HLA1/hla/hla_embeddings \
      --output      /srv/jupyterlab/workspace/HLA1/results/crossattn \
      --mode        independent \
      --epochs      20 --filters 1024 --hidden 1500 \
      --window_sizes 2 3 5 9
"""

import os
import gc
import csv
import math
import pickle
import time
import datetime
import argparse
import logging
from time import gmtime, strftime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn import metrics
from sklearn.metrics import roc_curve

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",    type=str, required=True)
parser.add_argument("--hla_emb_dir",  type=str,
                    default="/srv/jupyterlab/workspace/HLA1/hla/hla_embeddings")
parser.add_argument("--output",       type=str, required=True)
parser.add_argument("--mode",         type=str, default="independent",
                    choices=["independent", "cross"])
parser.add_argument("--epochs",       type=int, default=20)
parser.add_argument("--filters",      type=int, default=1024)
parser.add_argument("--hidden",       type=int, default=1500)
parser.add_argument("--batch_size",   type=int, default=256)
parser.add_argument("--kfold",        type=int, default=5)
parser.add_argument("--window_sizes", type=int, nargs="+", default=[2, 3, 5, 9])
parser.add_argument("--num_heads",    type=int, default=8,
                    help="Number of cross-attention heads")
parser.add_argument("--attn_dropout", type=float, default=0.1,
                    help="Dropout rate inside cross-attention")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

MAXSEQ      = 9
HLA_LEN     = 34
EMB_DIM     = 1280
NUM_CLASSES = 2

ALLELE_TO_HLA_FILE = {
    "HLA-A0101": "HLA-A0101.esm2",
    "HLA-A0201": "HLA-A0201.esm2",
    "HLA-A0301": "HLA-A0301.esm2",
    "HLA-A1101": "HLA-A1101.esm2",
    "HLA-A2402": "HLA-A2402.esm2",
    "HLA-B0702": "HLA-B0702.esm2",
    "HLA-B0801": "HLA-B0801.esm2",
    "HLA-B3501": "HLA-B3501.esm2",
    "HLA-B4001": "HLA-B4001.esm2",
    "HLA-B5101": "HLA-B5101.esm2",
    "HLA-B5701": "HLA-B5701.esm2",
}


# ── Data loaders ───────────────────────────────────────────────────────────────
def allele_from_fname(fname):
    return os.path.splitext(fname)[0].split("_")[0]


def load_esm2_file(fpath):
    """Return (9, 1280) float32."""
    with open(fpath, "rb") as f:
        arr = pickle.load(f)
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    assert arr.shape == (MAXSEQ, EMB_DIM), \
        f"Bad peptide shape {arr.shape} in {fpath}"
    return arr   # (9, 1280)


def load_hla_embeddings(hla_emb_dir):
    """Load full HLA pseudosequence embeddings (39, 1280) for each allele."""
    hla_embs = {}
    for key, fname in ALLELE_TO_HLA_FILE.items():
        fpath = os.path.join(hla_emb_dir, fname)
        with open(fpath, "rb") as f:
            arr = pickle.load(f)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        assert arr.shape == (HLA_LEN, EMB_DIM), \
            f"Bad HLA shape {arr.shape} for {key}"
        hla_embs[key] = arr   # (39, 1280) — full sequence, NOT mean-pooled
    logging.info("HLA full embeddings loaded for: %s", list(hla_embs.keys()))
    return hla_embs


def load_split_folder(folder, label, hla_embeddings):
    """
    Returns:
      pep_arr : (N, 9,  1280)  peptide embeddings
      hla_arr : (N, 39, 1280)  HLA embeddings (full, for cross-attention)
      y       : (N,)           int labels
    """
    fnames = sorted(f for f in os.listdir(folder) if f.endswith(".esm2"))
    peps, hlas, skipped = [], [], 0
    for fname in fnames:
        allele = allele_from_fname(fname)
        if allele not in hla_embeddings:
            skipped += 1
            continue
        fpath = os.path.join(folder, fname)
        try:
            pep = load_esm2_file(fpath)          # (9, 1280)
            hla = hla_embeddings[allele]          # (39, 1280)
            peps.append(pep)
            hlas.append(hla)
        except Exception:
            skipped += 1
    if skipped:
        logging.warning("skipped %d files in %s", skipped, folder)
    pep_arr = np.stack(peps, axis=0)             # (N, 9,  1280)
    hla_arr = np.stack(hlas, axis=0)             # (N, 39, 1280)
    y       = np.full(len(peps), label, dtype=np.int32)
    return pep_arr, hla_arr, y


def load_dataset(data_path, hla_embeddings):
    peps_tr, hlas_tr, labels_tr = [], [], []
    peps_te, hlas_te, labels_te = [], [], []

    # Support both flat (data_path/pos_train) and
    # per-allele (data_path/HLA-XXXXX/pos_train) layouts
    allele_dirs = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d.startswith("HLA-")
    )
    if allele_dirs:
        search_roots = [os.path.join(data_path, d) for d in allele_dirs]
        logging.info("Found %d allele subdirs: %s", len(allele_dirs), allele_dirs)
    else:
        search_roots = [data_path]

    for split, label, target in [
        ("pos_train", 1, "tr"), ("neg_train", 0, "tr"),
        ("pos_test",  1, "te"), ("neg_test",  0, "te"),
    ]:
        all_peps, all_hlas, all_ys = [], [], []
        for root in search_roots:
            folder = os.path.join(root, split)
            if not os.path.isdir(folder):
                continue
            p, h, y = load_split_folder(folder, label, hla_embeddings)
            if len(p) > 0:
                all_peps.append(p); all_hlas.append(h); all_ys.append(y)
        if not all_peps:
            logging.warning("No data found for split: %s", split)
            continue
        p = np.concatenate(all_peps, axis=0)
        h = np.concatenate(all_hlas, axis=0)
        y = np.concatenate(all_ys,   axis=0)
        logging.info("  %s: pep=%s hla=%s", split, p.shape, h.shape)
        if target == "tr":
            peps_tr.append(p); hlas_tr.append(h); labels_tr.append(y)
        else:
            peps_te.append(p); hlas_te.append(h); labels_te.append(y)

    pep_tr = np.concatenate(peps_tr, axis=0)
    hla_tr = np.concatenate(hlas_tr, axis=0)
    y_tr   = tf.keras.utils.to_categorical(
                 np.concatenate(labels_tr), NUM_CLASSES)

    pep_te = np.concatenate(peps_te, axis=0)
    hla_te = np.concatenate(hlas_te, axis=0)
    y_te   = tf.keras.utils.to_categorical(
                 np.concatenate(labels_te), NUM_CLASSES)

    return pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te


# ── Cross-Attention layer ──────────────────────────────────────────────────────
class PeptideHLACrossAttention(layers.Layer):
    """
    Multi-head cross-attention: peptide positions attend to HLA positions.

    Q = peptide  (batch, 9,  d_model)
    K = HLA      (batch, 39, d_model)
    V = HLA      (batch, 39, d_model)

    Output: (batch, 9, d_model)  — peptide enriched with HLA context

    Each of the 9 peptide positions learns which of the 39 HLA pseudosequence
    positions are most relevant for binding — directly modelling pocket contacts.
    """
    def __init__(self, d_model=1280, num_heads=8, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.d_model   = d_model
        self.d_head    = d_model // num_heads

        # Projections
        self.W_q = layers.Dense(d_model, use_bias=False)
        self.W_k = layers.Dense(d_model, use_bias=False)
        self.W_v = layers.Dense(d_model, use_bias=False)
        self.W_o = layers.Dense(d_model, use_bias=False)

        self.attn_dropout = layers.Dropout(dropout)
        self.layer_norm   = layers.LayerNormalization(epsilon=1e-6)

    def split_heads(self, x, batch_size):
        """(batch, seq, d_model) → (batch, heads, seq, d_head)"""
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.d_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, peptide, hla, training=False):
        """
        peptide : (batch, 9,  d_model)
        hla     : (batch, 39, d_model)
        returns : (batch, 9,  d_model)
        """
        batch_size = tf.shape(peptide)[0]

        Q = self.split_heads(self.W_q(peptide), batch_size)  # (B, H, 9,  dh)
        K = self.split_heads(self.W_k(hla),     batch_size)  # (B, H, 39, dh)
        V = self.split_heads(self.W_v(hla),     batch_size)  # (B, H, 39, dh)

        # Scaled dot-product attention
        scale  = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        scores = tf.matmul(Q, K, transpose_b=True) / scale   # (B, H, 9, 39)
        weights = tf.nn.softmax(scores, axis=-1)              # (B, H, 9, 39)
        weights = self.attn_dropout(weights, training=training)

        context = tf.matmul(weights, V)                       # (B, H, 9, dh)
        context = tf.transpose(context, perm=[0, 2, 1, 3])   # (B, 9, H, dh)
        context = tf.reshape(context,
                             (batch_size, MAXSEQ, self.d_model))  # (B, 9, d)

        output = self.W_o(context)                            # (B, 9, d_model)

        # Residual connection + LayerNorm
        output = self.layer_norm(peptide + output)            # (B, 9, d_model)
        return output, weights   # return weights for interpretability

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return cfg


# ── Position-wise Attention (same as mcnn_attention.py) ───────────────────────
class PositionAttention(layers.Layer):
    """Softmax weights over the 9 peptide positions."""
    def __init__(self, num_positions=9, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.score_dense   = layers.Dense(num_positions, use_bias=True)
        self.softmax       = layers.Softmax(axis=-1)

    def call(self, x, training=False):
        # x: (batch, 1, 9, D)
        x_sq    = tf.squeeze(x, axis=1)                 # (batch, 9, D)
        summary = tf.reduce_mean(x_sq, axis=-1)         # (batch, 9)
        scores  = self.score_dense(summary)             # (batch, 9)
        weights = self.softmax(scores)                  # (batch, 9)
        weights = tf.expand_dims(weights, axis=-1)      # (batch, 9, 1)
        attended = x_sq * weights                       # (batch, 9, D)
        return tf.expand_dims(attended, axis=1)         # (batch, 1, 9, D)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_positions": self.num_positions})
        return cfg


# ── Full Model ─────────────────────────────────────────────────────────────────
class DeepScanCrossAttn(Model):
    """
    Peptide-HLA Cross-Attention + Position-wise Attention + Multi-window CNN.

    Inputs: [peptide (B,9,1280), hla (B,39,1280)]
    Output: softmax (B, 2)
    """
    def __init__(self, maxseq, window_sizes, num_filters, num_hidden,
                 num_heads=8, attn_dropout=0.1):
        super().__init__()
        self.window_sizes = window_sizes
        self.maxseq       = maxseq

        # Cross-attention (peptide ← HLA)
        self.cross_attn = PeptideHLACrossAttention(
            d_model=EMB_DIM, num_heads=num_heads, dropout=attn_dropout)

        # Position-wise attention
        self.pos_attn = PositionAttention(num_positions=maxseq)

        # Multi-window CNN branches
        self.conv2d  = []
        self.maxpool = []
        self.flatten = []
        for ws in window_sizes:
            self.conv2d.append(layers.Conv2D(
                filters=num_filters, kernel_size=(1, ws),
                activation='relu', padding='valid',
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer='glorot_uniform'))
            self.maxpool.append(layers.MaxPooling2D(
                pool_size=(1, maxseq - ws + 1),
                strides=(1, maxseq - ws + 1), padding='valid'))
            self.flatten.append(layers.Flatten())

        self.concat  = layers.Concatenate()
        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(num_hidden, activation='relu',
                                bias_initializer=tf.constant_initializer(0.1),
                                kernel_initializer='glorot_uniform')
        self.fc2 = layers.Dense(NUM_CLASSES, activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, inputs, training=False):
        peptide, hla = inputs[0], inputs[1]   # (B,9,1280), (B,39,1280)

        # Step 1: Cross-attention — each peptide position attends to HLA
        x, _ = self.cross_attn(peptide, hla, training=training)  # (B, 9, 1280)

        # Step 2: Add channel dim for CNN → (B, 1, 9, 1280)
        x = tf.expand_dims(x, axis=1)

        # Step 3: Position-wise attention
        x = self.pos_attn(x, training=training)   # (B, 1, 9, 1280)

        # Step 4: Multi-window CNN
        _x = []
        for i in range(len(self.window_sizes)):
            _x.append(self.flatten[i](self.maxpool[i](self.conv2d[i](x))))
        x = self.concat(_x)
        x = self.dropout(x, training=training)
        return self.fc2(self.fc1(x))


# ── DataGenerator ──────────────────────────────────────────────────────────────
class DataGenerator(tf.keras.utils.Sequence):
    """Yields [peptide_batch, hla_batch], label_batch."""
    def __init__(self, pep, hla, y, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.pep, self.hla, self.y = pep, hla, y
        self.bs  = batch_size
        self.idx = np.arange(len(pep))

    def __len__(self):
        return int(np.ceil(len(self.pep) / self.bs))

    def __getitem__(self, i):
        idx = self.idx[i*self.bs:(i+1)*self.bs]
        return ((np.array([self.pep[j] for j in idx]),
                 np.array([self.hla[j] for j in idx])),
                np.array([self.y[j]   for j in idx]))

    def on_epoch_end(self):
        np.random.shuffle(self.idx)


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(model, pep_te, hla_te, y_te):
    pred = model.predict([pep_te, hla_te], verbose=0)
    y_true = np.argmax(y_te, axis=1)           # integer labels (0/1)
    fpr, tpr, thresholds = roc_curve(y_true, pred[:, 1])
    auc = metrics.auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]
    y_pred = (pred[:, 1] >= threshold).astype(int)
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc  = (TP + TN) / (TP + FP + TN + FN)
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    MCC  = (TP*TN - FP*FN) / denom if denom > 0 else 0.0
    return int(TP), int(FP), int(TN), int(FN), Sens, Spec, Acc, MCC, auc


# ── Train + eval ───────────────────────────────────────────────────────────────
def train_and_eval(pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te, run_label=""):
    model = DeepScanCrossAttn(
        maxseq=MAXSEQ,
        window_sizes=args.window_sizes,
        num_filters=args.filters,
        num_hidden=args.hidden,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout)

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Build by passing dummy data
    dummy_pep = np.zeros((1, MAXSEQ, EMB_DIM), dtype=np.float32)
    dummy_hla = np.zeros((1, HLA_LEN, EMB_DIM), dtype=np.float32)
    _ = model([dummy_pep, dummy_hla], training=False)
    logging.info("Model built. Trainable params: %s",
                 f"{model.count_params():,}")

    gen = DataGenerator(pep_tr, hla_tr, y_tr, args.batch_size)
    cb  = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.fit(gen, epochs=args.epochs, callbacks=[cb], verbose=2)

    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = evaluate(
        model, pep_te, hla_te, y_te)
    logging.info(
        "[%s] Sens=%.4f Spec=%.4f Acc=%.4f MCC=%.4f AUC=%.4f",
        run_label, Sens, Spec, Acc, MCC, AUC)

    # Extract position attention weights
    attn_layer = model.pos_attn
    sample_pep = pep_te[:min(500, len(pep_te))]
    sample_hla = hla_te[:min(500, len(hla_te))]
    # Run cross-attn first to get the attended representation
    attended, _ = model.cross_attn(sample_pep, sample_hla, training=False)
    attended_exp = tf.expand_dims(attended, axis=1)
    summary = tf.reduce_mean(tf.squeeze(attended_exp, axis=1), axis=-1)
    scores  = attn_layer.score_dense(summary)
    pos_weights = tf.nn.softmax(scores, axis=-1).numpy().mean(axis=0)
    logging.info("  Position weights (P1–P9): %s",
                 " ".join(f"{w:.3f}" for w in pos_weights))

    del model, gen
    gc.collect()
    tf.keras.backend.clear_session()

    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, pos_weights


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    logging.info("=== PLM + Peptide-HLA Cross-Attention + Position-Attention + MCNN ===")
    logging.info("data_path   : %s", args.data_path)
    logging.info("hla_emb_dir : %s", args.hla_emb_dir)
    logging.info("output      : %s", args.output)
    logging.info("num_heads   : %d", args.num_heads)

    hla_embeddings = load_hla_embeddings(args.hla_emb_dir)

    logging.info("Loading data ...")
    pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te = load_dataset(
        args.data_path, hla_embeddings)
    logging.info("train: pep=%s hla=%s  test: pep=%s hla=%s",
                 pep_tr.shape, hla_tr.shape, pep_te.shape, hla_te.shape)

    results_path = os.path.join(args.output, "results_crossattn.csv")
    write_header = not os.path.exists(results_path)

    if args.mode == "independent":
        t0 = time.time()
        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, pos_w = train_and_eval(
            pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te,
            run_label="StratC_CrossAttn")
        elapsed = time.time() - t0

        with open(results_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp", "mode", "model_type",
                    "window_sizes", "filters", "hidden", "epochs",
                    "num_heads",
                    "TP", "FP", "TN", "FN",
                    "Sens", "Spec", "Acc", "MCC", "AUC",
                    "attn_P1","attn_P2","attn_P3","attn_P4","attn_P5",
                    "attn_P6","attn_P7","attn_P8","attn_P9",
                    "elapsed_s"
                ])
            w.writerow([
                time.ctime(), "independent", "StratC_CrossAttn",
                str(args.window_sizes), args.filters, args.hidden, args.epochs,
                args.num_heads,
                TP, FP, TN, FN,
                f"{Sens:.4f}", f"{Spec:.4f}", f"{Acc:.4f}",
                f"{MCC:.4f}", f"{AUC:.4f}",
                *[f"{v:.4f}" for v in pos_w],
                f"{elapsed:.1f}"
            ])
        logging.info("Results saved: %s", results_path)

    elif args.mode == "cross":
        from sklearn.model_selection import StratifiedKFold
        kfold  = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=2)
        y_flat = np.argmax(y_tr, axis=1)
        fold_results = []

        for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(pep_tr, y_flat), 1):
            logging.info("  Cross-val fold %d/%d", fold_i, args.kfold)
            TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, pos_w = train_and_eval(
                pep_tr[tr_idx], hla_tr[tr_idx], y_tr[tr_idx],
                pep_tr[va_idx], hla_tr[va_idx], y_tr[va_idx],
                run_label=f"StratC_fold{fold_i}")
            fold_results.append([Sens, Spec, Acc, MCC, AUC])
            gc.collect()

        mean_r = np.mean(fold_results, axis=0)
        logging.info(
            "Cross-val mean: Sens=%.4f Spec=%.4f Acc=%.4f MCC=%.4f AUC=%.4f",
            *mean_r)

        with open(results_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp", "mode", "model_type",
                    "window_sizes", "filters", "hidden", "epochs", "num_heads",
                    "Sens", "Spec", "Acc", "MCC", "AUC"
                ])
            w.writerow([
                time.ctime(), "cross", "StratC_CrossAttn",
                str(args.window_sizes), args.filters, args.hidden, args.epochs,
                args.num_heads,
                f"{mean_r[0]:.4f}", f"{mean_r[1]:.4f}", f"{mean_r[2]:.4f}",
                f"{mean_r[3]:.4f}", f"{mean_r[4]:.4f}"
            ])
        logging.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()
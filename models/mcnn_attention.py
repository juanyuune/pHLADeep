"""
mcnn_attention.py  –  PLM + Position-wise Attention + MCNN
============================================================
Adds a learnable position-wise attention layer between the ESM-2
embeddings and the Multi-window CNN.

Architecture:
  Input  : (N, 1, 9, D)          D=1280 peptide-only | D=2560 HLA-fused
  Squeeze: (N, 9, D)
  Attn   : softmax weights over 9 positions → weighted (N, 9, D)
  Expand : (N, 1, 9, D)
  CNN    : Multi-window Conv2D + MaxPool + Flatten → concat
  FC     : Dense(hidden) → Dense(2, softmax)

The attention is a single linear layer (9 → 9) with softmax, applied
per-sample across the position axis.  No extra parameters over the
feature dimension, so it stays lightweight.

Usage (peptide-only, independent):
  python mcnn_attention.py \
      --data_path /srv/jupyterlab/workspace/HLA1/emb/esm2 \
      --output    /srv/jupyterlab/workspace/HLA1/results/attention \
      --mode      independent \
      --epochs    20 --filters 1024 --hidden 1500 \
      --window_sizes 2 3 5 9

Usage (HLA-fused, independent):
  python mcnn_attention.py \
      --data_path    /srv/jupyterlab/workspace/HLA1/emb/esm2 \
      --hla_emb_dir  /srv/jupyterlab/workspace/HLA1/hla/hla_embeddings \
      --use_hla \
      --output       /srv/jupyterlab/workspace/HLA1/results/attention_hla \
      --mode         independent \
      --epochs       20 --filters 1024 --hidden 1500 \
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
parser.add_argument("--use_hla",      action="store_true",
                    help="Fuse HLA pseudosequence (Strategy B)")
parser.add_argument("--output",       type=str, required=True)
parser.add_argument("--mode",         type=str, default="independent",
                    choices=["independent", "cross"])
parser.add_argument("--epochs",       type=int, default=20)
parser.add_argument("--filters",      type=int, default=1024)
parser.add_argument("--hidden",       type=int, default=1500)
parser.add_argument("--batch_size",   type=int, default=512)
parser.add_argument("--kfold",        type=int, default=5)
parser.add_argument("--window_sizes", type=int, nargs="+", default=[2, 3, 5, 9])
parser.add_argument("--attn_heads",   type=int, default=1,
                    help="Number of attention heads (1=single scalar weight per position)")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

MAXSEQ      = 9
NUM_CLASSES = 2
ALLELE_TO_HLA_FILE = {
    "HLA-A0201": "HLA-A0201.esm2",
    "HLA-A0301": "HLA-A0301.esm2",
    "HLA-B0801": "HLA-B0801.esm2",
    "HLA-B5701": "HLA-B5701.esm2",
}


# ── Data loaders ───────────────────────────────────────────────────────────────
def allele_from_fname(fname):
    return os.path.splitext(fname)[0].split("_")[0]


def load_esm2_file(fpath):
    with open(fpath, "rb") as f:
        arr = pickle.load(f)
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr   # (9, 1280)


def load_hla_mean_vecs(hla_emb_dir):
    mean_vecs = {}
    for key, fname in ALLELE_TO_HLA_FILE.items():
        fpath = os.path.join(hla_emb_dir, fname)
        with open(fpath, "rb") as f:
            arr = pickle.load(f)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        assert arr.shape == (39, 1280), f"Bad HLA shape {arr.shape} for {key}"
        mean_vecs[key] = arr.mean(axis=0)   # (1280,)
    logging.info("HLA mean-pool loaded for: %s", list(mean_vecs.keys()))
    return mean_vecs


def load_split_folder(folder, label, hla_mean_vecs=None):
    fnames = sorted(f for f in os.listdir(folder) if f.endswith(".esm2"))
    samples, skipped = [], 0
    for fname in fnames:
        fpath = os.path.join(folder, fname)
        try:
            pep = load_esm2_file(fpath)          # (9, 1280)
            if hla_mean_vecs is not None:
                allele = allele_from_fname(fname)
                hla_t  = np.tile(hla_mean_vecs[allele][np.newaxis, :], (9, 1))
                feat   = np.concatenate([pep, hla_t], axis=1)  # (9, 2560)
            else:
                feat = pep
            samples.append(feat[np.newaxis, :, :])   # (1, 9, D)
        except Exception as e:
            skipped += 1
    if skipped:
        logging.warning("skipped %d files in %s", skipped, folder)
    x = np.stack(samples, axis=0)               # (N, 1, 9, D)
    y = np.full(len(samples), label, dtype=np.int32)
    return x, y


def load_dataset(data_path, hla_mean_vecs=None):
    pieces_tr, labels_tr = [], []
    pieces_te, labels_te = [], []
    for split, label, target in [
        ("pos_train", 1, "tr"), ("neg_train", 0, "tr"),
        ("pos_test",  1, "te"), ("neg_test",  0, "te"),
    ]:
        folder = os.path.join(data_path, split)
        x, y = load_split_folder(folder, label, hla_mean_vecs)
        logging.info("  %s: %s", split, x.shape)
        if target == "tr":
            pieces_tr.append(x); labels_tr.append(y)
        else:
            pieces_te.append(x); labels_te.append(y)

    x_tr = np.concatenate(pieces_tr, axis=0)
    y_tr = tf.keras.utils.to_categorical(np.concatenate(labels_tr), NUM_CLASSES)
    x_te = np.concatenate(pieces_te, axis=0)
    y_te = tf.keras.utils.to_categorical(np.concatenate(labels_te), NUM_CLASSES)
    return x_tr, y_tr, x_te, y_te


# ── Position-wise Attention layer ──────────────────────────────────────────────
class PositionAttention(layers.Layer):
    """
    Learnable softmax weights over the 9 peptide positions.

    Input  : (batch, 1, 9, D)
    Output : (batch, 1, 9, D)   — same shape, positions re-weighted

    Implementation:
      1. Squeeze channel dim  → (batch, 9, D)
      2. Mean-pool over D     → (batch, 9)      (position summary)
      3. Dense(9) + softmax   → (batch, 9)      (attention weights)
      4. Multiply back        → (batch, 9, D)
      5. Expand channel dim   → (batch, 1, 9, D)
    """
    def __init__(self, num_positions=9, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.score_dense = layers.Dense(num_positions, use_bias=True)
        self.softmax     = layers.Softmax(axis=-1)

    def call(self, x, training=False):
        # x: (batch, 1, 9, D)
        x_sq   = tf.squeeze(x, axis=1)                  # (batch, 9, D)
        summary = tf.reduce_mean(x_sq, axis=-1)          # (batch, 9)
        scores  = self.score_dense(summary)              # (batch, 9)
        weights = self.softmax(scores)                   # (batch, 9)
        weights = tf.expand_dims(weights, axis=-1)       # (batch, 9, 1)
        attended = x_sq * weights                        # (batch, 9, D)
        return tf.expand_dims(attended, axis=1)          # (batch, 1, 9, D)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_positions": self.num_positions})
        return cfg


# ── Model ──────────────────────────────────────────────────────────────────────
class DeepScanAttn(Model):
    """Multi-window CNN with position-wise attention pre-layer."""
    def __init__(self, maxseq, num_feature, window_sizes, num_filters, num_hidden):
        super().__init__()
        self.window_sizes = window_sizes
        self.maxseq       = maxseq

        # Position attention
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
                strides=(1, maxseq), padding='valid'))
            self.flatten.append(layers.Flatten())

        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(num_hidden, activation='relu',
                                bias_initializer=tf.constant_initializer(0.1),
                                kernel_initializer='glorot_uniform')
        self.fc2 = layers.Dense(NUM_CLASSES, activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, training=False):
        # Position attention gate
        x = self.pos_attn(x, training=training)   # (batch, 1, 9, D)

        # Multi-window CNN
        _x = []
        for i in range(len(self.window_sizes)):
            _x.append(self.flatten[i](self.maxpool[i](self.conv2d[i](x))))
        x = layers.Concatenate()(_x)
        x = self.dropout(x, training=training)
        return self.fc2(self.fc1(x))


# ── DataGenerator ──────────────────────────────────────────────────────────────
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y, self.bs = x, y, batch_size
        self.idx = np.arange(len(x))
    def __len__(self): return int(np.ceil(len(self.x) / self.bs))
    def __getitem__(self, i):
        idx = self.idx[i*self.bs:(i+1)*self.bs]
        return np.array([self.x[j] for j in idx]), np.array([self.y[j] for j in idx])


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(model, x_test, y_test):
    pred = model.predict(x_test, verbose=0)
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], pred[:, 1])
    auc = metrics.auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]
    y_pred = (pred[:, 1] >= threshold).astype(int)
    TN, FP, FN, TP = metrics.confusion_matrix(y_test[:, 1], y_pred).ravel()
    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc  = (TP + TN) / (TP + FP + TN + FN)
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    MCC  = (TP*TN - FP*FN) / denom if denom > 0 else 0.0
    return int(TP), int(FP), int(TN), int(FN), Sens, Spec, Acc, MCC, auc


# ── Train + eval ───────────────────────────────────────────────────────────────
def train_and_eval(x_tr, y_tr, x_te, y_te, num_feature, run_label=""):
    model = DeepScanAttn(
        maxseq=MAXSEQ, num_feature=num_feature,
        window_sizes=args.window_sizes,
        num_filters=args.filters, num_hidden=args.hidden)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.build((None, 1, MAXSEQ, num_feature))

    gen = DataGenerator(x_tr, y_tr, args.batch_size)
    cb  = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.fit(gen, epochs=args.epochs, callbacks=[cb], verbose=2)

    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = evaluate(model, x_te, y_te)
    logging.info("[%s] Sens=%.4f Spec=%.4f Acc=%.4f MCC=%.4f AUC=%.4f",
                 run_label, Sens, Spec, Acc, MCC, AUC)

    # Extract learned attention weights (averaged over test set)
    attn_layer = model.pos_attn
    test_sample = x_te[:min(500, len(x_te))]
    summary = tf.reduce_mean(tf.squeeze(test_sample, axis=1), axis=-1)  # (N,9)
    scores  = attn_layer.score_dense(summary)
    weights = tf.nn.softmax(scores, axis=-1).numpy()   # (N, 9)
    mean_weights = weights.mean(axis=0)                # (9,)
    logging.info("  Attention weights (P1–P9): %s",
                 " ".join(f"{w:.3f}" for w in mean_weights))

    del model, gen
    gc.collect()
    tf.keras.backend.clear_session()

    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, mean_weights


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    logging.info("=== PLM + Position-Attention + MCNN ===")
    logging.info("data_path : %s", args.data_path)
    logging.info("use_hla   : %s", args.use_hla)
    logging.info("output    : %s", args.output)

    hla_mean_vecs = load_hla_mean_vecs(args.hla_emb_dir) if args.use_hla else None
    num_feature   = 2560 if args.use_hla else 1280
    model_tag     = "HLA_StratB_attn" if args.use_hla else "peptide_attn"

    logging.info("Loading data ...")
    x_tr, y_tr, x_te, y_te = load_dataset(args.data_path, hla_mean_vecs)
    logging.info("train: %s  test: %s", x_tr.shape, x_te.shape)

    results_path = os.path.join(args.output, "results_attention.csv")
    write_header = not os.path.exists(results_path)

    if args.mode == "independent":
        t0 = time.time()
        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, attn_w = train_and_eval(
            x_tr, y_tr, x_te, y_te, num_feature, run_label=model_tag)
        elapsed = time.time() - t0

        with open(results_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp", "mode", "model_type",
                    "window_sizes", "filters", "hidden", "epochs",
                    "TP", "FP", "TN", "FN",
                    "Sens", "Spec", "Acc", "MCC", "AUC",
                    "attn_P1","attn_P2","attn_P3","attn_P4","attn_P5",
                    "attn_P6","attn_P7","attn_P8","attn_P9",
                    "elapsed_s"
                ])
            w.writerow([
                time.ctime(), "independent", model_tag,
                str(args.window_sizes), args.filters, args.hidden, args.epochs,
                TP, FP, TN, FN,
                f"{Sens:.4f}", f"{Spec:.4f}", f"{Acc:.4f}",
                f"{MCC:.4f}", f"{AUC:.4f}",
                *[f"{v:.4f}" for v in attn_w],
                f"{elapsed:.1f}"
            ])
        logging.info("Results saved: %s", results_path)

    elif args.mode == "cross":
        from sklearn.model_selection import StratifiedKFold
        kfold  = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=2)
        y_flat = np.argmax(y_tr, axis=1)
        fold_results = []

        for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(x_tr, y_flat), 1):
            logging.info("  Cross-val fold %d/%d", fold_i, args.kfold)
            X_tr, X_va = x_tr[tr_idx], x_tr[va_idx]
            Y_tr, Y_va = y_tr[tr_idx], y_tr[va_idx]
            TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, attn_w = train_and_eval(
                X_tr, Y_tr, X_va, Y_va, num_feature,
                run_label=f"{model_tag}_fold{fold_i}")
            fold_results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
            del X_tr, X_va, Y_tr, Y_va
            gc.collect()

        mean_r = np.mean(fold_results, axis=0)
        logging.info("Cross-val mean: Sens=%.4f Spec=%.4f Acc=%.4f MCC=%.4f AUC=%.4f",
                     *mean_r[4:])

        with open(results_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp", "mode", "model_type",
                    "window_sizes", "filters", "hidden", "epochs",
                    "Sens", "Spec", "Acc", "MCC", "AUC"
                ])
            w.writerow([
                time.ctime(), "cross", model_tag,
                str(args.window_sizes), args.filters, args.hidden, args.epochs,
                f"{mean_r[4]:.4f}", f"{mean_r[5]:.4f}", f"{mean_r[6]:.4f}",
                f"{mean_r[7]:.4f}", f"{mean_r[8]:.4f}"
            ])
        logging.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()
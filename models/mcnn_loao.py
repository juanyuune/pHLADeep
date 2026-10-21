"""
loao_mcnn.py  –  Leave-One-Allele-Out (LOAO) evaluation
=========================================================
Trains on 3 alleles, tests on the held-out 4th, for both:
  (A) Peptide-only   : ESM-2 (9, 1280) → (N, 1, 9, 1280)
  (B) HLA-fused StratB: (9, 2560) → (N, 1, 9, 2560)

4 folds total:
  Fold 1: hold out HLA-A*02:01  (A0201)
  Fold 2: hold out HLA-A*03:01  (A0301)
  Fold 3: hold out HLA-B*08:01  (B0801)
  Fold 4: hold out HLA-B*57:01  (B5701)

Data layout (individual .esm2 files):
  /srv/jupyterlab/workspace/HLA1/emb/esm2/{pos_train,neg_train,pos_test,neg_test}/
  Filenames: HLA-A0201_PEPTIDESEQ.esm2

Usage:
  python loao_mcnn.py \
      --emb_path /srv/jupyterlab/workspace/HLA1/emb/esm2 \
      --output   /srv/jupyterlab/workspace/HLA1/results/loao \
      --model    both \
      --epochs   20 \
      --filters  1024 \
      --hidden   1500
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb_path", type=str,
                    default="/srv/jupyterlab/workspace/HLA1/emb/esm2")
parser.add_argument("--hla_emb_dir", type=str,
                    default="/srv/jupyterlab/workspace/HLA1/hla/hla_embeddings")
parser.add_argument("--output",   type=str,
                    default="/srv/jupyterlab/workspace/HLA1/results/loao")
parser.add_argument("--model",    type=str, default="both",
                    choices=["peptide", "hla", "both"])
parser.add_argument("--epochs",   type=int, default=20)
parser.add_argument("--filters",  type=int, default=1024)
parser.add_argument("--hidden",   type=int, default=1500)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--window_sizes", type=int, nargs="+", default=[2, 3, 5, 9])
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

ALLELES    = ["HLA-A0201", "HLA-A0301", "HLA-B0801", "HLA-B5701"]
SPLITS     = ["pos_train", "neg_train", "pos_test", "neg_test"]
NUM_CLASSES = 2

ALLELE_TO_HLA_FILE = {
    "HLA-A0201": "HLA-A0201.esm2",
    "HLA-A0301": "HLA-A0301.esm2",
    "HLA-B0801": "HLA-B0801.esm2",
    "HLA-B5701": "HLA-B5701.esm2",
}

# ── HLA mean-pool embeddings (Strategy B) ─────────────────────────────────────
def load_hla_mean_vecs(hla_emb_dir):
    mean_vecs = {}
    for key, fname in ALLELE_TO_HLA_FILE.items():
        fpath = os.path.join(hla_emb_dir, fname)
        with open(fpath, "rb") as f:
            arr = pickle.load(f)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
        assert arr.shape == (39, 1280), f"Bad HLA shape {arr.shape} for {key}"
        mean_vecs[key] = arr.mean(axis=0)   # (1280,)
    logging.info("HLA mean-pool loaded for: %s", list(mean_vecs.keys()))
    return mean_vecs


# ── Per-file loaders ───────────────────────────────────────────────────────────
def allele_from_fname(fname):
    """HLA-A0201_PEPTIDE.esm2  →  'HLA-A0201'"""
    stem = os.path.splitext(fname)[0]
    return stem.split("_")[0]


def load_esm2_file(fpath):
    """Return (9, 1280) numpy float32."""
    with open(fpath, "rb") as f:
        arr = pickle.load(f)
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected shape {arr.shape} in {fpath}")
    return arr   # (9, 1280)


def load_split_folder(folder, label, allele_filter=None, hla_mean_vecs=None):
    """
    Load all .esm2 files from `folder`.

    allele_filter : list of allele keys to INCLUDE (None = include all)
    hla_mean_vecs : if provided, fuse HLA → (N,1,9,2560); else peptide-only (N,1,9,1280)

    Returns (x, y_raw) where y_raw is int array of `label`.
    """
    fnames = sorted(f for f in os.listdir(folder) if f.endswith(".esm2"))
    samples = []
    skipped = 0
    for fname in fnames:
        allele = allele_from_fname(fname)
        if allele_filter is not None and allele not in allele_filter:
            continue
        fpath = os.path.join(folder, fname)
        try:
            pep = load_esm2_file(fpath)          # (9, 1280)
            if hla_mean_vecs is not None:
                hla  = hla_mean_vecs[allele]     # (1280,)
                hla_t = np.tile(hla[np.newaxis, :], (9, 1))   # (9, 1280)
                feat  = np.concatenate([pep, hla_t], axis=1)  # (9, 2560)
            else:
                feat = pep                        # (9, 1280)
            samples.append(feat[np.newaxis, :, :])  # (1, 9, D)
        except Exception as e:
            skipped += 1
    if skipped:
        logging.warning("skipped %d files in %s", skipped, folder)
    if not samples:
        raise ValueError(f"No samples loaded from {folder} with filter {allele_filter}")
    x = np.stack(samples, axis=0)          # (N, 1, 9, D)
    y = np.full(len(samples), label, dtype=np.int32)
    return x, y


def build_dataset(emb_path, train_alleles, test_alleles, hla_mean_vecs=None):
    """
    Combine pos/neg from all 4 split folders, filtering by allele lists.
    Returns x_train, y_train, x_test, y_test (one-hot labels).
    """
    pieces_tr, labels_tr = [], []
    pieces_te, labels_te = [], []

    for split, label in [("pos_train", 1), ("neg_train", 0)]:
        folder = os.path.join(emb_path, split)
        x, y = load_split_folder(folder, label, train_alleles, hla_mean_vecs)
        pieces_tr.append(x); labels_tr.append(y)
        logging.info("  %s (train alleles): %s", split, x.shape)

    # test alleles come from BOTH pos/neg test AND train folders
    # (we want every sample of the held-out allele regardless of split)
    for split, label in [("pos_train", 1), ("neg_train", 0),
                          ("pos_test",  1), ("neg_test",  0)]:
        folder = os.path.join(emb_path, split)
        x, y = load_split_folder(folder, label, test_alleles, hla_mean_vecs)
        pieces_te.append(x); labels_te.append(y)
        logging.info("  %s (test allele): %s", split, x.shape)

    x_tr = np.concatenate(pieces_tr, axis=0)
    y_tr = tf.keras.utils.to_categorical(
        np.concatenate(labels_tr), NUM_CLASSES)

    x_te = np.concatenate(pieces_te, axis=0)
    y_te = tf.keras.utils.to_categorical(
        np.concatenate(labels_te), NUM_CLASSES)

    return x_tr, y_tr, x_te, y_te


# ── Model ──────────────────────────────────────────────────────────────────────
class DeepScan(Model):
    def __init__(self, maxseq, num_feature,
                 window_sizes, num_filters, num_hidden):
        super().__init__()
        self.window_sizes = window_sizes
        self.maxseq = maxseq
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
        _x = []
        for i in range(len(self.window_sizes)):
            _x.append(self.flatten[i](self.maxpool[i](self.conv2d[i](x))))
        x = layers.Concatenate()(_x)
        x = self.dropout(x, training=training)
        return self.fc2(self.fc1(x))


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
    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, auc


# ── Train one fold ─────────────────────────────────────────────────────────────
def train_and_eval(x_tr, y_tr, x_te, y_te, maxseq, num_feature, label):
    model = DeepScan(
        maxseq=maxseq, num_feature=num_feature,
        window_sizes=args.window_sizes,
        num_filters=args.filters, num_hidden=args.hidden)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.build((None, 1, maxseq, num_feature))

    gen = DataGenerator(x_tr, y_tr, args.batch_size)
    cb  = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.fit(gen, epochs=args.epochs, callbacks=[cb], verbose=0)

    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = evaluate(model, x_te, y_te)
    logging.info(
        "[%s] Sens=%.4f Spec=%.4f Acc=%.4f MCC=%.4f AUC=%.4f",
        label, Sens, Spec, Acc, MCC, AUC)

    del model, gen
    gc.collect()
    tf.keras.backend.clear_session()

    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC


# ── Main LOAO loop ─────────────────────────────────────────────────────────────
def main():
    logging.info("=== Leave-One-Allele-Out Evaluation ===")
    logging.info("emb_path : %s", args.emb_path)
    logging.info("output   : %s", args.output)

    hla_mean_vecs = load_hla_mean_vecs(args.hla_emb_dir) \
        if args.model in ("hla", "both") else None

    results_path = os.path.join(args.output, "results_loao.csv")
    header_written = os.path.exists(results_path)

    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not header_written:
            writer.writerow([
                "timestamp", "held_out_allele", "model_type",
                "window_sizes", "filters", "hidden", "epochs",
                "TP", "FP", "TN", "FN",
                "Sens", "Spec", "Acc", "MCC", "AUC",
                "n_train_pos", "n_train_neg", "n_test_pos", "n_test_neg"
            ])

        for held_out in ALLELES:
            train_alleles = [a for a in ALLELES if a != held_out]
            test_alleles  = [held_out]

            logging.info("\n─── Fold: hold out %s ───", held_out)
            logging.info("    train alleles: %s", train_alleles)

            # ── Peptide-only ───────────────────────────────────────────────
            if args.model in ("peptide", "both"):
                logging.info("  [peptide-only] Building dataset ...")
                x_tr, y_tr, x_te, y_te = build_dataset(
                    args.emb_path, train_alleles, test_alleles,
                    hla_mean_vecs=None)
                n_tr_pos = int((np.argmax(y_tr, axis=1) == 1).sum())
                n_tr_neg = int((np.argmax(y_tr, axis=1) == 0).sum())
                n_te_pos = int((np.argmax(y_te, axis=1) == 1).sum())
                n_te_neg = int((np.argmax(y_te, axis=1) == 0).sum())
                logging.info("  train pos=%d neg=%d | test pos=%d neg=%d",
                             n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)

                t0 = time.time()
                TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = train_and_eval(
                    x_tr, y_tr, x_te, y_te,
                    maxseq=9, num_feature=1280,
                    label=f"peptide hold_out={held_out}")
                elapsed = time.time() - t0

                writer.writerow([
                    time.ctime(), held_out, "peptide_only",
                    str(args.window_sizes), args.filters, args.hidden, args.epochs,
                    TP, FP, TN, FN,
                    f"{Sens:.4f}", f"{Spec:.4f}", f"{Acc:.4f}",
                    f"{MCC:.4f}", f"{AUC:.4f}",
                    n_tr_pos, n_tr_neg, n_te_pos, n_te_neg
                ])
                f.flush()
                del x_tr, y_tr, x_te, y_te
                gc.collect()

            # ── HLA-fused (Strategy B) ─────────────────────────────────────
            if args.model in ("hla", "both"):
                logging.info("  [HLA-fused] Building dataset ...")
                x_tr, y_tr, x_te, y_te = build_dataset(
                    args.emb_path, train_alleles, test_alleles,
                    hla_mean_vecs=hla_mean_vecs)
                n_tr_pos = int((np.argmax(y_tr, axis=1) == 1).sum())
                n_tr_neg = int((np.argmax(y_tr, axis=1) == 0).sum())
                n_te_pos = int((np.argmax(y_te, axis=1) == 1).sum())
                n_te_neg = int((np.argmax(y_te, axis=1) == 0).sum())
                logging.info("  train pos=%d neg=%d | test pos=%d neg=%d",
                             n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)

                t0 = time.time()
                TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = train_and_eval(
                    x_tr, y_tr, x_te, y_te,
                    maxseq=9, num_feature=2560,
                    label=f"HLA_StratB hold_out={held_out}")
                elapsed = time.time() - t0

                writer.writerow([
                    time.ctime(), held_out, "HLA_StratB",
                    str(args.window_sizes), args.filters, args.hidden, args.epochs,
                    TP, FP, TN, FN,
                    f"{Sens:.4f}", f"{Spec:.4f}", f"{Acc:.4f}",
                    f"{MCC:.4f}", f"{AUC:.4f}",
                    n_tr_pos, n_tr_neg, n_te_pos, n_te_neg
                ])
                f.flush()
                del x_tr, y_tr, x_te, y_te
                gc.collect()

    logging.info("\n=== LOAO complete. Results: %s ===", results_path)


if __name__ == "__main__":
    main()
import os
import gc
import pickle
import logging
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PEP_LEN  = 9
HLA_LEN  = 34

ALLELE_TAGS = [
    "HLA-A0101", "HLA-A0201", "HLA-A0301", "HLA-A1101", "HLA-A2402",
    "HLA-B0702", "HLA-B0801", "HLA-B3501", "HLA-B4001", "HLA-B5101",
    "HLA-B5701",
]


def load_hla_embeddings(hla_dir):
    hla_embs = {}
    for allele_tag in ALLELE_TAGS:
        matched = [f for f in os.listdir(hla_dir) if f.startswith(allele_tag + ".")]
        if not matched:
            raise FileNotFoundError(f"No HLA embedding found for {allele_tag} in {hla_dir}")
        path = os.path.join(hla_dir, matched[0])
        with open(path, "rb") as f:
            emb = pickle.load(f)
        emb = np.array(emb, dtype=np.float32)
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb[0]
        assert emb.shape[0] == HLA_LEN, \
            f"Unexpected HLA length {emb.shape[0]} for {allele_tag}"
        hla_embs[allele_tag] = emb
        logging.info("  Loaded HLA %s: %s", allele_tag, emb.shape)
    return hla_embs


def load_split(pos_npy, neg_npy):
    pos = np.load(pos_npy).astype(np.float32)
    neg = np.load(neg_npy).astype(np.float32)

    def to_4d(arr):
        if arr.ndim == 5:
            arr = arr[:, 0, 0, :, :]
        if arr.ndim == 4:
            arr = arr[:, 0, :, :]
        if arr.ndim == 3:
            arr = arr[:, np.newaxis, :, :]
        return arr

    pos = to_4d(pos)
    neg = to_4d(neg)
    x   = np.concatenate([pos, neg], axis=0)
    y   = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    return x, y


def load_strategy_a(base_path):
    allele_dirs = sorted(
        d for d in os.listdir(base_path)
        if d.startswith("HLA-") and os.path.isdir(os.path.join(base_path, d))
    )
    logging.info("Strategy A: found %d allele dirs", len(allele_dirs))

    x_tr_list, y_tr_list = [], []
    x_te_list, y_te_list = [], []

    for allele_tag in allele_dirs:
        adir   = os.path.join(base_path, allele_tag)
        pos_tr = os.path.join(adir, "pos_train.npy")
        neg_tr = os.path.join(adir, "neg_train.npy")
        pos_te = os.path.join(adir, "pos_test.npy")
        neg_te = os.path.join(adir, "neg_test.npy")

        missing = [p for p in [pos_tr, neg_tr, pos_te, neg_te] if not os.path.exists(p)]
        if missing:
            logging.warning("  %s: missing files %s — skipping", allele_tag, missing)
            continue

        x_tr, y_tr = load_split(pos_tr, neg_tr)
        x_te, y_te = load_split(pos_te, neg_te)
        logging.info("  %s: train=%s  test=%s", allele_tag, x_tr.shape, x_te.shape)
        x_tr_list.append(x_tr); y_tr_list.append(y_tr)
        x_te_list.append(x_te); y_te_list.append(y_te)
        gc.collect()

    x_train = np.concatenate(x_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    x_test  = np.concatenate(x_te_list, axis=0)
    y_test  = np.concatenate(y_te_list, axis=0)

    y_train = tf.keras.utils.to_categorical(y_train, 2)
    y_test  = tf.keras.utils.to_categorical(y_test,  2)

    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test,  y_test  = shuffle(x_test,  y_test,  random_state=42)

    logging.info("Strategy A  train: %s  test: %s", x_train.shape, x_test.shape)
    return x_train, y_train, x_test, y_test


def load_strategy_b(base_path, hla_dir):
    hla_embs = load_hla_embeddings(hla_dir)

    allele_dirs = sorted(
        d for d in os.listdir(base_path)
        if d.startswith("HLA-") and os.path.isdir(os.path.join(base_path, d))
    )
    logging.info("Strategy B: found %d allele dirs", len(allele_dirs))

    x_tr_list, y_tr_list = [], []
    x_te_list, y_te_list = [], []

    for allele_tag in allele_dirs:
        if allele_tag not in hla_embs:
            logging.warning("  %s: no HLA embedding — skipping", allele_tag)
            continue

        adir   = os.path.join(base_path, allele_tag)
        pos_tr = os.path.join(adir, "pos_train.npy")
        neg_tr = os.path.join(adir, "neg_train.npy")
        pos_te = os.path.join(adir, "pos_test.npy")
        neg_te = os.path.join(adir, "neg_test.npy")

        missing = [p for p in [pos_tr, neg_tr, pos_te, neg_te] if not os.path.exists(p)]
        if missing:
            logging.warning("  %s: missing files %s — skipping", allele_tag, missing)
            continue

        x_tr, y_tr = load_split(pos_tr, neg_tr)
        x_te, y_te = load_split(pos_te, neg_te)

        hla_vec = hla_embs[allele_tag]

        def fuse(x_pep, hla_vec):
            N = x_pep.shape[0]
            hla_tiled = np.tile(hla_vec[np.newaxis, np.newaxis, :, :], (N, 1, 1, 1))
            return np.concatenate([x_pep, hla_tiled], axis=2)

        x_tr = fuse(x_tr, hla_vec)
        x_te = fuse(x_te, hla_vec)

        logging.info("  %s: train=%s  test=%s", allele_tag, x_tr.shape, x_te.shape)
        x_tr_list.append(x_tr); y_tr_list.append(y_tr)
        x_te_list.append(x_te); y_te_list.append(y_te)
        gc.collect()

    x_train = np.concatenate(x_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    x_test  = np.concatenate(x_te_list, axis=0)
    y_test  = np.concatenate(y_te_list, axis=0)

    y_train = tf.keras.utils.to_categorical(y_train, 2)
    y_test  = tf.keras.utils.to_categorical(y_test,  2)

    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test,  y_test  = shuffle(x_test,  y_test,  random_state=42)

    logging.info("Strategy B  train: %s  test: %s", x_train.shape, x_test.shape)
    return x_train, y_train, x_test, y_test


def MCNN_data_load(base_path, mode="A",
                   hla_dir="/srv/jupyterlab/workspace/HLA1/hla/hla_embeddings"):
    mode = mode.upper()
    if mode == "A":
        return load_strategy_a(base_path)
    elif mode == "B":
        return load_strategy_b(base_path, hla_dir)
    else:
        raise ValueError(f"mode must be 'A' or 'B', got '{mode}'")

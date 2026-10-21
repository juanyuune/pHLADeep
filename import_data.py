"""
import_data.py  —  Strategy A and Strategy B data loader
=========================================================

Strategy A  (peptide only):
    MCNN_data_load(base_path, mode="A")
    Input shape : (N, 1, 9, dim)   →  use --maxseq 9

Strategy B  (peptide + HLA concatenated):
    MCNN_data_load(base_path, mode="B")
    Input shape : (N, 1, 43, dim)  →  use --maxseq 43  (9 pep + 34 HLA)

Directory structure expected:
    base_path/
    ├── HLA-A0101/
    │   ├── pos_train.npy
    │   ├── neg_train.npy
    │   ├── pos_test.npy
    │   └── neg_test.npy
    └── ...

Usage:
    from import_data import MCNN_data_load

    # Strategy A
    x_tr, y_tr, x_te, y_te = MCNN_data_load(args.data_path, mode="A")

    # Strategy B
    x_tr, y_tr, x_te, y_te = MCNN_data_load(args.data_path, mode="B",
                                              hla_dir=args.hla_emb_dir)
"""

import os
import gc
import pickle
import logging
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── Constants ──────────────────────────────────────────────────────────────────
PEP_LEN  = 9
HLA_LEN  = 34

# All 11 allele tags (folder names)
ALLELE_TAGS = [
    "HLA-A0101", "HLA-A0201", "HLA-A0301", "HLA-A1101", "HLA-A2402",
    "HLA-B0702", "HLA-B0801", "HLA-B3501", "HLA-B4001", "HLA-B5101",
    "HLA-B5701",
]


# ── HLA loader ─────────────────────────────────────────────────────────────────
def load_hla_embeddings(hla_dir: str) -> dict:
    """Load all 11 HLA pseudosequence embeddings.
    Auto-detects file extension (.esm2, .ankh, .pt5, .prostt5, etc.)
    and embedding dimension from the files in hla_dir.
    Returns dict: allele_tag → np.array (34, dim).
    """
    hla_embs = {}
    for allele_tag in ALLELE_TAGS:
        # Find any file matching HLA-A0101.* in hla_dir
        matched = [
            f for f in os.listdir(hla_dir)
            if f.startswith(allele_tag + ".")
        ]
        if not matched:
            raise FileNotFoundError(
                f"No HLA embedding found for {allele_tag} in {hla_dir}\n"
                f"Expected a file like {allele_tag}.esm2 or {allele_tag}.ankh"
            )
        fname = matched[0]
        path  = os.path.join(hla_dir, fname)

        with open(path, "rb") as f:
            emb = pickle.load(f)
        emb = np.array(emb, dtype=np.float32)
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb[0]                           # (1, 34, dim) → (34, dim)
        assert emb.shape[0] == HLA_LEN, \
            f"Unexpected HLA length {emb.shape[0]} for {allele_tag}, expected {HLA_LEN}"
        hla_embs[allele_tag] = emb
        logging.info(f"  Loaded HLA {allele_tag}: {emb.shape}  [{fname}]")
    return hla_embs


# ── Per-split loader ───────────────────────────────────────────────────────────
def load_split(pos_npy: str, neg_npy: str) -> tuple:
    """Load one pos/neg .npy pair.
    Handles both (N, 1, 9, dim) and (N, 1, 1, 9, dim) from get_dataset.py.
    Returns pep_arr (N, 1, 9, dim), y_raw (N,) int.
    """
    pos = np.load(pos_npy).astype(np.float32)
    neg = np.load(neg_npy).astype(np.float32)

    # Squeeze extra dims produced by some get_dataset.py versions
    def to_4d(arr):
        if arr.ndim == 5:
            arr = arr[:, 0, 0, :, :]        # (N, 9, dim)
        if arr.ndim == 4:
            arr = arr[:, 0, :, :]           # (N, 9, dim)
        if arr.ndim == 3:
            arr = arr[:, np.newaxis, :, :]  # (N, 1, 9, dim)
        return arr

    pos = to_4d(pos)
    neg = to_4d(neg)

    x    = np.concatenate([pos, neg], axis=0)
    y    = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    return x, y


# ── Strategy A: peptide only ───────────────────────────────────────────────────
def load_strategy_a(base_path: str):
    """Pool all 11 alleles, peptide embedding only.
    Returns x (N, 1, 9, dim), y (N, 2) one-hot.
    """
    allele_dirs = sorted(
        d for d in os.listdir(base_path)
        if d.startswith("HLA-") and os.path.isdir(os.path.join(base_path, d))
    )
    logging.info(f"Strategy A: found {len(allele_dirs)} allele dirs")

    x_tr_list, y_tr_list = [], []
    x_te_list, y_te_list = [], []

    for allele_tag in allele_dirs:
        adir = os.path.join(base_path, allele_tag)
        pos_tr = os.path.join(adir, "pos_train.npy")
        neg_tr = os.path.join(adir, "neg_train.npy")
        pos_te = os.path.join(adir, "pos_test.npy")
        neg_te = os.path.join(adir, "neg_test.npy")

        missing = [p for p in [pos_tr, neg_tr, pos_te, neg_te]
                   if not os.path.exists(p)]
        if missing:
            logging.warning(f"  {allele_tag}: missing files {missing} — skipping")
            continue

        x_tr, y_tr = load_split(pos_tr, neg_tr)
        x_te, y_te = load_split(pos_te, neg_te)

        logging.info(f"  {allele_tag}: train={x_tr.shape}  test={x_te.shape}")
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

    logging.info(f"Strategy A  train: {x_train.shape}  test: {x_test.shape}")
    logging.info(f"  Train pos={int(y_train[:,1].sum()):,}  neg={int(y_train[:,0].sum()):,}")
    logging.info(f"  Test  pos={int(y_test[:,1].sum()):,}   neg={int(y_test[:,0].sum()):,}")
    return x_train, y_train, x_test, y_test


# ── Strategy B: peptide + HLA concatenated ────────────────────────────────────
def load_strategy_b(base_path: str, hla_dir: str):
    """Pool all 11 alleles, concatenate HLA pseudosequence along seq axis.
    Fused shape: (N, 1, 43, dim)  =  peptide(9) + HLA(34).
    Use --maxseq 43 in mcnn.py.
    """
    hla_embs = load_hla_embeddings(hla_dir)

    allele_dirs = sorted(
        d for d in os.listdir(base_path)
        if d.startswith("HLA-") and os.path.isdir(os.path.join(base_path, d))
    )
    logging.info(f"Strategy B: found {len(allele_dirs)} allele dirs")

    x_tr_list, y_tr_list = [], []
    x_te_list, y_te_list = [], []

    for allele_tag in allele_dirs:
        if allele_tag not in hla_embs:
            logging.warning(f"  {allele_tag}: no HLA embedding — skipping")
            continue

        adir = os.path.join(base_path, allele_tag)
        pos_tr = os.path.join(adir, "pos_train.npy")
        neg_tr = os.path.join(adir, "neg_train.npy")
        pos_te = os.path.join(adir, "pos_test.npy")
        neg_te = os.path.join(adir, "neg_test.npy")

        missing = [p for p in [pos_tr, neg_tr, pos_te, neg_te]
                   if not os.path.exists(p)]
        if missing:
            logging.warning(f"  {allele_tag}: missing files {missing} — skipping")
            continue

        x_tr, y_tr = load_split(pos_tr, neg_tr)   # (N, 1, 9, dim)
        x_te, y_te = load_split(pos_te, neg_te)

        # Tile HLA: (34, dim) → (N, 1, 34, dim)
        hla_vec = hla_embs[allele_tag]             # (34, dim)

        def fuse(x_pep, hla_vec):
            N = x_pep.shape[0]
            hla_tiled = np.tile(
                hla_vec[np.newaxis, np.newaxis, :, :], (N, 1, 1, 1)
            )                                      # (N, 1, 34, dim)
            return np.concatenate([x_pep, hla_tiled], axis=2)  # (N, 1, 43, dim)

        x_tr = fuse(x_tr, hla_vec)
        x_te = fuse(x_te, hla_vec)

        logging.info(f"  {allele_tag}: train={x_tr.shape}  test={x_te.shape}")
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

    logging.info(f"Strategy B  train: {x_train.shape}  test: {x_test.shape}")
    logging.info(f"  Train pos={int(y_train[:,1].sum()):,}  neg={int(y_train[:,0].sum()):,}")
    logging.info(f"  Test  pos={int(y_test[:,1].sum()):,}   neg={int(y_test[:,0].sum()):,}")
    return x_train, y_train, x_test, y_test


# ── Main entry point ───────────────────────────────────────────────────────────
def MCNN_data_load(base_path: str, mode: str = "A",
                   hla_dir: str = "/srv/jupyterlab/workspace/HLA1/hla/hla_embeddings"):
    """
    Args:
        base_path : path to expanded_npy/ root folder
        mode      : "A" = peptide only  |  "B" = peptide + HLA concat
        hla_dir   : folder with HLA embedding files (needed for mode B)
                    Supports any extension: .esm2, .ankh, .pt5, .prostt5

    Returns:
        x_train, y_train, x_test, y_test
        mode A → x shape (N, 1, 9,  dim)  use --maxseq 9
        mode B → x shape (N, 1, 43, dim)  use --maxseq 43
    """
    mode = mode.upper()
    if mode == "A":
        return load_strategy_a(base_path)
    elif mode == "B":
        return load_strategy_b(base_path, hla_dir)
    else:
        raise ValueError(f"mode must be 'A' or 'B', got '{mode}'")
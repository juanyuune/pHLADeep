import os
import gc
import pickle
import numpy as np
import tensorflow as tf

HLA_EMB_DIR = "/srv/jupyterlab/workspace/HLA1/hla/hla_embeddings"

ALLELE_TO_FILE = {
    "HLA-A0201": "HLA-A0201.esm2",
    "HLA-A0301": "HLA-A0301.esm2",
    "HLA-B0801": "HLA-B0801.esm2",
    "HLA-B5701": "HLA-B5701.esm2",
}

def _canonical(allele_str):
    s = allele_str.replace("*", "").replace(":", "").replace(" ", "")
    if s in ALLELE_TO_FILE:
        return s
    if not s.startswith("HLA-"):
        s = "HLA-" + s
    if s in ALLELE_TO_FILE:
        return s
    raise KeyError(f"Cannot map allele '{allele_str}' to a known key. Known: {list(ALLELE_TO_FILE.keys())}")


def _load_hla_embeddings():
    mean_vecs = {}
    for key, fname in ALLELE_TO_FILE.items():
        fpath = os.path.join(HLA_EMB_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"HLA embedding not found: {fpath}")
        with open(fpath, "rb") as f:
            arr = pickle.load(f)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
        assert arr.ndim == 2 and arr.shape == (39, 1280), \
            f"Unexpected HLA embedding shape for {key}: {arr.shape}"
        mean_vecs[key] = arr.mean(axis=0)
        print(f"  [HLA] {key}: mean-pool shape {mean_vecs[key].shape}")
    return mean_vecs


def _load_esm2_file(fpath):
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
        raise ValueError(f"Unexpected peptide embedding shape: {arr.shape} in {fpath}")
    return arr


def _allele_from_filename(fname):
    stem = os.path.splitext(fname)[0]
    allele_raw = stem.split("_")[0]
    return _canonical(allele_raw)


def _load_folder(folder, label, hla_mean_vecs):
    fnames = sorted(f for f in os.listdir(folder) if f.endswith(".esm2"))
    if len(fnames) == 0:
        raise ValueError(f"No .esm2 files found in {folder}")

    samples = []
    skipped = 0
    for fname in fnames:
        fpath = os.path.join(folder, fname)
        try:
            allele_key = _allele_from_filename(fname)
            pep        = _load_esm2_file(fpath)
            hla        = hla_mean_vecs[allele_key]
            hla_tiled  = np.tile(hla[np.newaxis, :], (pep.shape[0], 1))
            fused      = np.concatenate([pep, hla_tiled], axis=1)
            fused      = fused[np.newaxis, :, :]
            samples.append(fused)
        except Exception:
            skipped += 1
            continue

    if skipped > 0:
        print(f"  [warn] skipped {skipped} files in {folder}")

    x = np.concatenate(samples, axis=0)
    if x.ndim == 3:
        x = x[:, np.newaxis, :, :]
    y = np.full(len(samples), label, dtype=np.int32)
    return x, y


def MCNN_data_load_hla(emb_path):
    print(f"[import_data_hla] Loading from: {emb_path}")

    for sub in ["pos_train", "neg_train", "pos_test", "neg_test"]:
        p = os.path.join(emb_path, sub)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Expected folder not found: {p}")

    hla_mean_vecs = _load_hla_embeddings()

    x_pos_tr, y_pos_tr = _load_folder(os.path.join(emb_path, "pos_train"), 1, hla_mean_vecs)
    print(f"  pos_train: {x_pos_tr.shape}")
    x_neg_tr, y_neg_tr = _load_folder(os.path.join(emb_path, "neg_train"), 0, hla_mean_vecs)
    print(f"  neg_train: {x_neg_tr.shape}")
    x_pos_te, y_pos_te = _load_folder(os.path.join(emb_path, "pos_test"),  1, hla_mean_vecs)
    print(f"  pos_test:  {x_pos_te.shape}")
    x_neg_te, y_neg_te = _load_folder(os.path.join(emb_path, "neg_test"),  0, hla_mean_vecs)
    print(f"  neg_test:  {x_neg_te.shape}")

    x_train     = np.concatenate([x_pos_tr, x_neg_tr], axis=0)
    y_train_raw = np.concatenate([y_pos_tr, y_neg_tr], axis=0)
    y_train     = tf.keras.utils.to_categorical(y_train_raw, 2)

    x_test     = np.concatenate([x_pos_te, x_neg_te], axis=0)
    y_test_raw = np.concatenate([y_pos_te, y_neg_te], axis=0)
    y_test     = tf.keras.utils.to_categorical(y_test_raw, 2)

    print(f"  x_train: {x_train.shape}  x_test: {x_test.shape}")

    gc.collect()
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", required=True)
    args = parser.parse_args()

    x_tr, y_tr, x_te, y_te = MCNN_data_load_hla(args.emb_path)
    print("x_train:", x_tr.shape, "y_train:", y_tr.shape)
    print("x_test: ", x_te.shape, "y_test: ", y_te.shape)

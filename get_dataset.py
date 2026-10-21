import os
import argparse
import numpy as np
import h5py


def _detect_format(path):
    with open(path, "rb") as f:
        magic = f.read(16)
    if magic[:6] == b'\x93NUMPY':
        return 'numpy'
    if magic[:4] == b'\x89HDF':
        return 'hdf5'
    if magic[:2] == b'PK':
        return 'torch'
    return 'unknown'


def _load_hdf5(path):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        for k in keys:
            arr = np.array(f[k])
            if arr.ndim >= 1:
                return arr
    raise ValueError(f"No readable array in: {path}")


def _load_torch(path):
    import torch
    t = torch.load(path, map_location="cpu")
    if hasattr(t, 'numpy'):
        return t.numpy()
    if isinstance(t, dict):
        for v in t.values():
            if hasattr(v, 'numpy'):
                return v.numpy()
    raise ValueError(f"Cannot extract array from torch file: {path}")


def loadData(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.hdf5', '.h5'):
        return _load_hdf5(path)

    if ext in ('.txt', '.csv'):
        data = np.loadtxt(path)
        print(f"loaded {path}, shape {data.shape}")
        return data

    if ext in ('.npy', '.esm2', '.prottrans', '.ankh', '.pt5', '.prostt5'):
        fmt = _detect_format(path)
        print(f"  {os.path.basename(path)} detected as {fmt}")

        if fmt == 'numpy':
            data = np.load(path, allow_pickle=True)
            print(f"  shape: {data.shape}")
            return data

        if fmt == 'hdf5':
            data = _load_hdf5(path)
            print(f"  shape (hdf5): {data.shape}")
            return data

        if fmt == 'torch':
            data = _load_torch(path)
            print(f"  shape (torch): {data.shape}")
            return data

        try:
            data = np.load(path, allow_pickle=True)
            print(f"  shape (numpy fallback): {data.shape}")
            return data
        except Exception:
            pass

        # Try pickle for .ankh, .pt5, .prostt5 files
        try:
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            data = np.array(data)
            print(f"  shape (pickle): {data.shape}")
            return data
        except Exception:
            pass

        with open(path, "rb") as f:
            magic = f.read(32)
        raise ValueError(
            f"Unknown format: {path}\n"
            f"First 32 bytes: {magic}"
        )

    raise ValueError(f"Unsupported extension: {path}")


def normalize_embedding(data):
    # make sure shape is (L, feature_dim) before padding
    data = np.array(data)

    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    if data.ndim == 3 and data.shape[1] == 1:
        return data[:, 0, :]
    if data.ndim == 1:
        return data[np.newaxis, :]

    raise ValueError(f"Unexpected embedding shape: {data.shape}")


def get_series_feature(org_data, maxseq):
    # pad or truncate to fixed length (maxseq, dim)
    L, dim = org_data.shape
    print(f"  feature dim: {dim}")

    data = np.zeros((maxseq, dim), dtype=np.float32)
    if L < maxseq:
        data[:L, :] = org_data
    else:
        data[:, :] = org_data[:maxseq, :]

    return data.reshape((1, 1, maxseq, dim))


def saveData(path, data):
    print(f"saving to {path}, shape: {data.shape}")
    np.save(path, data)


def main(path_input, path_output, data_type, maxseq):
    result  = []
    skipped = []

    files = sorted(os.listdir(path_input))

    for fname in files:
        if not fname.endswith(data_type):
            continue

        full_path = os.path.join(path_input, fname)

        try:
            data = loadData(full_path)
        except Exception as e:
            print(f"error loading {full_path}: {e}")
            skipped.append(full_path)
            continue

        if data.size == 0:
            print(f"skipping empty file: {fname}")
            skipped.append(full_path)
            continue

        print(f"\nprocessing: {fname}, shape={data.shape}")

        try:
            data    = normalize_embedding(data)
            feature = get_series_feature(data, maxseq)
            result.append(feature)
        except Exception as e:
            print(f"error processing {fname}: {e}")
            skipped.append(full_path)
            continue

    if skipped:
        print(f"\nskipped {len(skipped)} file(s):")
        for s in skipped:
            print(f"  {s}")

    if len(result) == 0:
        raise ValueError(f"no valid data found in {path_input}")

    final = np.concatenate(result, axis=0)
    saveData(path_output, final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in",     "--path_input",   type=str, required=True)
    parser.add_argument("-out",    "--path_output",  type=str, required=True)
    parser.add_argument("-dt",     "--data_type",    type=str, required=True)
    parser.add_argument("-maxseq", "--max_sequence", type=int, default=0)
    args = parser.parse_args()

    main(args.path_input, args.path_output, args.data_type, args.max_sequence)
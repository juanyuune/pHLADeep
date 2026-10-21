"""
loao_eval.py  –  Leave-One-Allele-Out (LOAO) evaluation
=========================================================
Tests generalisation to unseen alleles: B*40:01 and B*51:01.
For each target allele:
  - Train on all OTHER 10 alleles
  - Test on the held-out allele

Runs for all three strategies:
  A  – peptide only  (uses existing Keras MCNN code: mcnn_keras.py / deepScan)
  B  – peptide+HLA   (same Keras model, maxseq=43)
  C  – cross-attn    (PyTorch model)

NOTE: This script only handles Strategy C (PyTorch).
      For A and B, re-use your existing Keras training scripts with
      --loao_allele argument (see comments inside).

Usage:
  conda activate pytorch-env
  python loao_eval.py \
      --data_path   /srv/jupyterlab/workspace/HLA1/data/expanded_npy_ankh \
      --hla_emb_dir /srv/jupyterlab/workspace/HLA1/hla/hla_embeddings_ankh \
      --output      /srv/jupyterlab/workspace/HLA1/results/loao \
      --num_feature 1536 --plm_name ankh \
      --epochs 20 --filters 1024 --hidden 500 \
      --window_sizes 2 3 5 9 \
      --batch_size 128 \
      --loao_alleles HLA-B4001 HLA-B5101
"""

import os
import gc
import csv
import math
import time
import pickle
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from sklearn.metrics import roc_curve

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",    type=str, required=True)
parser.add_argument("--hla_emb_dir",  type=str, required=True)
parser.add_argument("--output",       type=str, required=True)
parser.add_argument("--num_feature",  type=int, default=1536)
parser.add_argument("--plm_name",     type=str, default="ankh")
parser.add_argument("--epochs",       type=int, default=20)
parser.add_argument("--filters",      type=int, default=1024)
parser.add_argument("--hidden",       type=int, default=500)
parser.add_argument("--batch_size",   type=int, default=128)
parser.add_argument("--window_sizes", type=int, nargs="+", default=[2, 3, 5, 9])
parser.add_argument("--num_heads",    type=int, default=8)
parser.add_argument("--attn_dropout", type=float, default=0.1)
parser.add_argument("--lr",           type=float, default=1e-3)
parser.add_argument("--loao_alleles", type=str, nargs="+",
                    default=["HLA-B4001", "HLA-B5101"],
                    help="Allele tags to hold out one at a time")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

MAXSEQ      = 9
HLA_LEN     = 34
EMB_DIM     = args.num_feature
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Device: %s", device)
if torch.cuda.is_available():
    logging.info("GPU: %s", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

ALLELE_TAGS = [
    "HLA-A0101", "HLA-A0201", "HLA-A0301", "HLA-A1101", "HLA-A2402",
    "HLA-B0702", "HLA-B0801", "HLA-B3501", "HLA-B4001", "HLA-B5101",
    "HLA-B5701",
]

# ── Data ───────────────────────────────────────────────────────────────────────
def load_hla_embeddings(hla_emb_dir):
    hla_embs = {}
    for allele_tag in ALLELE_TAGS:
        matched = [f for f in os.listdir(hla_emb_dir)
                   if f.startswith(allele_tag + ".")]
        if not matched:
            raise FileNotFoundError(f"No HLA embedding for {allele_tag}")
        fpath = os.path.join(hla_emb_dir, matched[0])
        with open(fpath, "rb") as f:
            arr = pickle.load(f)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        hla_embs[allele_tag] = arr
    return hla_embs


def load_npy_split(npy_path, label, hla_emb):
    arr = np.load(npy_path)
    arr = arr[:, 0, :, :].astype(np.float32)
    N   = arr.shape[0]
    hla_tiled = np.tile(hla_emb[np.newaxis, :, :], (N, 1, 1))
    y         = np.full(N, label, dtype=np.int64)
    return arr, hla_tiled, y


def load_loao_dataset(data_path, hla_embeddings, held_out_allele):
    """
    Train on all alleles EXCEPT held_out_allele.
    Test  on ONLY held_out_allele (all its pos+neg samples).
    """
    peps_tr, hlas_tr, labels_tr = [], [], []
    peps_te, hlas_te, labels_te = [], [], []

    allele_dirs = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d.startswith("HLA-")
    )

    for allele_tag in allele_dirs:
        if allele_tag not in hla_embeddings:
            continue
        allele_dir = os.path.join(data_path, allele_tag)
        hla_emb    = hla_embeddings[allele_tag]

        is_held_out = (allele_tag == held_out_allele)

        for split_file, label in [("pos_train.npy", 1), ("neg_train.npy", 0),
                                   ("pos_test.npy",  1), ("neg_test.npy",  0)]:
            fpath = os.path.join(allele_dir, split_file)
            if not os.path.exists(fpath):
                continue
            p, h, y = load_npy_split(fpath, label, hla_emb)

            if is_held_out:
                # ALL samples of held-out allele go to test
                peps_te.append(p);  hlas_te.append(h);  labels_te.append(y)
            else:
                # Training alleles: train split → train, test split → still train
                # (we maximise training data from other alleles)
                peps_tr.append(p);  hlas_tr.append(h);  labels_tr.append(y)

    pep_tr = np.concatenate(peps_tr,   axis=0)
    hla_tr = np.concatenate(hlas_tr,   axis=0)
    y_tr   = np.concatenate(labels_tr, axis=0)
    pep_te = np.concatenate(peps_te,   axis=0)
    hla_te = np.concatenate(hlas_te,   axis=0)
    y_te   = np.concatenate(labels_te, axis=0)

    logging.info("LOAO  held-out=%s", held_out_allele)
    logging.info("  Train: pep=%s  pos=%d neg=%d",
                 pep_tr.shape, (y_tr==1).sum(), (y_tr==0).sum())
    logging.info("  Test:  pep=%s  pos=%d neg=%d",
                 pep_te.shape, (y_te==1).sum(), (y_te==0).sum())
    return pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te


# ── Model (same as Strategy C) ─────────────────────────────────────────────────
class PepHLADataset(Dataset):
    def __init__(self, pep, hla, y):
        self.pep = torch.tensor(pep, dtype=torch.float16)
        self.hla = torch.tensor(hla, dtype=torch.float16)
        self.y   = torch.tensor(y,   dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i):
        return self.pep[i], self.hla[i], self.y[i]


class PeptideHLACrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1, attn_dim=256):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_head    = attn_dim // num_heads
        self.d_model   = d_model
        self.attn_dim  = attn_dim
        self.W_q = nn.Linear(d_model, attn_dim, bias=False)
        self.W_k = nn.Linear(d_model, attn_dim, bias=False)
        self.W_v = nn.Linear(d_model, attn_dim, bias=False)
        self.W_o = nn.Linear(attn_dim, d_model, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale      = math.sqrt(self.d_head)

    def forward(self, peptide, hla):
        B = peptide.size(0)
        def split_heads(x, seq_len):
            return x.view(B, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        Q = split_heads(self.W_q(peptide), MAXSEQ)
        K = split_heads(self.W_k(hla),     HLA_LEN)
        V = split_heads(self.W_v(hla),     HLA_LEN)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(B, MAXSEQ, self.attn_dim)
        out = self.W_o(context)
        out = self.layer_norm(peptide + out)
        return out, weights


class PositionAttention(nn.Module):
    def __init__(self, num_positions):
        super().__init__()
        self.score = nn.Linear(num_positions, num_positions, bias=False)
    def forward(self, x):
        summary = x.mean(dim=-1)
        weights = F.softmax(self.score(summary), dim=-1).unsqueeze(-1)
        return x * weights


class DeepScanCrossAttn(nn.Module):
    def __init__(self, maxseq, emb_dim, window_sizes, num_filters, num_hidden,
                 num_heads=8, attn_dropout=0.1):
        super().__init__()
        self.window_sizes = window_sizes
        self.cross_attn = PeptideHLACrossAttention(
            d_model=emb_dim, num_heads=num_heads,
            dropout=attn_dropout, attn_dim=256)
        self.pos_attn   = PositionAttention(num_positions=maxseq)
        self.cnn_dim    = 256
        self.proj_cnn   = nn.Linear(emb_dim, self.cnn_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(1, ws)) for ws in window_sizes])
        self.pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((1, 1)) for _ in window_sizes])
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(num_filters * len(window_sizes), num_hidden)
        self.fc2 = nn.Linear(num_hidden, NUM_CLASSES)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, peptide, hla):
        x, attn_w = self.cross_attn(peptide, hla)
        x = self.pos_attn(x)
        x = self.proj_cnn(x)
        x = x.unsqueeze(1)
        branch_outs = []
        for conv, pool in zip(self.convs, self.pools):
            h = F.relu(conv(x))
            h = pool(h).view(h.size(0), -1)
            branch_outs.append(h)
        x = torch.cat(branch_outs, dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x), attn_w


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, pep_te, hla_te, y_te):
    model.eval()
    pep_t = torch.tensor(pep_te, dtype=torch.float32)
    hla_t = torch.tensor(hla_te, dtype=torch.float32)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(pep_t), 512):
            pb = pep_t[i:i+512].to(device)
            hb = hla_t[i:i+512].to(device)
            logits, _ = model(pb, hb)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
    pred_prob = np.concatenate(all_probs)
    fpr, tpr, thresholds = roc_curve(y_te, pred_prob)
    auc = metrics.auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]
    y_pred = (pred_prob >= threshold).astype(int)
    TN, FP, FN, TP = metrics.confusion_matrix(y_te, y_pred).ravel()
    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc  = (TP + TN) / (TP + FP + TN + FN)
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    MCC  = (TP*TN - FP*FN) / denom if denom > 0 else 0.0
    F1   = 2*TP / (2*TP + FP + FN) if 2*TP + FP + FN > 0 else 0.0
    return int(TP), int(FP), int(TN), int(FN), Sens, Spec, Acc, MCC, auc, F1, \
           fpr, tpr, thresholds, ix, threshold, pred_prob


# ── Train ─────────────────────────────────────────────────────────────────────
def train_and_eval(pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te, held_out):
    model = DeepScanCrossAttn(
        maxseq=MAXSEQ, emb_dim=EMB_DIM,
        window_sizes=args.window_sizes, num_filters=args.filters,
        num_hidden=args.hidden, num_heads=args.num_heads,
        attn_dropout=args.attn_dropout).to(device)

    n_pos = (y_tr == 1).sum();  n_neg = (y_tr == 0).sum()
    w_pos = n_neg / (n_pos + n_neg);  w_neg = n_pos / (n_pos + n_neg)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler    = torch.amp.GradScaler('cuda')

    dataset = PepHLADataset(pep_tr, hla_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True,
                         persistent_workers=True)
    n_batches = len(loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch_i, (pep_b, hla_b, y_b) in enumerate(loader, 1):
            pep_b = pep_b.to(device, non_blocking=True)
            hla_b = hla_b.to(device, non_blocking=True)
            y_b   = y_b.to(device,   non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(pep_b, hla_b)
                loss = criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * len(y_b)
        epoch_loss /= len(dataset)
        logging.info("  [%s] Epoch %2d/%d  loss=%.4f  time=%.0fs",
                     held_out, epoch, args.epochs, epoch_loss, time.time()-t0)

    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, \
        fpr, tpr, thresholds, ix, threshold, pred_prob = evaluate(
            model, pep_te, hla_te, y_te)

    logging.info("[LOAO %s] AUC=%.4f Sens=%.4f Spec=%.4f Acc=%.4f MCC=%.4f F1=%.4f",
                 held_out, AUC, Sens, Spec, Acc, MCC, F1)

    # Save pkl
    pkl_name = f"loao_{held_out}_{args.plm_name}_{EMB_DIM}dim_{args.filters}f_{args.hidden}h.pkl"
    pkl_path = os.path.join(args.output, pkl_name)
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "plm": args.plm_name, "held_out": held_out,
            "fpr": fpr, "tpr": tpr, "thresholds": thresholds,
            "auc": AUC, "gmean_ix": ix, "threshold": threshold,
            "y_true": y_te, "y_score": pred_prob,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "Sens": Sens, "Spec": Spec, "Acc": Acc, "MCC": MCC, "F1": F1
        }, f)
    logging.info("pkl saved: %s", pkl_path)

    del model, dataset, loader
    gc.collect();  torch.cuda.empty_cache()
    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    logging.info("=== LOAO Evaluation — Strategy C (Cross-Attention) ===")
    hla_embeddings = load_hla_embeddings(args.hla_emb_dir)

    results_path = os.path.join(args.output, f"loao_results_stratC_{args.plm_name}.csv")
    with open(results_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["held_out_allele", "TP", "FP", "TN", "FN",
                         "Sens", "Spec", "Acc", "MCC", "AUC", "F1"])

        for held_out in args.loao_alleles:
            if held_out not in ALLELE_TAGS:
                logging.warning("Unknown allele %s — skipping", held_out)
                continue

            logging.info("\n" + "="*55)
            logging.info("LOAO: holding out  %s", held_out)
            logging.info("="*55)

            pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te = load_loao_dataset(
                args.data_path, hla_embeddings, held_out)

            TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1 = train_and_eval(
                pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te, held_out)

            writer.writerow([held_out, TP, FP, TN, FN,
                             f"{Sens:.4f}", f"{Spec:.4f}", f"{Acc:.4f}",
                             f"{MCC:.4f}", f"{AUC:.4f}", f"{F1:.4f}"])
            csvf.flush()

    logging.info("\nAll LOAO results saved: %s", results_path)
    print(f"\nDone. Results: {results_path}")


if __name__ == "__main__":
    main()
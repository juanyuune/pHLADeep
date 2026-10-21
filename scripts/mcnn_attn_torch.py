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

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",    type=str, required=True)
parser.add_argument("--hla_emb_dir",  type=str,
                    default="/srv/jupyterlab/workspace/HLA1/hla/hla_embeddings_ankh")
parser.add_argument("--output",       type=str, required=True)
parser.add_argument("--mode",         type=str, default="independent",
                    choices=["independent", "cross"])
parser.add_argument("--epochs",       type=int, default=20)
parser.add_argument("--filters",      type=int, default=1024)
parser.add_argument("--hidden",       type=int, default=500)
parser.add_argument("--batch_size",   type=int, default=256)
parser.add_argument("--kfold",        type=int, default=5)
parser.add_argument("--window_sizes", type=int, nargs="+", default=[2, 3, 5, 9])
parser.add_argument("--num_heads",    type=int, default=8)
parser.add_argument("--attn_dropout", type=float, default=0.1)
parser.add_argument("--lr",           type=float, default=1e-3)
parser.add_argument("--num_feature",  type=int, default=1536)
parser.add_argument("--plm_name",     type=str, default="")
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


def load_hla_embeddings(hla_emb_dir):
    hla_embs = {}
    for allele_tag in ALLELE_TAGS:
        matched = [f for f in os.listdir(hla_emb_dir)
                   if f.startswith(allele_tag + ".")]
        if not matched:
            raise FileNotFoundError(
                f"No HLA embedding file found for {allele_tag} in {hla_emb_dir}")
        fname = matched[0]
        fpath = os.path.join(hla_emb_dir, fname)
        with open(fpath, "rb") as f:
            arr = pickle.load(f)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        assert arr.shape[0] == HLA_LEN, \
            f"Bad HLA shape {arr.shape} for {allele_tag}: expected ({HLA_LEN}, {EMB_DIM})"
        hla_embs[allele_tag] = arr
    logging.info("HLA embeddings loaded for: %s", list(hla_embs.keys()))
    return hla_embs


def load_npy_split(npy_path, label, hla_emb):
    arr = np.load(npy_path)          # (N, 1, 9, dim)
    arr = arr[:, 0, :, :]           # (N, 9, dim)
    arr = arr.astype(np.float32)
    N   = arr.shape[0]
    hla_tiled = np.tile(hla_emb[np.newaxis, :, :], (N, 1, 1))  # (N, 34, dim)
    y         = np.full(N, label, dtype=np.int64)
    return arr, hla_tiled, y


def load_dataset(data_path, hla_embeddings):
    peps_tr, hlas_tr, labels_tr = [], [], []
    peps_te, hlas_te, labels_te = [], [], []

    allele_dirs = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d.startswith("HLA-")
    )
    logging.info("Found %d allele dirs: %s", len(allele_dirs), allele_dirs)

    for allele_tag in allele_dirs:
        if allele_tag not in hla_embeddings:
            logging.warning("No HLA embedding for %s — skipping", allele_tag)
            continue

        allele_dir = os.path.join(data_path, allele_tag)
        hla_emb    = hla_embeddings[allele_tag]

        for split_file, label, target in [
            ("pos_train.npy", 1, "tr"),
            ("neg_train.npy", 0, "tr"),
            ("pos_test.npy",  1, "te"),
            ("neg_test.npy",  0, "te"),
        ]:
            fpath = os.path.join(allele_dir, split_file)
            if not os.path.exists(fpath):
                logging.warning("Missing: %s", fpath)
                continue

            p, h, y = load_npy_split(fpath, label, hla_emb)
            logging.info("  %s/%s: pep=%s hla=%s",
                         allele_tag, split_file, p.shape, h.shape)

            if target == "tr":
                peps_tr.append(p); hlas_tr.append(h); labels_tr.append(y)
            else:
                peps_te.append(p); hlas_te.append(h); labels_te.append(y)

    pep_tr = np.concatenate(peps_tr,   axis=0)
    hla_tr = np.concatenate(hlas_tr,   axis=0)
    y_tr   = np.concatenate(labels_tr, axis=0)
    pep_te = np.concatenate(peps_te,   axis=0)
    hla_te = np.concatenate(hlas_te,   axis=0)
    y_te   = np.concatenate(labels_te, axis=0)

    logging.info("Total train: pep=%s hla=%s  labels: pos=%d neg=%d",
                 pep_tr.shape, hla_tr.shape,
                 (y_tr == 1).sum(), (y_tr == 0).sum())
    logging.info("Total test:  pep=%s hla=%s  labels: pos=%d neg=%d",
                 pep_te.shape, hla_te.shape,
                 (y_te == 1).sum(), (y_te == 0).sum())

    return pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te


class PepHLADataset(Dataset):
    def __init__(self, pep, hla, y):
        # float16 to halve CPU→GPU transfer bandwidth
        self.pep = torch.tensor(pep, dtype=torch.float16)
        self.hla = torch.tensor(hla, dtype=torch.float16)
        self.y   = torch.tensor(y,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

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

        # project down to attn_dim to avoid OOM on large PLMs
        self.W_q = nn.Linear(d_model, attn_dim, bias=False)
        self.W_k = nn.Linear(d_model, attn_dim, bias=False)
        self.W_v = nn.Linear(d_model, attn_dim, bias=False)
        self.W_o = nn.Linear(attn_dim, d_model, bias=False)

        self.attn_drop  = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale      = math.sqrt(self.d_head)

    def forward(self, peptide, hla):
        # peptide: (B, 9, D)   hla: (B, 34, D)
        B = peptide.size(0)

        def split_heads(x, seq_len):
            return x.view(B, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.W_q(peptide), MAXSEQ)   # (B, H, 9,  dh)
        K = split_heads(self.W_k(hla),     HLA_LEN)  # (B, H, 34, dh)
        V = split_heads(self.W_v(hla),     HLA_LEN)  # (B, H, 34, dh)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B,H,9,34)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        context = torch.matmul(weights, V)                 # (B, H, 9, dh)
        context = context.transpose(1, 2).contiguous()     # (B, 9, H, dh)
        context = context.view(B, MAXSEQ, self.attn_dim)   # (B, 9, attn_dim)

        out = self.W_o(context)                            # (B, 9, D)
        out = self.layer_norm(peptide + out)               # residual + norm
        return out, weights


class PositionAttention(nn.Module):
    def __init__(self, num_positions=9):
        super().__init__()
        self.score = nn.Linear(num_positions, num_positions, bias=True)

    def forward(self, x):
        summary = x.mean(dim=-1)                          # (B, 9)
        weights = F.softmax(self.score(summary), dim=-1)  # (B, 9)
        weights = weights.unsqueeze(-1)                   # (B, 9, 1)
        return x * weights                                # (B, 9, D)


class DeepScanCrossAttn(nn.Module):
    def __init__(self, maxseq, emb_dim, window_sizes, num_filters, num_hidden,
                 num_heads=8, attn_dropout=0.1):
        super().__init__()
        self.window_sizes = window_sizes
        self.maxseq       = maxseq

        self.cross_attn = PeptideHLACrossAttention(
            d_model=emb_dim, num_heads=num_heads, dropout=attn_dropout,
            attn_dim=256)
        self.pos_attn   = PositionAttention(num_positions=maxseq)

        # project to smaller dim before CNN
        self.cnn_dim  = 256
        self.proj_cnn = nn.Linear(emb_dim, self.cnn_dim)

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for ws in window_sizes:
            self.convs.append(nn.Conv2d(
                in_channels=1, out_channels=num_filters,
                kernel_size=(1, ws), padding=0))
            self.pools.append(nn.AdaptiveMaxPool2d((1, 1)))

        concat_dim = num_filters * len(window_sizes)
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(concat_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, NUM_CLASSES)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, peptide, hla):
        x, attn_w = self.cross_attn(peptide, hla)   # (B, 9, D)
        x = self.pos_attn(x)                         # (B, 9, D)
        x = self.proj_cnn(x)                         # (B, 9, 256)
        x = x.unsqueeze(1)                           # (B, 1, 9, 256)

        branch_outs = []
        for conv, pool in zip(self.convs, self.pools):
            h = F.relu(conv(x))
            h = pool(h)
            h = h.view(h.size(0), -1)
            branch_outs.append(h)

        x = torch.cat(branch_outs, dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, attn_w


def evaluate(model, pep_te, hla_te, y_te):
    model.eval()
    pep_t = torch.tensor(pep_te, dtype=torch.float32)
    hla_t = torch.tensor(hla_te, dtype=torch.float32)
    all_probs = []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(pep_t), bs):
            pb = pep_t[i:i+bs].to(device)
            hb = hla_t[i:i+bs].to(device)
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
    return int(TP), int(FP), int(TN), int(FN), Sens, Spec, Acc, MCC, auc, \
           fpr, tpr, thresholds, ix, threshold, pred_prob


def get_pos_weights(model, pep_te, hla_te):
    model.eval()
    pep_t = torch.tensor(pep_te[:500], dtype=torch.float32).to(device)
    hla_t = torch.tensor(hla_te[:500], dtype=torch.float32).to(device)
    with torch.no_grad():
        x, _ = model.cross_attn(pep_t, hla_t)
        summary = x.mean(dim=-1)
        weights = F.softmax(model.pos_attn.score(summary), dim=-1)
    return weights.cpu().numpy().mean(axis=0)


def train_and_eval(pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te, run_label="",
                   weights_path=None):
    model = DeepScanCrossAttn(
        maxseq=MAXSEQ,
        emb_dim=EMB_DIM,
        window_sizes=args.window_sizes,
        num_filters=args.filters,
        num_hidden=args.hidden,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model built. Trainable params: %s", f"{total_params:,}")

    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    w_pos = n_neg / (n_pos + n_neg)
    w_neg = n_pos / (n_pos + n_neg)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device)
    logging.info("Class weights: neg=%.4f pos=%.4f", w_neg, w_pos)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler    = torch.amp.GradScaler('cuda')

    dataset = PepHLADataset(pep_tr, hla_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True,
                         persistent_workers=True)

    best_loss  = float("inf")
    patience   = 10
    no_improve = 0

    n_batches = len(loader)
    t_epoch_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_batch_start = time.time()
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

            if batch_i % 100 == 0 or batch_i == n_batches:
                elapsed = time.time() - t_batch_start
                speed   = 100 / elapsed if batch_i % 100 == 0 else (batch_i % 100 or 100) / elapsed
                eta_ep  = (n_batches - batch_i) * elapsed / max(batch_i % 100 or 100, 1) / 60
                print(f"  Epoch {epoch}/{args.epochs}  batch {batch_i}/{n_batches}"
                      f"  batch_loss={loss.item():.4f}"
                      f"  {speed:.1f} batch/s  ETA_epoch={eta_ep:.1f}min", flush=True)
                t_batch_start = time.time()

        epoch_loss /= len(dataset)
        epoch_secs = time.time() - t_epoch_start
        remaining_epochs = args.epochs - epoch
        eta_total = epoch_secs * remaining_epochs / 60
        logging.info("Epoch %2d/%d  loss=%.4f  epoch_time=%.0fs  ETA_total=%.0fmin",
                     epoch, args.epochs, epoch_loss, epoch_secs, eta_total)
        t_epoch_start = time.time()

        if epoch_loss < best_loss - 1e-4:
            best_loss  = epoch_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info("Early stopping at epoch %d", epoch)
                break

    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, \
        fpr, tpr, thresholds, ix, threshold, pred_prob = evaluate(
            model, pep_te, hla_te, y_te)

    logging.info(
        "[%s] Sens=%.4f Spec=%.4f Acc=%.4f MCC=%.4f AUC=%.4f",
        run_label, Sens, Spec, Acc, MCC, AUC)

    pos_weights = get_pos_weights(model, pep_te, hla_te)
    logging.info("  Position weights (P1–P9): %s",
                 " ".join(f"{w:.3f}" for w in pos_weights))

    if weights_path is not None:
        torch.save(model.state_dict(), weights_path)
        logging.info("Model weights saved: %s", weights_path)

    del model, dataset, loader
    gc.collect()
    torch.cuda.empty_cache()

    return (TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC,
            fpr, tpr, thresholds, ix, threshold, pred_prob, pos_weights)


def main():
    logging.info("=== PLM + Peptide-HLA Cross-Attention + Position-Attention + MCNN ===")
    logging.info("data_path   : %s", args.data_path)
    logging.info("hla_emb_dir : %s", args.hla_emb_dir)
    logging.info("output      : %s", args.output)
    logging.info("num_feature : %d", EMB_DIM)
    logging.info("plm_name    : %s", args.plm_name)
    logging.info("num_heads   : %d", args.num_heads)

    hla_embeddings = load_hla_embeddings(args.hla_emb_dir)

    logging.info("Loading data ...")
    pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te = load_dataset(
        args.data_path, hla_embeddings)

    label = f"{args.plm_name}_" if args.plm_name else ""
    results_path = os.path.join(args.output, "results_crossattn.csv")
    write_header = not os.path.exists(results_path)

    if args.mode == "independent":
        weights_path = os.path.join(
            args.output,
            f"model_{label}{EMB_DIM}dim_{args.filters}f_{args.hidden}h"
            f"_{args.window_sizes}.pt")
        t0 = time.time()
        (TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC,
         fpr, tpr, thresholds, ix, threshold, pred_prob, pos_w) = train_and_eval(
            pep_tr, hla_tr, y_tr, pep_te, hla_te, y_te,
            run_label=f"StratC_{args.plm_name}",
            weights_path=weights_path)
        elapsed = time.time() - t0

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {AUC:.4f}')
        plt.scatter(fpr[ix], tpr[ix], marker='*', s=200, color='red',
                    label=f'G-Mean threshold={threshold:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve — Strategy C ({args.plm_name})')
        plt.legend()
        roc_png = os.path.join(
            args.output,
            f'roc_curve_{label}{EMB_DIM}dim_{args.filters}f_{args.hidden}h.png')
        plt.savefig(roc_png, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info("ROC curve saved: %s", roc_png)

        roc_data = {
            "plm":        args.plm_name,
            "strategy":   "C",
            "fpr":        fpr,
            "tpr":        tpr,
            "thresholds": thresholds,
            "auc":        AUC,
            "gmean_ix":   ix,
            "threshold":  threshold,
            "y_true":     y_te,
            "y_score":    pred_prob,
        }
        pkl_path = os.path.join(
            args.output,
            f'roc_data_{label}{EMB_DIM}dim_{args.filters}f_{args.hidden}h.pkl')
        with open(pkl_path, "wb") as f:
            pickle.dump(roc_data, f)
        logging.info("ROC data pkl saved: %s", pkl_path)

        with open(results_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp", "mode", "model_type", "plm_name",
                    "window_sizes", "filters", "hidden", "epochs", "num_heads",
                    "TP", "FP", "TN", "FN",
                    "Sens", "Spec", "Acc", "MCC", "AUC",
                    "attn_P1","attn_P2","attn_P3","attn_P4","attn_P5",
                    "attn_P6","attn_P7","attn_P8","attn_P9",
                    "elapsed_s"
                ])
            w.writerow([
                time.ctime(), "independent", "StratC_CrossAttn", args.plm_name,
                str(args.window_sizes), args.filters, args.hidden,
                args.epochs, args.num_heads,
                TP, FP, TN, FN,
                f"{Sens:.4f}", f"{Spec:.4f}", f"{Acc:.4f}",
                f"{MCC:.4f}", f"{AUC:.4f}",
                *[f"{v:.4f}" for v in pos_w],
                f"{elapsed:.1f}"
            ])
        logging.info("Results saved: %s", results_path)

    elif args.mode == "cross":
        from sklearn.model_selection import StratifiedKFold
        kfold        = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=2)
        fold_results = []

        for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(pep_tr, y_tr), 1):
            logging.info("  Cross-val fold %d/%d", fold_i, args.kfold)
            (TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC,
             fpr, tpr, thresholds, ix, threshold, pred_prob, pos_w) = train_and_eval(
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
                    "timestamp", "mode", "model_type", "plm_name",
                    "window_sizes", "filters", "hidden", "epochs", "num_heads",
                    "Sens", "Spec", "Acc", "MCC", "AUC"
                ])
            w.writerow([
                time.ctime(), "cross", "StratC_CrossAttn", args.plm_name,
                str(args.window_sizes), args.filters, args.hidden,
                args.epochs, args.num_heads,
                f"{mean_r[0]:.4f}", f"{mean_r[1]:.4f}", f"{mean_r[2]:.4f}",
                f"{mean_r[3]:.4f}", f"{mean_r[4]:.4f}"
            ])
        logging.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()

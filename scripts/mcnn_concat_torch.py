import os
import gc
import csv
import math
import pickle
import datetime
import time
import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",    type=str, required=True)
parser.add_argument("--output",       type=str, required=True)
parser.add_argument("--strategy",     type=str, default="B", choices=["A","B"])
parser.add_argument("--hla_emb_dir",  type=str,
                    default="/srv/jupyterlab/workspace/HLA1/hla/hla_embeddings_ankh")
parser.add_argument("--plm_name",     type=str, default="ankh")
parser.add_argument("--window_sizes", type=int, nargs="+", default=[2,3,5,9])
parser.add_argument("--filters",      type=int, default=1024)
parser.add_argument("--hidden",       type=int, default=500)
parser.add_argument("--maxseq",       type=int, default=43)
parser.add_argument("--num_feature",  type=int, default=1536)
parser.add_argument("--epochs",       type=int, default=20)
parser.add_argument("--batch_size",   type=int, default=64)
parser.add_argument("--kfold",        type=int, default=5)
parser.add_argument("--mode",         type=str, default="independent",
                    choices=["independent","cross"])
parser.add_argument("--patience",     type=int, default=10)
args = parser.parse_args()

MAXSEQ       = args.maxseq
NUM_FEATURE  = args.num_feature
NUM_FILTER   = args.filters
NUM_HIDDEN   = args.hidden
WINDOW_SIZES = args.window_sizes
BS           = args.batch_size

os.makedirs(args.output, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name}  ({prop.total_memory/1e9:.1f} GB)", flush=True)


class DeepScan(nn.Module):
    def __init__(self, maxseq, num_feature, window_sizes, num_filters, num_hidden):
        super().__init__()
        self.window_sizes = window_sizes

        self.convs = nn.ModuleList([
            nn.Conv1d(num_feature, num_filters, kernel_size=ws, padding=0)
            for ws in window_sizes
        ])
        self.pools = nn.ModuleList([
            nn.AdaptiveMaxPool1d(1) for _ in window_sizes
        ])

        flat_dim = num_filters * len(window_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(flat_dim, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # x: (N, 1, maxseq, num_feature)
        x = x.squeeze(1)                        # (N, maxseq, num_feature)
        x = x.permute(0, 2, 1)                 # (N, num_feature, maxseq)
        outs = []
        for conv, pool in zip(self.convs, self.pools):
            o = torch.relu(conv(x))             # (N, num_filters, maxseq-ws+1)
            o = pool(o)                          # (N, num_filters, 1)
            o = o.squeeze(-1)                    # (N, num_filters)
            outs.append(o)
        x = torch.cat(outs, dim=1)              # (N, num_filters * n_windows)
        return self.classifier(x)               # (N, 2)


def load_data():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_data import MCNN_data_load
    print(f"Loading data (strategy={args.strategy}) ...", flush=True)
    x_tr, y_tr, x_te, y_te = MCNN_data_load(
        args.data_path, mode=args.strategy, hla_dir=args.hla_emb_dir
    )
    # import_data returns one-hot labels → convert to class indices
    y_tr = np.argmax(y_tr, axis=1).astype(np.int64)
    y_te = np.argmax(y_te, axis=1).astype(np.int64)
    print(f"x_train: {x_tr.shape}  x_test: {x_te.shape}", flush=True)
    assert x_tr.shape[2] == MAXSEQ,      f"maxseq mismatch: {x_tr.shape[2]} vs {MAXSEQ}"
    assert x_tr.shape[3] == NUM_FEATURE, f"feature mismatch: {x_tr.shape[3]} vs {NUM_FEATURE}"
    return x_tr, y_tr, x_te, y_te


def get_class_weight(y):
    w = compute_class_weight('balanced', classes=np.array([0,1]), y=y)
    print(f"Class weights: neg={w[0]:.4f}  pos={w[1]:.4f}", flush=True)
    return torch.tensor(w, dtype=torch.float32).to(device)


def train_epoch(model, x, y, optimizer, criterion, batch_size):
    # manual numpy batch loop — same pattern as TF DataGenerator
    model.train()
    N = len(x)
    idx = np.random.permutation(N)
    total_loss = 0.0
    n_batches = math.ceil(N / batch_size)

    for b in range(n_batches):
        bi = idx[b * batch_size : (b+1) * batch_size]
        xb = torch.tensor(x[bi], dtype=torch.float32, device=device)
        yb = torch.tensor(y[bi], dtype=torch.long,    device=device)

        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(bi)

        if (b+1) % 100 == 0 or (b+1) == n_batches:
            print(f"  batch {b+1}/{n_batches}  loss={loss.item():.4f}", flush=True)

    return total_loss / N


def evaluate(model, x_test, y_test_raw, output_dir, plm_name="", save_pkl=False):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(x_test), BS):
            batch = torch.tensor(x_test[i:i+BS], dtype=torch.float32, device=device)
            probs = torch.softmax(model(batch), dim=1)[:,1].cpu().numpy()
            all_probs.append(probs)
    y_score = np.concatenate(all_probs)
    y_true  = y_test_raw

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    ix  = np.argmax(np.sqrt(tpr * (1 - fpr)))
    thr = thresholds[ix]
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    sens = tp/(tp+fn) if tp+fn>0 else 0.0
    spec = tn/(fp+tn) if fp+tn>0 else 0.0
    acc  = (tp+tn)/(tp+fp+tn+fn)
    denom_mcc = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn-fp*fn)/denom_mcc if denom_mcc>0 else 0.0
    f1   = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0

    print(f"AUC={auc:.4f}  Sens={sens:.4f}  Spec={spec:.4f}  Acc={acc:.4f}  "
          f"MCC={mcc:.4f}  F1={f1:.4f}", flush=True)
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}  threshold={thr:.4f}", flush=True)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label = f"{plm_name}_" if plm_name else ""
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
    plt.scatter(fpr[ix], tpr[ix], marker='*', s=200, color='red', label=f'thr={thr:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    png = os.path.join(output_dir,
        f'roc_curve_{label}{NUM_FEATURE}dim_{NUM_FILTER}f_{NUM_HIDDEN}h.png')
    plt.savefig(png, dpi=150, bbox_inches='tight'); plt.close()
    print(f"ROC PNG: {png}", flush=True)

    if save_pkl:
        roc_data = {"plm": plm_name, "fpr": fpr, "tpr": tpr, "thresholds": thresholds,
                    "auc": auc, "gmean_ix": ix, "threshold": thr,
                    "y_true": y_true, "y_score": y_score}
        pkl = os.path.join(output_dir,
            f'roc_data_{label}{NUM_FEATURE}dim_{NUM_FILTER}f_{NUM_HIDDEN}h.pkl')
        with open(pkl, "wb") as f:
            pickle.dump(roc_data, f)
        print(f"ROC pkl: {pkl}", flush=True)

    return tp, fp, tn, fn, sens, spec, acc, mcc, auc


x_train, y_train, x_test, y_test = load_data()
cw        = get_class_weight(y_train)
criterion = nn.CrossEntropyLoss(weight=cw)
start_time = datetime.datetime.now()
write_data = [time.ctime(), args.mode, str(WINDOW_SIZES), NUM_FILTER, NUM_HIDDEN]

if args.mode == "independent":
    model = DeepScan(MAXSEQ, NUM_FEATURE, WINDOW_SIZES, NUM_FILTER, NUM_HIDDEN).to(device)
    opt   = optim.Adam(model.parameters())

    print(f"\nTraining on {device}  |  {len(x_train):,} train  {len(x_test):,} test  "
          f"batch={BS}", flush=True)

    # warm up CUDA kernels to avoid stall on first batch
    if device.type == "cuda":
        print("Warming up CUDA kernels ...", flush=True)
        dummy = torch.zeros(2, 1, MAXSEQ, NUM_FEATURE, device=device)
        with torch.no_grad():
            _ = model(dummy)
        del dummy
        torch.cuda.synchronize()
        print("Warmup done.", flush=True)

    best_loss  = float('inf')
    no_improve = 0
    for epoch in range(1, args.epochs+1):
        t0   = time.time()
        loss = train_epoch(model, x_train, y_train, opt, criterion, BS)
        dt   = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs}  loss={loss:.4f}  ({dt:.1f}s)", flush=True)

        if loss < best_loss - 1e-4:
            best_loss = loss; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    print("\nEvaluating ...", flush=True)
    tp,fp,tn,fn,sens,spec,acc,mcc,auc = evaluate(
        model, x_test, y_test, args.output, plm_name=args.plm_name, save_pkl=True)
    write_data.extend([tp,fp,tn,fn,sens,spec,acc,mcc,auc])

    label = f"{args.plm_name}_" if args.plm_name else ""
    wpath = os.path.join(args.output,
        f"model_{label}{NUM_FEATURE}dim_{NUM_FILTER}f_{NUM_HIDDEN}h_{WINDOW_SIZES}.pt")
    torch.save(model.state_dict(), wpath)
    print(f"Model saved: {wpath}", flush=True)

write_data.append(datetime.datetime.now() - start_time)
with open(os.path.join(args.output, "results_plm_mcnn.csv"), "a", newline="") as f:
    csv.writer(f).writerow(write_data)
print("Done.", flush=True)

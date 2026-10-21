import os
import gc
import csv
import math
import datetime
import pickle
import time
from time import gmtime, strftime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- args ---
parser = argparse.ArgumentParser()
parser.add_argument("--output",       type=str, required=True,  help="output folder for weights and results")
parser.add_argument("--window_sizes", type=int, nargs="+",      default=[8, 16])
parser.add_argument("--filters",      type=int,                 default=1024)
parser.add_argument("--hidden",       type=int,                 default=500)
parser.add_argument("--maxseq",       type=int,                 default=9)
parser.add_argument("--num_feature",  type=int,                 default=1280)
parser.add_argument("--epochs",       type=int,                 default=20)
parser.add_argument("--batch_size",   type=int,                 default=512)
parser.add_argument("--kfold",        type=int,                 default=5)
parser.add_argument("--imbalance",    type=str,                 default=None,
                    choices=[None, "SMOTE", "ADASYN", "RANDOM"])
parser.add_argument("--mode",         type=str,                 default="independent",
                    choices=["independent", "cross"])
parser.add_argument("--data_path",    type=str, required=True,  help="path to expanded_npy/ root folder")
parser.add_argument("--strategy",     type=str, default="A",    choices=["A", "B"],
                    help="A = peptide only (maxseq 9)  |  B = peptide + HLA concat (maxseq 43)")
parser.add_argument("--hla_emb_dir",  type=str,
                    default="/srv/jupyterlab/workspace/HLA1/hla/hla_embeddings",
                    help="HLA .esm2 embedding folder — required for strategy B")
parser.add_argument("--class_weight", action="store_true", default=True,
                    help="Apply inverse-frequency class weights to handle class imbalance (default: True)")
parser.add_argument("--no_class_weight", dest="class_weight", action="store_false",
                    help="Disable class weights")
parser.add_argument("--plm_name",     type=str, default="",
                    help="PLM label for output filenames (e.g. ankh, prot5xl, esm2, prostt5)")

args = parser.parse_args()

MAXSEQ       = args.maxseq
NUM_FEATURE  = args.num_feature
NUM_FILTER   = args.filters
NUM_HIDDEN   = args.hidden
BATCH_SIZE   = args.batch_size
WINDOW_SIZES = args.window_sizes
NUM_CLASSES  = 2
EPOCHS       = args.epochs
K_FOLD       = args.kfold

os.makedirs(args.output, exist_ok=True)

write_data = []
start_time = datetime.datetime.now()
write_data.append(time.ctime())
write_data.append(args.mode)
write_data.append(str(WINDOW_SIZES))
write_data.append(NUM_FILTER)
write_data.append(NUM_HIDDEN)
write_data.append(args.imbalance)
write_data.append(args.class_weight)


def time_log(message):
    print(message, " : ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))


def compute_weights(y_onehot):
    """Compute inverse-frequency class weights from one-hot labels.
    Returns dict {0: w0, 1: w1} for use in model.fit(class_weight=...).
    The minority class receives a higher weight so the loss function
    penalises misclassifying it more — equivalent to balanced resampling
    without modifying the dataset.
    """
    y_flat = np.argmax(y_onehot, axis=1)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_flat
    )
    cw = {0: float(weights[0]), 1: float(weights[1])}
    logging.info("Class weights: neg=%.4f  pos=%.4f  (ratio neg:pos = %.2f:1)",
                 cw[0], cw[1], cw[0] / cw[1])
    return cw


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data       = data
        self.labels     = labels
        self.batch_size = batch_size
        self.indexes    = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        idx          = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data   = np.array([self.data[i]   for i in idx])
        batch_labels = np.array([self.labels[i] for i in idx])
        return batch_data, batch_labels


class DeepScan(Model):
    def __init__(self, input_shape=(1, MAXSEQ, NUM_FEATURE),
                 window_sizes=[8, 16], num_filters=256, num_hidden=500):
        super(DeepScan, self).__init__()
        self.window_sizes = window_sizes
        self.conv2d  = []
        self.maxpool = []
        self.flatten = []
        for ws in self.window_sizes:
            self.conv2d.append(layers.Conv2D(
                filters=num_filters,
                kernel_size=(1, ws),
                activation=tf.nn.relu,
                padding='valid',
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            self.maxpool.append(layers.MaxPooling2D(
                pool_size=(1, MAXSEQ - ws + 1),
                strides=(1, MAXSEQ),
                padding='valid'
            ))
            self.flatten.append(layers.Flatten())
        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(
            num_hidden,
            activation=tf.nn.relu,
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.fc2 = layers.Dense(
            NUM_CLASSES,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, x, training=False):
        _x = []
        for i in range(len(self.window_sizes)):
            x_conv = self.conv2d[i](x)
            x_maxp = self.maxpool[i](x_conv)
            x_flat = self.flatten[i](x_maxp)
            _x.append(x_flat)
        x = layers.Concatenate()(_x)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def handle_imbalance(mode, x_train, y_train):
    if mode is None or mode == "None":
        return x_train, y_train

    x_2d = x_train.reshape(x_train.shape[0], -1)

    # Convert one-hot back to 1D for SMOTE
    y_1d = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train

    if mode == "SMOTE":
        sampler = SMOTE(random_state=42)
    elif mode == "ADASYN":
        sampler = ADASYN(random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)

    x_res, y_res = sampler.fit_resample(x_2d, y_1d)
    x_res = x_res.reshape(x_res.shape[0], 1, MAXSEQ, NUM_FEATURE)

    del x_2d
    gc.collect()

    y_res = tf.keras.utils.to_categorical(y_res, NUM_CLASSES)
    return x_res, y_res


def model_test(model, x_test, y_test, output_dir, plm_name="", save_pkl=False):
    print(x_test.shape)
    pred = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], pred[:, 1])
    auc = metrics.auc(fpr, tpr)

    gmeans    = np.sqrt(tpr * (1 - fpr))
    ix        = np.argmax(gmeans)
    threshold = thresholds[ix]
    print(f'Best Threshold={threshold}, G-Mean={gmeans[ix]}')

    y_pred = (pred[:, 1] >= threshold).astype(int)
    TN, FP, FN, TP = metrics.confusion_matrix(y_test[:, 1], y_pred).ravel()

    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc  = (TP + TN) / (TP + FP + TN + FN)
    MCC  = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) \
           if (TP + FP) > 0 and (FP + TN) > 0 and (TP + FN) > 0 and (TN + FN) > 0 else 0.0
    F1     = 2 * TP / (2 * TP + FP + FN)
    Prec   = TP / (TP + FP) if TP + FP > 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, '
          f'Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={auc:.4f}, '
          f'F1={F1:.4f}, Prec={Prec:.4f}, Recall={Recall:.4f}\n')

    label = f"{plm_name}_" if plm_name else ""

    # --- save ROC curve PNG ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.scatter(fpr[ix], tpr[ix], marker='*', s=200, color='red', label=f'G-Mean threshold={threshold:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_path = os.path.join(output_dir, f'roc_curve_{label}{NUM_FEATURE}dim_{NUM_FILTER}f_{NUM_HIDDEN}h.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'ROC curve saved: {roc_path}')

    # --- save ROC data pkl (independent mode only) ---
    if save_pkl:
        roc_data = {
            "plm":       plm_name,
            "fpr":       fpr,
            "tpr":       tpr,
            "thresholds": thresholds,
            "auc":       auc,
            "gmean_ix":  ix,
            "threshold": threshold,
            "y_true":    y_test[:, 1],
            "y_score":   pred[:, 1],
        }
        pkl_path = os.path.join(output_dir, f'roc_data_{label}{NUM_FEATURE}dim_{NUM_FILTER}f_{NUM_HIDDEN}h.pkl')
        with open(pkl_path, "wb") as f:
            pickle.dump(roc_data, f)
        print(f'ROC data pkl saved: {pkl_path}')

    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, auc


# --- load data ---
logging.info("Loading data...")
from import_data import MCNN_data_load
x_train, y_train, x_test, y_test = MCNN_data_load(
    args.data_path,
    mode=args.strategy,
    hla_dir=args.hla_emb_dir
)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# --- training ---
if args.mode == "cross":
    time_log("Start cross-validation")
    kfold   = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=2)
    results = []
    y_flat  = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
    i = 1
    for train_idx, test_idx in kfold.split(x_train, y_flat):
        print(i, "/", K_FOLD, '\n')
        X_train, X_test = x_train[train_idx], x_train[test_idx]
        Y_train, Y_test = y_train[train_idx], y_train[test_idx]
        print(X_train.shape, X_test.shape)

        X_train, Y_train = handle_imbalance(args.imbalance, X_train, Y_train)
        generator = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE)

        # Class weights — computed after any resampling
        cw = compute_weights(Y_train) if args.class_weight else None
        if cw:
            logging.info("Fold %d class weights: %s", i, cw)

        model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN, window_sizes=WINDOW_SIZES)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.build(input_shape=(None, 1, MAXSEQ, NUM_FEATURE))
        model.fit(
            generator,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
            class_weight=cw,
            verbose=1,
            shuffle=True
        )

        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(
            model, X_test, Y_test, args.output, plm_name=args.plm_name, save_pkl=False
        )
        results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
        i += 1

        del X_train, X_test, Y_train, Y_test
        gc.collect()

    mean_r = np.mean(results, axis=0)
    print(f'Sens={mean_r[4]:.4f}, Spec={mean_r[5]:.4f}, Acc={mean_r[6]:.4f}, '
          f'MCC={mean_r[7]:.4f}, AUC={mean_r[8]:.4f}')
    write_data.extend(mean_r)


if args.mode == "independent":
    x_train, y_train = handle_imbalance(args.imbalance, x_train, y_train)
    generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)

    # Class weights — computed after any resampling
    cw = compute_weights(y_train) if args.class_weight else None
    if cw:
        logging.info("Class weights applied: %s", cw)

    time_log("Start training")
    model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN, window_sizes=WINDOW_SIZES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, 1, MAXSEQ, NUM_FEATURE))
    model.summary()
    model.fit(
        generator,
        epochs=EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
        class_weight=cw,
        shuffle=True
    )
    time_log("End training")

    time_log("Start evaluation")
    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(
        model, x_test, y_test, args.output, plm_name=args.plm_name, save_pkl=True
    )
    write_data.extend([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
    time_log("End evaluation")


# --- save results ---
def save_csv(write_data, start, output_dir):
    end   = datetime.datetime.now()
    write_data.append(end - start)
    fpath = os.path.join(output_dir, "results_plm_mcnn.csv")
    with open(fpath, "a", newline="") as f:
        csv.writer(f).writerow(write_data)
    print(f"Results saved: {fpath}")

save_csv(write_data, start_time, args.output)


# --- save model weights ---
if args.mode == "independent":
    label = f"{args.plm_name}_" if args.plm_name else ""
    weights_path = os.path.join(
        args.output,
        f"model_{label}{NUM_FEATURE}dim_{NUM_FILTER}f_{NUM_HIDDEN}h_{WINDOW_SIZES}.weights.h5"
    )
    model.save_weights(weights_path)
    print(f"Model saved: {weights_path}")
# pHLADeep

**pHLADeep: Protein Language Model-Based HLA Class I Binding Prediction with External Validation on MS-Confirmed Cancer Neoepitopes**

Juan Peter Timothy Yuune<sup>1</sup>, Van The Le<sup>2</sup>, Dinh-Quy Nguyen<sup>1</sup>, Duc-Loc Nguyen<sup>1</sup>, Yu-Yen Ou<sup>1,2</sup>*

<sup>1</sup> Department of Computer Science and Engineering, Yuan Ze University, Chung-Li 32003, Taiwan  
<sup>2</sup> Graduate Program in Biomedical Informatics, Yuan Ze University, Chung-Li 32003, Taiwan  
*Corresponding author: yien@saturn.yzu.edu.tw

---

## Overview

pHLADeep is a deep learning framework for HLA class I peptide–MHC binding prediction. It uses per-residue embeddings from **Ankh-Large** — a protein language model (PLM) pre-trained on ~250 million protein sequences — as input to a multi-scale convolutional neural network (MCNN). Four PLMs and three allele-conditioning strategies were systematically evaluated; **pHLADeep-Concat** was identified as the best-performing configuration.

---

## Model Variants

| Variant | Input | Strategy |
|---|---|---|
| pHLADeep-Pep | Peptide embedding only (9 × 1536) | Peptide-only |
| **pHLADeep-Concat** | Peptide + HLA pseudo-sequence concatenated (43 × 1536) | **Best** |
| pHLADeep-Attn | Peptide–HLA cross-attention (9 × 1536) | Cross-attention |

All three variants share the same MCNN backbone: four parallel Conv1d branches with kernel widths {2, 3, 5, 9}, 1,024 filters each, global adaptive max-pooling, 4,096-dim concat → Dropout(0.7) → FC(500) → FC(2, softmax).

---

## Results

### Held-out benchmark (11 alleles, 14,948 pairs)

| Method | AUC | Sensitivity | Specificity | MCC |
|---|---|---|---|---|
| **pHLADeep-Concat** | **0.9585** | **0.8848** | **0.8971** | **0.7351** |
| MHCflurry 2.0 | 0.9413 | — | — | 0.7145 |
| pHLADeep-Attn | 0.9381 | 0.8686 | 0.8708 | 0.6922 |
| NetMHCpan 4.2 | 0.9284 | — | — | 0.6712 |
| pHLADeep-Pep | 0.9152 | 0.8486 | 0.8623 | 0.6590 |

### External validation — MS-confirmed cancer neoepitopes (Gurung et al., 2023, *Nature Biotechnology*)

| Method | AUC |
|---|---|
| MHCflurry 2.0 | 0.9938 |
| NetMHCpan 4.2 | 0.9875 (9-mers only) |
| **pHLADeep-Concat** | **0.9070** |
| pHLADeep-Attn | 0.8947 |
| pHLADeep-Pep | 0.7272 |

pHLADeep-Concat achieves AUC = 0.9070 without any HLA-specific pre-training or fine-tuning, trained on 11 alleles only. On 9-mer neoepitopes specifically it reaches AUC = 0.9760, within 1.15 percentage points of MHCflurry 2.0 (0.9837).

---

## Dataset

- **Source:** IEDB binding assay data + EpiScan functional screen (Bruno et al., *Nature Biotechnology*, 2023)
- **Size:** 78,693 peptide–HLA pairs across 11 alleles
- **Split:** 80:20 stratified per allele (train: 63,745 / test: 14,948)
- **Class imbalance:** corrected via inverse-frequency loss weighting (no undersampling or SMOTE)

**HLA alleles covered:**

| Allele | Supertype | Total pairs |
|---|---|---|
| HLA-A\*01:01 | A1 | 9,325 |
| HLA-A\*02:01 | A2 | 11,345 |
| HLA-A\*03:01 | A3 | 5,306 |
| HLA-A\*11:01 | A3 | 9,623 |
| HLA-A\*24:02 | A24 | 11,553 |
| HLA-B\*07:02 | B7 | 16,771 |
| HLA-B\*08:01 | B8 | 2,262 |
| HLA-B\*35:01 | B35 | 10,387 |
| HLA-B\*40:01† | B44 | 363 |
| HLA-B\*51:01† | B51 | 660 |
| HLA-B\*57:01 | B57 | 2,098 |

†Low-data alleles (<1,000 examples); retained for training but excluded from leave-one-allele-out evaluation.

---

## Repository Structure

```
pHLADeep/
├── get_emb_ankh.py              # Generate Ankh-Large embeddings for peptides
├── get_emb_hla_pseudo_ankh.py   # Generate Ankh-Large embeddings for HLA pseudo-sequences (34 × 1536)
├── import_data.py               # Data loader — Strategy A (peptide-only) and B (Concat)
├── import_data_hla.py           # Data loader — per-sample correct allele fusion (ESM-2)
├── mcnn.py                      # Train pHLADeep-Pep (TensorFlow/Keras)
├── mcnn_concat_torch.py         # Train pHLADeep-Concat (PyTorch)
├── mcnn_attn_torch.py           # Train pHLADeep-Attn — cross-attention + MCNN (PyTorch)
├── benchmark_netmhc.py          # Benchmark vs. NetMHCpan 4.2 and MHCflurry 2.0
├── case_study_ms.py             # External validation on Gurung et al. 2023 neoepitopes
└── compute_stratified_auc.py    # Length- and allele-stratified AUC (Tables 10b–10f)
```

---

## Requirements


```
torch>=2.0
tensorflow>=2.10
ankh
numpy
pandas
h5py
scikit-learn
imbalanced-learn
matplotlib
openpyxl
```

NetMHCpan 4.2 and MHCflurry 2.0 must be installed separately for benchmarking.

---

## Usage

### Step 1 — Generate HLA pseudo-sequence embeddings

```bash
python get_emb_hla_pseudo_ankh.py \
    --csv  /path/to/hla_pseudosequences.csv \
    --out  /path/to/hla_embeddings_ankh
```

Output: one `.npy` file per allele, shape `(34, 1536)`.

### Step 2 — Generate peptide embeddings

```bash
python get_emb_ankh.py \
    --fasta /path/to/peptides.fasta \
    --out   /path/to/peptide_embeddings_ankh
```

### Step 3 — Train pHLADeep-Concat (recommended)

```bash
python mcnn_concat_torch.py \
    --data_path   /path/to/embeddings \
    --hla_emb_dir /path/to/hla_embeddings_ankh \
    --output      /path/to/results \
    --plm_name    ankh \
    --num_feature 1536 \
    --epochs      20 \
    --filters     1024 \
    --hidden      500
```

### Step 4 — Train pHLADeep-Attn

```bash
python mcnn_attn_torch.py \
    --data_path   /path/to/embeddings \
    --hla_emb_dir /path/to/hla_embeddings_ankh \
    --output      /path/to/results \
    --mode        independent \
    --plm_name    ankh \
    --num_feature 1536 \
    --window_sizes 2 3 5 9 \
    --epochs      20 \
    --filters     1024 \
    --hidden      500
```

### Step 5 — Benchmark

```bash
python benchmark_netmhc.py
```

### Step 6 — External validation

```bash
python case_study_ms.py
```

---

## Citation

If you use pHLADeep in your work, please cite:

> Yuune JPT, Le VT, Nguyen DQ, Nguyen DL, Ou YY. pHLADeep: Protein Language Model-Based HLA Class I Binding Prediction with External Validation on MS-Confirmed Cancer Neoepitopes. *[journal, year, doi]*.

---

## Acknowledgements

This work was supported by the National Science and Technology Council, Taiwan, under Grant No. NSTC 112-2221-E-155-020-MY3, NSTC 114-2221-E-155-028-MY3, and NSTC 115-2811-E-155-003.

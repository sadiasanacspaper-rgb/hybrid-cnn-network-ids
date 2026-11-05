# Hybrid CNN Network Intrusion Detection System

[![License: MIT](https://img.shields.io/github/license/sadiasanacspaper-rgb/hybrid-cnn-network-ids)](./LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/sadiasanacspaper-rgb/hybrid-cnn-network-ids)](https://github.com/sadiasanacspaper-rgb/hybrid-cnn-network-ids/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/sadiasanacspaper-rgb/hybrid-cnn-network-ids)](https://github.com/sadiasanacspaper-rgb/hybrid-cnn-network-ids)
[![Platform](https://img.shields.io/badge/platform-Google%20Colab-orange)](https://colab.research.google.com/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A research-grade, **single Google Colab cell** that builds, trains and evaluates a hybrid
network intrusion detection system on the combined **CIC-IDS2017** and **CSE-CIC-IDS2018**
datasets, maps detections to **MITRE ATT&CK**, and serves everything through an interactive
Gradio "SOC" dashboard.

> Datasets are fetched automatically from public Hugging Face Hub mirrors — no manual
> upload, no Kaggle key required. If every source fails, a clearly-labelled synthetic
> fallback keeps the pipeline runnable for a demo.

## Architecture

The **1D-CNN is a feature extractor, not the classifier**. Both datasets are harmonized to
a shared ~59-feature numeric space, scaled, and fed through:

```
Conv1D(64) → BatchNorm → ReLU → MaxPool
Conv1D(128) → BatchNorm → ReLU
Conv1D(128) → BatchNorm → ReLU
GlobalAveragePooling1D
Dense(64, "embedding")   <-- the feature vector used below
Dropout → Dense(1, sigmoid)   (training head only)
```

The 64-D embedding is then handed to three independent classical heads, which are the
models actually compared:

| Model | Head |
|---|---|
| **CNN + SVM** | RBF `SVC` (full train set, or a 30k stratified subsample when train is large) |
| **CNN + Decision Tree** | `DecisionTreeClassifier`, `GridSearchCV`-tuned |
| **CNN + Random Forest** | `RandomForestClassifier`, `RandomizedSearchCV`-tuned |

A plain CNN-softmax row is also reported as a reference baseline, but it is **not** one of
the three compared hybrids.

## What the notebook does

1. **Acquisition** — downloads the canonical CIC-IDS2017 (`c01dsnap/CIC-IDS2017`) and
   CSE-CIC-IDS2018 (`pcy12345BSU/CSE-CIC-IDS-2018`) CSVs from Hugging Face Hub, with
   fallback mirrors and a synthetic generator as a last resort.
2. **Memory-safe loading** — streams every CSV in chunks, drops rows of already-full
   classes before the costly numeric work, keeps a per-class stratified cap (minority
   attack classes are preserved, `BENIGN` is capped), and caches the harmonized result to
   Parquet so re-runs skip loading entirely.
3. **Harmonization** — normalizes 2017/2018 CICFlowMeter column names onto one schema,
   drops IP/port/timestamp identifier columns, and maps raw labels to attack families.
4. **MITRE ATT&CK mapping** — each attack family is mapped to a real ATT&CK Enterprise
   tactic/technique/ID with an explicit confidence level; anything unrecognized is stored
   as `"Not confidently mapped"` rather than guessed.
5. **Preprocessing** — median imputation + `StandardScaler`, fit on the training split only.
6. **Feature analysis** — mutual information, Random Forest importance, correlation, PCA.
7. **CNN training** — class-weighted, early-stopped; embeddings extracted for train/val/test.
8. **Three hybrids trained & tuned** on the embeddings.
9. **Evaluation** — accuracy, precision, recall, F1, specificity, FPR/FNR, ROC-AUC, PR-AUC,
   MCC, balanced accuracy, confusion matrices, per-dataset generalization, per-attack
   detection rate, and a dedicated **accuracy study** (train/val/test/CV accuracy,
   accuracy-vs-complexity, and a raw-features-vs-CNN-embedding representation comparison).
10. **Leaderboard** ranked by F1 → Recall → ROC-AUC → low FPR (not plain accuracy).
11. **Artifacts saved** to `IDS_Results/`: ~18 figures, CSVs, the fitted preprocessor, the
    CNN feature extractor, all three hybrid models, and a plain-text research report.
12. **Interactive dashboard** (Gradio): Executive overview, Dataset Intelligence, Model
    Comparison, Accuracy Study, MITRE ATT&CK explorer, Threat Analysis, and a Live/Manual
    flow-prediction tab.

## Usage

1. Open a new Google Colab notebook (GPU runtime recommended but not required).
2. Paste the **entire contents** of [`ids_hybrid_colab.py`](./ids_hybrid_colab.py) into one cell.
3. Run it. First run downloads and caches the datasets; subsequent runs reuse the cache.
4. A public Gradio share link is printed at the end.

### Configuration

All tunables live at the top of the file:

```python
USE_SAMPLING = True          # Colab-friendly stratified subsample vs. full dataset
MAX_ROWS_PER_DATASET = 320_000
MAX_ROWS_PER_CLASS   = 25_000
BENIGN_CAP           = 90_000
CNN_EPOCHS = 18
LAUNCH_DASHBOARD = True
GRADIO_SHARE = True
```

## Research integrity

All reported numbers are computed at runtime from the actual experiment — nothing is
hard-coded. If the datasets cannot be downloaded, the pipeline prints the real error and
falls back to a synthetic dataset *that is labelled as such* in the console output, report
and dashboard, rather than silently presenting fabricated results as real ones.

## Known limitations

- Flow-level CICFlowMeter features only (no raw packet payload).
- Binary detection head; per-attack-family performance is reported as recall, not a full
  multiclass classifier.
- Under `USE_SAMPLING=True`, class prevalence is capped rather than exactly preserved —
  set `USE_SAMPLING=False` for a full-scale run if you have the RAM.
- CIC-IDS2017 and CSE-CIC-IDS2018 share the same capture toolchain/testbed, so
  cross-dataset generalization results are optimistic relative to a genuinely unseen
  production network.
- MITRE ATT&CK mappings are interpretations of dataset labels, not verified adversary
  telemetry; the `Infiltration` mapping is explicitly marked low-confidence.
- No quantum component. The "representation" chart in the Accuracy Study tab compares raw
  scaled features against the CNN embedding and is structured as a drop-in slot for a
  future quantum/QFT-enhanced feature map, if one is added.

## License

MIT — see [LICENSE](./LICENSE).

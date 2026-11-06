# Marathi Hate & Offensive Speech Dataset Preparation & Model Benchmarking

> **TE Major Project** — Supervised Data Pipeline, Baseline SVM, and Transformer Fine-Tuning for Marathi YouTube Comment Moderation.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset Specifications](#dataset-specifications)
  - [Preprocessing & Filtering](#preprocessing--filtering)
  - [Class Schema & Distribution](#class-schema--distribution)
  - [Train / Validation / Test Splits](#train--validation--test-splits)
- [Models & Training Pipelines](#models--training-pipelines)
  - [1. Baseline: TF-IDF + Linear SVM](#1-baseline-tf-idf--linear-svm)
  - [2. Deep Learning: MahaHate Multi-RoBERTa](#2-deep-learning-mahahate-multi-roberta)
- [Benchmark Results](#benchmark-results)
  - [Overall Performance](#overall-performance)
  - [Class-Wise Classification Breakdown](#class-wise-classification-breakdown)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
  - [Running the SVM Baseline](#running-the-svm-baseline)
  - [Fine-Tuning MahaHate RoBERTa](#fine-tuning-mahahate-roberta)
  - [Compare Saved Predictions & Inspect Errors](#compare-saved-predictions--inspect-errors)
  - [Quick Inference Snippet](#quick-inference-snippet)
- [Academic Notes & Limitations](#academic-notes--limitations)

---

## 📖 Overview

This repository houses the supervised data preparation pipeline and experimental benchmarks for classifying Marathi YouTube comments into four moderation categories: **Non-offensive (`NOT`)**, **Offensive (`OFFN`)**, **Profane (`PRFN`)**, and **Hate Speech (`HATE`)**.

Key deliverables in this directory include:
- **Cleaned & Stratified Datasets**: Curated splits (70% train, 15% validation, 15% test) derived from 30,000 raw annotated comments.
- **Auditing Files**: Separated blank/unlabelled rows and ambiguous entries flagged for human review.
- **Traditional ML Baseline**: Word and character n-gram TF-IDF pipeline paired with a class-balanced Linear Support Vector Classifier (LinearSVC).
- **Fine-Tuned Transformer**: Domain-specific fine-tuning of `l3cube-pune/mahahate-multi-roberta` achieving a **76.98% Test Macro-F1**.
- **Complete Evaluation Artifacts**: Confusion matrices, detailed per-class JSON/text classification reports, and test prediction CSVs with error analysis flags.

---

## 📁 Repository Structure

```text
TE_training_data_prep/
├── README.md                                      # Project documentation (this file)
├── README.txt                                     # Summary notes from the initial dataset preparation
├── requirements.txt                              # Lightweight baseline and analysis dependencies
├── requirements-roberta.txt                      # Optional transformer training dependencies
├── compare_predictions.py                        # Aggregate SVM/RoBERTa comparison and error analysis
├── dataset_split_summary.json                     # JSON record of split sizes, percentages, and class distributions
│
├── dataset/                                       # Official stratified experimental splits
│   ├── train.csv                                  # Training set (70%, 7,189 rows)
│   ├── validation.csv                             # Validation set (15%, 1,541 rows)
│   └── test.csv                                   # Final held-out test set (15%, 1,541 rows)
│
├── labelled_comments_clean.csv                    # Complete set of valid, clean labelled comments (10,271 rows)
├── ambiguous_for_review.csv                       # Flagged "AMBIGUOUS" rows isolated for review (3 rows)
├── unlabelled_comments_not_used_for_training.csv  # Unlabelled or blank rows excluded from training (19,725 rows)
│
├── train_svm_baseline.py                          # Training script for TF-IDF + LinearSVC baseline
├── train_mahahate_roberta.py                      # Training script for fine-tuning MahaHate Multi-RoBERTa
│
├── svm_baseline_results/                          # Artifacts & metrics for SVM baseline
│   ├── metrics_summary.json                       # Hyperparameters, test accuracy & macro-F1
│   ├── test_classification_report.json            # Machine-readable per-class report
│   ├── test_classification_report.txt             # Human-readable per-class report
│   ├── test_confusion_matrix.csv                  # 4x4 confusion matrix
│   ├── test_predictions.csv                       # Predictions with correct_prediction flag
│   ├── tfidf_linear_svm.joblib                    # Serialized best trained pipeline
│   └── validation_model_selection.csv             # Validation performance across C values
│
└── mahahate_roberta_results/                      # Artifacts & metrics for fine-tuned Transformer
    ├── best_model/                                # Saved Hugging Face model & tokenizer
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    ├── metrics_summary.json                       # Fine-tuning parameters and final test scores
    ├── test_classification_report.json            # Machine-readable per-class report
    ├── test_classification_report.txt             # Human-readable per-class report
    ├── test_confusion_matrix.csv                  # 4x4 confusion matrix
    ├── test_predictions.csv                       # Prediction log for error auditing
    └── training_history.csv                       # Per-epoch train/val loss, accuracy, and macro-F1
```

---

## 📊 Dataset Specifications

### Preprocessing & Filtering
- **Source**: Initial extraction of 30,000 raw comments (`Baseline_file.xlsx`).
- **Cleaning Filters**:
  - Excluded 1 row with empty/blank comment text.
  - Excluded 19,725 unlabelled rows (saved separately in `unlabelled_comments_not_used_for_training.csv`).
  - Excluded 3 rows labelled as `AMBIGUOUS` (saved separately in `ambiguous_for_review.csv`).
- **Usable Dataset**: **10,271** verified labelled instances saved in `labelled_comments_clean.csv`.

### Class Schema & Distribution
Comments are categorized into 4 mutually exclusive classes:
- **`NOT`**: Non-offensive, normal conversation, positive or neutral feedback.
- **`OFFN`**: Offensive, insulting, derogatory, or rude language not directed systematically at protected traits.
- **`PRFN`**: Profane or vulgar language, slang, or swear words without targeted hate.
- **`HATE`**: Hate speech targeting social identity, religion, caste, gender, ethnicity, or community.

| Category | Description | Total Count | Proportion |
| :--- | :--- | :---: | :---: |
| **`NOT`** | Non-Offensive | 6,975 | 67.91% |
| **`OFFN`** | Offensive | 1,854 | 18.05% |
| **`HATE`** | Hate Speech | 1,090 | 10.61% |
| **`PRFN`** | Profane | 352 | 3.43% |
| **Total** | | **10,271** | **100.0%** |

### Train / Validation / Test Splits
Data was split using **Stratified Random Sampling** (`random_state=42`) with a **70% / 15% / 15%** ratio to preserve class proportions:

| Class | Train Set (70%) | Validation Set (15%) | Test Set (15%) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **`NOT`** | 4,882 | 1,047 | 1,046 | 6,975 |
| **`OFFN`** | 1,298 | 278 | 278 | 1,854 |
| **`HATE`** | 763 | 163 | 164 | 1,090 |
| **`PRFN`** | 246 | 53 | 53 | 352 |
| **Total Rows** | **7,189** | **1,541** | **1,541** | **10,271** |

---

## 🤖 Models & Training Pipelines

### 1. Baseline: TF-IDF + Linear SVM
- **Script**: `train_svm_baseline.py`
- **Feature Representation**: `FeatureUnion` combining:
  - **Word n-grams**: $(1, 2)$, sublinear TF, minimum document frequency $2$, max $250,000$ features.
  - **Character n-grams**: $(2, 5)$, sublinear TF, minimum document frequency $2$, max $300,000$ features (capturing morphological variations and Devanagari affixes).
- **Classifier**: `LinearSVC` with `class_weight='balanced'` and `random_state=42`.
- **Hyperparameter Search**: Evaluated $C \in \{0.5, 1.0, 2.0\}$ against validation Macro-F1. $C=2.0$ achieved the highest validation Macro-F1 ($0.7014$).

### 2. Deep Learning: MahaHate Multi-RoBERTa
- **Script**: `train_mahahate_roberta.py`
- **Base Checkpoint**: [`l3cube-pune/mahahate-multi-roberta`](https://huggingface.co/l3cube-pune/mahahate-multi-roberta)
- **Architecture**: RoBERTa encoder pre-trained on Marathi/Indic text with a 4-class sequence classification head matching our label order (`HATE: 0`, `OFFN: 1`, `PRFN: 2`, `NOT: 3`).
- **Optimization & Hyperparameters**:
  - Max sequence length: $128$ tokens
  - Batch size: $2$ per device $\times$ Gradient accumulation steps $8$ (effective batch size = $16$)
  - Optimizer: `AdamW` ($\text{lr}=2\times 10^{-5}$, weight decay $0.01$)
  - Scheduler: Linear warmup over $10\%$ of total steps followed by linear decay
  - Mixed Precision: PyTorch AMP (`fp16`) with gradient scaling
  - Model selection: Monitored validation Macro-F1 per epoch with early stopping patience of $2$
  - Best model reached at **Epoch 4**.

---

## 📈 Benchmark Results

### Overall Performance (Held-Out Test Set)

> [!IMPORTANT]
> The test split (`dataset/test.csv`, $1,541$ rows) was strictly held out and evaluated only once per model following validation model selection.

| Model / Approach | Selected Epoch / C | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LinearSVC Baseline** | $C = 2.0$ | 0.7014 | 83.97% | 66.73% | 83.41% |
| **MahaHate Multi-RoBERTa** | Epoch 4 | **0.7459** | **86.96%** | **76.98%** | **86.91%** |

*MahaHate Multi-RoBERTa outperforms the linear baseline by **+10.25 percentage points** in Test Macro-F1 and **+2.99%** in Accuracy.*

### Class-Wise Classification Breakdown

#### Baseline Linear SVM:
```text
              precision    recall  f1-score   support
         NOT       0.86      0.96      0.91      1046
        PRFN       0.86      0.34      0.49        53
        OFFN       0.82      0.74      0.78       278
        HATE       0.63      0.41      0.49       164

    accuracy                           0.84      1541
   macro avg       0.79      0.61      0.67      1541
weighted avg       0.83      0.84      0.83      1541
```

#### Fine-Tuned MahaHate Multi-RoBERTa:
```text
              precision    recall  f1-score   support
        HATE       0.71      0.64      0.68       164
        OFFN       0.81      0.79      0.80       278
        PRFN       0.91      0.55      0.68        53
         NOT       0.90      0.94      0.92      1046

    accuracy                           0.87      1541
   macro avg       0.83      0.73      0.77      1541
weighted avg       0.87      0.87      0.87      1541
```

### Key Analytical Takeaways
1. **Significant Gains in Minority Classes**:
   - `HATE` class F1 improved from **$0.49 \to 0.68$** (+19 points), with recall jumping from $0.41 \to 0.64$.
   - `PRFN` class F1 improved from **$0.49 \to 0.68$** (+19 points), with recall jumping from $0.34 \to 0.55$.
2. **High Precision for Flagged Content**:
   - MahaHate RoBERTa achieves $91\%$ precision on `PRFN`, $81\%$ on `OFFN`, and $71\%$ on `HATE`, drastically reducing false positives compared to simpler keyword/n-gram heuristics.

---

## 🛠 Installation & Setup

Use Python 3.9+ and create a virtual environment before installing packages.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The lightweight dependencies are enough for the SVM baseline and prediction comparison. For RoBERTa fine-tuning, install the optional dependencies:

```powershell
pip install -r requirements-roberta.txt
```

The optional file includes PyTorch. If you need a specific CUDA build, install the matching PyTorch build for your machine first, then install the remaining packages. CPU training is supported but can take substantially longer.

---

## 🚀 Usage Guide

### Running the SVM Baseline
To run model selection and evaluate on the test split:
```powershell
python train_svm_baseline.py --data-dir dataset --outdir svm_baseline_results --c-values "0.5,1.0,2.0"
```

Available arguments:
- `--data-dir`: Directory containing `train.csv`, `validation.csv`, and `test.csv` (default: `dataset` or `TE_training_data_prep`).
- `--train`, `--validation`, `--test`: Explicit individual paths to CSV splits.
- `--outdir`: Directory to store models, predictions, and reports (default: `svm_baseline_results`).
- `--c-values`: Comma-separated float regularization strengths.

### Fine-Tuning MahaHate RoBERTa
To train and evaluate the transformer model:
```powershell
python train_mahahate_roberta.py `
  --data-dir ./dataset `
  --output-dir ./mahahate_roberta_results `
  --epochs 4 `
  --batch-size 2 `
  --grad-accum 8 `
  --max-length 128 `
  --lr 2e-5 `
  --patience 2
```

Outputs are automatically saved to `mahahate_roberta_results/`, including the serialized best checkpoint under `best_model/`.

### Compare Saved Predictions & Inspect Errors

After running both model scripts on the same ordered `dataset/test.csv`, compare their saved predictions:

```powershell
python compare_predictions.py `
  --svm svm_baseline_results/test_predictions.csv `
  --roberta mahahate_roberta_results/test_predictions.csv `
  --json-out comparison_results/summary.json
```

The command checks that both files contain the same test comments and actual labels in the same order. It reports accuracy, macro-F1, per-class precision/recall/F1, common misclassification pairs, and which model alone got each row correct. Optional JSON output contains aggregate metrics only; it does not include comment text. Results describe this labelled test split and should not be treated as deployment performance.

### Quick Inference Snippet

#### Loading the Trained SVM Pipeline
```python
import joblib

model = joblib.load("svm_baseline_results/tfidf_linear_svm.joblib")
comments = ["हा व्हिडिओ खूप चांगला आहे", "तू मूर्ख आहेस का?"]
predictions = model.predict(comments)
print(predictions)  # e.g., ['NOT', 'OFFN']
```

#### Loading the Fine-Tuned Transformer
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "mahahate_roberta_results/best_model"
ID2LABEL = {0: "HATE", 1: "OFFN", 2: "PRFN", 3: "NOT"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

text = "हा अतिशय घाणेरडा प्रकार आहे."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_id = logits.argmax(dim=-1).item()

print(f"Predicted Class: {ID2LABEL[predicted_id]}")
```

---

## ⚠️ Academic Notes & Limitations

1. **Absence of Video Identifiers**:
   - The raw comments workbook did not contain `video_id`. Consequently, the split is stratified at the comment level rather than grouped by source video. Near-duplicate or thread-related comments may exist across splits. For final camera-ready or publication benchmarks, verify or recover video IDs where feasible.
2. **Lexicon-Assisted Sampling**:
   - The annotated subset was initially populated through lexicon-assisted discovery. As a result, the distribution of offensive and hateful comments is significantly higher than in organic, uncurated YouTube comments. Keep this sampling characteristic in mind when quoting operational deployment numbers.
3. **Reproducibility**:
   - Random seeds (`seed=42`) are fixed across splitting (`dataset_split_summary.json`), SVM baseline training, and PyTorch/CUDA execution.


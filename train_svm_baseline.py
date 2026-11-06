#!/usr/bin/env python
"""
TF-IDF + Linear SVM baseline for the TE Major Project.

Expected CSV columns:
    comment_text, final_category

Default folder layout:
    project_folder/
      train_svm_baseline.py
      dataset/
        train.csv
        validation.csv
        test.csv

The script chooses C using validation macro-F1, then evaluates the selected
pipeline once on the held-out test set. It saves the trained pipeline and
metrics to svm_baseline_results/.

Important: evaluation reflects the provided labelled subset. If labels were
assigned to lexicon-selected comments, results may not generalize to all
YouTube comments.
"""

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = ["NOT", "PRFN", "OFFN", "HATE"]


def read_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {split_name} file: {path}\n"
            "Pass the correct paths using --train, --validation, and --test."
        )

    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"comment_text", "final_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{split_name} is missing columns {sorted(missing)}. "
            f"Found: {df.columns.tolist()}"
        )

    df["comment_text"] = df["comment_text"].astype(str)
    df["final_category"] = df["final_category"].astype(str).str.strip().str.upper()
    df = df[df["final_category"].isin(LABELS)].copy()
    df = df[df["comment_text"].str.strip().ne("")].copy()

    if df.empty:
        raise ValueError(f"No usable labelled rows found in {split_name}: {path}")
    missing_classes = set(LABELS) - set(df["final_category"].unique())
    if missing_classes:
        print(f"[WARNING] {split_name} has no examples for: {sorted(missing_classes)}")
    return df


def make_pipeline(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=250_000,
                    sublinear_tf=True,
                    strip_accents=None,
                    lowercase=True,
                    dtype="float32",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=300_000,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype="float32",
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LinearSVC(C=c_value, class_weight="balanced", random_state=42)),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate TF-IDF + Linear SVM.")
    parser.add_argument("--data-dir", default="dataset",
                        help="Folder containing train.csv, validation.csv, test.csv.")
    parser.add_argument("--train", default=None, help="Optional explicit train CSV path.")
    parser.add_argument("--validation", default=None, help="Optional explicit validation CSV path.")
    parser.add_argument("--test", default=None, help="Optional explicit test CSV path.")
    parser.add_argument("--outdir", default="svm_baseline_results",
                        help="Output directory for model and metrics.")
    parser.add_argument("--c-values", default="0.5,1,2",
                        help="Comma-separated C values selected by validation macro-F1.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = Path(args.train) if args.train else data_dir / "train.csv"
    val_path = Path(args.validation) if args.validation else data_dir / "validation.csv"
    test_path = Path(args.test) if args.test else data_dir / "test.csv"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = read_split(train_path, "training")
    val_df = read_split(val_path, "validation")
    test_df = read_split(test_path, "test")

    X_train, y_train = train_df["comment_text"], train_df["final_category"]
    X_val, y_val = val_df["comment_text"], val_df["final_category"]
    X_test, y_test = test_df["comment_text"], test_df["final_category"]

    c_values = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]
    if not c_values:
        raise ValueError("No valid C values supplied.")
    if any(not math.isfinite(value) or value <= 0 for value in c_values):
        raise ValueError("Every C value must be a finite number greater than zero.")

    print("=" * 68)
    print("TF-IDF + LINEAR SVM BASELINE")
    print("=" * 68)
    print(f"Training rows  : {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows      : {len(test_df):,}")
    print("\nTraining label distribution:")
    print(y_train.value_counts().reindex(LABELS, fill_value=0).to_string())

    validation_results = []
    best_pipeline = None
    best_c = None
    best_val_f1 = -1.0

    for c_value in c_values:
        print(f"\nFitting model with C={c_value} ...")
        model = make_pipeline(c_value)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_macro_f1 = f1_score(y_val, val_pred, labels=LABELS, average="macro", zero_division=0)
        val_accuracy = accuracy_score(y_val, val_pred)
        validation_results.append({
            "C": c_value,
            "validation_accuracy": float(val_accuracy),
            "validation_macro_f1": float(val_macro_f1),
        })
        print(f"  Validation accuracy : {val_accuracy:.4f}")
        print(f"  Validation macro-F1 : {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_c = c_value
            best_pipeline = model

    # The test split is evaluated only after C has been selected on validation.
    print(f"\nSelected C={best_c} (validation macro-F1={best_val_f1:.4f})")
    print("Evaluating selected model on held-out test set ...")
    test_pred = best_pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, labels=LABELS, average="macro", zero_division=0)
    report_dict = classification_report(
        y_test, test_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, test_pred, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_test, test_pred, labels=LABELS)

    model_path = outdir / "tfidf_linear_svm.joblib"
    joblib.dump(best_pipeline, model_path)

    pd.DataFrame(validation_results).to_csv(
        outdir / "validation_model_selection.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        cm, index=[f"actual_{x}" for x in LABELS],
        columns=[f"predicted_{x}" for x in LABELS]
    ).to_csv(outdir / "test_confusion_matrix.csv", encoding="utf-8-sig")

    (outdir / "test_classification_report.txt").write_text(report_text, encoding="utf-8")
    (outdir / "test_classification_report.json").write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )

    prediction_df = test_df.copy()
    prediction_df["predicted_category"] = test_pred
    prediction_df["correct_prediction"] = (
        prediction_df["final_category"] == prediction_df["predicted_category"]
    )
    prediction_df.to_csv(outdir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "TF-IDF word (1,2) + character (2,5) n-grams + LinearSVC",
        "selected_C": best_c,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "validation_macro_f1_for_selected_model": float(best_val_f1),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "labels": LABELS,
        "class_weight": "balanced",
        "random_state": 42,
        "warning": (
            "Metrics describe this labelled subset only. If labels are concentrated "
            "among lexicon-matched comments, do not assume results generalize to all comments."
        ),
    }
    (outdir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nTEST CLASSIFICATION REPORT")
    print(report_text)
    print(f"Test accuracy : {test_accuracy:.4f}")
    print(f"Test macro-F1 : {test_macro_f1:.4f}")
    print(f"\nSaved outputs to: {outdir.resolve()}")
    print(f"Saved model to  : {model_path.resolve()}")


if __name__ == "__main__":
    main()

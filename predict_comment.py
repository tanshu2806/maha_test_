#!/usr/bin/env python3
"""Run local Marathi comments through saved SVM, RoBERTa, or both models."""

import argparse
import csv
from pathlib import Path

LABELS = ["HATE", "OFFN", "PRFN", "NOT"]


def load_svm(model_path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"SVM pipeline not found: {model_path}. Train it with train_svm_baseline.py first."
        )
    import joblib

    return joblib.load(model_path)


def predict_svm(model, texts):
    return [str(label) for label in model.predict(texts)]


def load_roberta(model_path, device_name):
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"RoBERTa checkpoint not found: {model_path}. Train it with train_mahahate_roberta.py first."
        )

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-enabled PyTorch device is available.")
    device = torch.device(
        "cuda" if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()) else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return torch, tokenizer, model, device


def predict_roberta(components, texts):
    torch, tokenizer, model, device = components
    encoded = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=128, padding=True
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.no_grad():
        predicted_ids = model(**encoded).logits.argmax(dim=-1).tolist()
    if any(predicted_id < 0 or predicted_id >= len(LABELS) for predicted_id in predicted_ids):
        raise ValueError("Model returned an unexpected class id.")
    return [LABELS[predicted_id] for predicted_id in predicted_ids]


def build_predictors(args):
    predictors = {}
    if args.model in ("svm", "both"):
        svm_model = load_svm(args.svm_model)
        predictors["svm_prediction"] = lambda texts: predict_svm(svm_model, texts)
    if args.model in ("roberta", "both"):
        roberta = load_roberta(args.roberta_model, args.device)
        predictors["roberta_prediction"] = lambda texts: predict_roberta(roberta, texts)
    return predictors


def predict_csv(input_path, output_path, text_column, predictors, batch_size):
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("Output CSV must be a different file from the input CSV.")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {input_path}")
        if text_column not in reader.fieldnames:
            raise ValueError(
                f"Text column {text_column!r} not found in {input_path}; "
                f"available columns: {reader.fieldnames}"
            )
        rows = []
        for row_number, row in enumerate(reader, start=1):
            text = row.get(text_column) or ""
            if not text.strip():
                raise ValueError(f"Blank text in column {text_column!r} at CSV data row {row_number}.")
            rows.append((row_number, text))

    if not rows:
        raise ValueError(f"No comment rows found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["prediction_row", *predictors]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            texts = [text for _, text in batch]
            batch_predictions = {
                name: predict(texts) for name, predict in predictors.items()
            }
            if any(len(labels) != len(batch) for labels in batch_predictions.values()):
                raise ValueError("A model returned a different number of predictions than input rows.")
            for index, (row_number, _) in enumerate(batch):
                writer.writerow({
                    "prediction_row": row_number,
                    **{name: labels[index] for name, labels in batch_predictions.items()},
                })
    print(f"Predictions for {len(rows)} rows saved to {output_path} (comment text omitted).")


def main():
    parser = argparse.ArgumentParser(
        description="Classify one Marathi comment or a CSV with saved SVM, RoBERTa, or both."
    )
    parser.add_argument("--model", choices=["svm", "roberta", "both"], default="both")
    parser.add_argument("--text", help="Single comment; omit to enter it interactively.")
    parser.add_argument("--input-csv", type=Path, help="Optional CSV containing comments for batch prediction.")
    parser.add_argument("--output-csv", type=Path, help="Label-only CSV output for --input-csv.")
    parser.add_argument("--text-column", default="comment_text", help="Input CSV text column.")
    parser.add_argument("--batch-size", type=int, default=8, help="Rows processed per inference batch.")
    parser.add_argument(
        "--svm-model",
        type=Path,
        default=Path("svm_baseline_results/tfidf_linear_svm.joblib"),
    )
    parser.add_argument(
        "--roberta-model",
        type=Path,
        default=Path("mahahate_roberta_results/best_model"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if bool(args.input_csv) != bool(args.output_csv):
        parser.error("--input-csv and --output-csv must be supplied together.")
    if args.input_csv and args.text is not None:
        parser.error("--text cannot be combined with --input-csv.")

    predictors = build_predictors(args)
    if args.input_csv:
        predict_csv(args.input_csv, args.output_csv, args.text_column, predictors, args.batch_size)
        return

    text = args.text if args.text is not None else input("Marathi comment: ")
    if not text.strip():
        parser.error("Comment text must not be blank.")

    predictions = {name: predict([text])[0] for name, predict in predictors.items()}
    for name, label in predictions.items():
        print(f"{name.removesuffix('_prediction').upper()}: {label}")
    if len(predictions) == 2 and len(set(predictions.values())) > 1:
        print("The models disagree; review the comment and both model errors before drawing a conclusion.")


if __name__ == "__main__":
    main()

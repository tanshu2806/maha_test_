#!/usr/bin/env python3
"""Run one local Marathi comment through the saved SVM, RoBERTa, or both."""

import argparse
from pathlib import Path

LABELS = ["HATE", "OFFN", "PRFN", "NOT"]


def predict_svm(model_path, text):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"SVM pipeline not found: {model_path}. Train it with train_svm_baseline.py first."
        )
    import joblib

    model = joblib.load(model_path)
    return str(model.predict([text])[0])


def predict_roberta(model_path, text, device_name):
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

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.no_grad():
        predicted_id = int(model(**encoded).logits.argmax(dim=-1).item())
    if predicted_id < 0 or predicted_id >= len(LABELS):
        raise ValueError(f"Model returned an unexpected class id: {predicted_id}")
    return LABELS[predicted_id]


def main():
    parser = argparse.ArgumentParser(
        description="Classify one Marathi comment with the saved SVM, RoBERTa, or both."
    )
    parser.add_argument("--model", choices=["svm", "roberta", "both"], default="both")
    parser.add_argument("--text", help="Comment text; omit to enter it interactively.")
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

    text = args.text if args.text is not None else input("Marathi comment: ")
    if not text.strip():
        parser.error("Comment text must not be blank.")

    predictions = {}
    if args.model in ("svm", "both"):
        predictions["SVM"] = predict_svm(args.svm_model, text)
    if args.model in ("roberta", "both"):
        predictions["RoBERTa"] = predict_roberta(args.roberta_model, text, args.device)

    for model_name, label in predictions.items():
        print(f"{model_name}: {label}")
    if len(predictions) == 2 and len(set(predictions.values())) > 1:
        print("The models disagree; review the comment and both model errors before drawing a conclusion.")


if __name__ == "__main__":
    main()

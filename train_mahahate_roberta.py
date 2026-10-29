#!/usr/bin/env python
"""Fine-tune l3cube-pune/mahahate-multi-roberta on the project's existing split.

Expected CSV columns: comment_text, final_category
Expected files in --data-dir: train.csv, validation.csv, test.csv
Class IDs intentionally follow the checkpoint's pretrained label order:
HATE=0, OFFN=1, PRFN=2, NOT=3.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL_NAME = "l3cube-pune/mahahate-multi-roberta"
LABELS = ["HATE", "OFFN", "PRFN", "NOT"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CommentDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.texts = frame["comment_text"].astype(str).tolist()
        self.labels = frame["final_category"].astype(str).str.strip().str.upper().map(LABEL2ID).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding=False, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_batch(items, tokenizer):
    labels = torch.stack([x.pop("labels") for x in items])
    batch = tokenizer.pad(items, padding=True, return_tensors="pt")
    batch["labels"] = labels
    return batch


def load_split(path):
    df = pd.read_csv(path)
    required = {"comment_text", "final_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["comment_text", "final_category"]).copy()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df["final_category"] = df["final_category"].astype(str).str.strip().str.upper()
    df = df[(df["comment_text"] != "") & df["final_category"].isin(LABEL2ID)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid labelled rows found in {path}")
    return df


def evaluate(model, loader, device, use_amp):
    model.eval()
    y_true, y_pred, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            preds = outputs.logits.argmax(dim=-1)
            y_true.extend(batch["labels"].cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(LABELS))), zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./dataset", help="Folder containing train.csv, validation.csv, test.csv")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default="./mahahate_roberta_results")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("=" * 68)
    print("MAHAHATE MULTI-ROBERTA FINE-TUNING")
    print("=" * 68)
    print(f"Model       : {args.model_name}")
    print(f"Device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GiB")
    print(f"AMP fp16    : {use_amp}")
    print(f"Batch size  : {args.batch_size} x grad accumulation {args.grad_accum}")
    print(f"Max length  : {args.max_length}")

    train_df = load_split(data_dir / "train.csv")
    val_df = load_split(data_dir / "validation.csv")
    test_df = load_split(data_dir / "test.csv")
    print(f"Train rows  : {len(train_df):,}")
    print(f"Val rows    : {len(val_df):,}")
    print(f"Test rows   : {len(test_df):,}")
    print("Label ID mapping:", LABEL2ID)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    # Preserve the pretrained four-class head's semantic ordering.
    model.to(device)

    collate = lambda items: collate_batch(items, tokenizer)
    train_loader = DataLoader(CommentDataset(train_df, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate, pin_memory=use_amp)
    val_loader = DataLoader(CommentDataset(val_df, tokenizer, args.max_length), batch_size=max(2, args.batch_size), shuffle=False, collate_fn=collate, pin_memory=use_amp)
    test_loader = DataLoader(CommentDataset(test_df, tokenizer, args.max_length), batch_size=max(2, args.batch_size), shuffle=False, collate_fn=collate, pin_memory=use_amp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    best_dir = out_dir / "best_model"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if step % 250 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} | batch {step}/{len(train_loader)} | avg train loss {running_loss/step:.4f}")

        val = evaluate(model, val_loader, device, use_amp)
        record = {"epoch": epoch, "train_loss": running_loss / max(1, len(train_loader)), "validation_loss": val["loss"], "validation_accuracy": val["accuracy"], "validation_macro_f1": val["macro_f1"]}
        history.append(record)
        print(f"Epoch {epoch} summary: train_loss={record['train_loss']:.4f}, val_loss={val['loss']:.4f}, val_accuracy={val['accuracy']:.4f}, val_macro_f1={val['macro_f1']:.4f}")

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved new best model (validation macro-F1={best_f1:.4f})")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
                break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    print("\nLoading best validation checkpoint for final test evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test = evaluate(model, test_loader, device, use_amp)
    y_true, y_pred = test["y_true"], test["y_pred"]
    report_dict = classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    pd.DataFrame(cm, index=[f"actual_{x}" for x in LABELS], columns=[f"predicted_{x}" for x in LABELS]).to_csv(out_dir / "test_confusion_matrix.csv")
    pd.DataFrame({"actual_label": [ID2LABEL[x] for x in y_true], "predicted_label": [ID2LABEL[x] for x in y_pred], "comment_text": test_df["comment_text"].tolist()}).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "test_classification_report.txt").write_text(report_text, encoding="utf-8")
    (out_dir / "test_classification_report.json").write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "model": args.model_name,
        "approach": "fine-tuning pretrained four-class MahaHate RoBERTa",
        "label_mapping": LABEL2ID,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_loss": test["loss"],
        "train_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df),
        "max_length": args.max_length, "batch_size": args.batch_size, "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr, "seed": args.seed,
        "elapsed_minutes": round((time.time() - start_time) / 60, 2),
        "warning": "Metrics apply to this labelled subset. Lexicon-based sampling can bias results; do not assume generalization to all YouTube comments."
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nFINAL TEST CLASSIFICATION REPORT")
    print(report_text)
    print(f"Test accuracy : {test['accuracy']:.4f}")
    print(f"Test macro-F1 : {test['macro_f1']:.4f}")
    print(f"Best epoch    : {best_epoch}")
    print(f"Outputs saved : {out_dir.resolve()}")
    print("IMPORTANT: test set evaluated once after selecting the checkpoint by validation macro-F1.")

if __name__ == "__main__":
    main()

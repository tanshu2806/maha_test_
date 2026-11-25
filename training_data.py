"""Shared, privacy-conscious validation for the project's labelled CSV splits."""

import csv
from pathlib import Path

import pandas as pd


def read_labeled_split(path, split_name, labels):
    """Load a labelled split, report filtered rows, and return only usable data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} split file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError(f"{split_name} split has no CSV header: {path}")

    normalized_header = [str(column).strip() for column in header]
    if len(set(normalized_header)) != len(normalized_header):
        raise ValueError(f"{split_name} split has duplicate column names after trimming whitespace: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    df.columns = normalized_header
    required = {"comment_text", "final_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")

    source_rows = len(df)
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df["final_category"] = df["final_category"].astype(str).str.strip().str.upper()
    blank_text = df["comment_text"].eq("")
    valid_label = df["final_category"].isin(labels)
    usable = ~blank_text & valid_label

    print(
        f"[DATA] {split_name}: source_rows={source_rows}, used_rows={int(usable.sum())}, "
        f"blank_text_rows={int(blank_text.sum())}, unsupported_label_rows={int((~valid_label).sum())}"
    )
    df = df.loc[usable].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No usable labelled rows found in {split_name}: {path}")

    missing_labels = set(labels) - set(df["final_category"].unique())
    if missing_labels:
        print(f"[WARNING] {split_name} has no examples for: {sorted(missing_labels)}")
    return df

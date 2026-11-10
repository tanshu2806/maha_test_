#!/usr/bin/env python3
"""Audit normalized duplicate comments across prepared dataset splits.

Only counts and class distributions are printed or saved; comment text is never
included in the report.
"""

import argparse
import csv
import json
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

LABELS = ["NOT", "PRFN", "OFFN", "HATE"]


def normalize_comment(text):
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.casefold().split())


def read_split(name, path):
    if not path.is_file():
        raise FileNotFoundError(f"{name} split file not found: {path}")

    comments = {}
    label_counts = Counter()
    blank_text_rows = 0
    total_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{name} split has no CSV header: {path}")
        columns = {column.strip() for column in reader.fieldnames}
        required = {"comment_text", "final_category"}
        missing = required - columns
        if missing:
            raise ValueError(f"{name} split is missing columns: {sorted(missing)}")

        # Normalize header whitespace while keeping the original row values.
        reader.fieldnames = [column.strip() for column in reader.fieldnames]
        for line_number, row in enumerate(reader, start=2):
            total_rows += 1
            text = (row.get("comment_text") or "").strip()
            label = (row.get("final_category") or "").strip().upper()
            if label not in LABELS:
                raise ValueError(f"Unknown label in {path} at line {line_number}: {label!r}")
            if not text:
                blank_text_rows += 1
                continue
            key = normalize_comment(text)
            comments.setdefault(key, Counter())[label] += 1
            label_counts[label] += 1

    return {
        "rows": total_rows,
        "nonblank_labelled_rows": sum(label_counts.values()),
        "blank_text_rows": blank_text_rows,
        "class_counts": {label: label_counts[label] for label in LABELS},
        "comments": comments,
    }


def summarize_duplicates(split):
    comments = split["comments"]
    groups = [labels for labels in comments.values() if sum(labels.values()) > 1]
    return {
        "duplicate_groups": len(groups),
        "rows_beyond_first_copy": sum(sum(labels.values()) - 1 for labels in groups),
    }


def summarize_overlap(left_name, left, right_name, right):
    left_comments = left["comments"]
    right_comments = right["comments"]
    shared = left_comments.keys() & right_comments.keys()
    label_set_differences = sum(
        set(left_comments[text]) != set(right_comments[text]) for text in shared
    )
    return {
        "splits": [left_name, right_name],
        "shared_normalized_comment_groups": len(shared),
        "left_rows_in_shared_groups": sum(sum(left_comments[text].values()) for text in shared),
        "right_rows_in_shared_groups": sum(sum(right_comments[text].values()) for text in shared),
        "groups_with_different_label_sets": label_set_differences,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Report class counts and normalized duplicate overlap without printing comment text."
    )
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--validation", type=Path, default=Path("dataset/validation.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--json-out", type=Path, help="Optional path for aggregate-only JSON output.")
    args = parser.parse_args()

    paths = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    splits = {name: read_split(name, path) for name, path in paths.items()}

    report = {
        "normalization": "Unicode NFKC, casefold, and collapsed whitespace",
        "splits": {
            name: {
                "rows": split["rows"],
                "nonblank_labelled_rows": split["nonblank_labelled_rows"],
                "blank_text_rows": split["blank_text_rows"],
                "class_counts": split["class_counts"],
                "within_split_duplicates": summarize_duplicates(split),
            }
            for name, split in splits.items()
        },
        "cross_split_overlap": [
            summarize_overlap(left_name, splits[left_name], right_name, splits[right_name])
            for left_name, right_name in combinations(paths, 2)
        ],
        "note": (
            "Normalized overlap is a potential split-leakage signal, not proof of source identity. "
            "No comment text is included."
        ),
    }

    print("Split audit (aggregate counts only)")
    for name, details in report["splits"].items():
        print(
            f"{name}: rows={details['rows']}, blank={details['blank_text_rows']}, "
            f"duplicates={details['within_split_duplicates']['duplicate_groups']} groups, "
            f"class_counts={details['class_counts']}"
        )
    print("\nCross-split normalized overlap")
    for overlap in report["cross_split_overlap"]:
        pair = " / ".join(overlap["splits"])
        print(
            f"{pair}: {overlap['shared_normalized_comment_groups']} shared groups; "
            f"label-set differences={overlap['groups_with_different_label_sets']}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nAggregate report saved to {args.json_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare saved SVM and RoBERTa predictions without printing comment text."""

import argparse
import csv
import json
from pathlib import Path

LABELS = ["NOT", "PRFN", "OFFN", "HATE"]


def load_predictions(path: Path, truth_columns, prediction_columns):
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {path}")

        def find_column(candidates):
            return next((column for column in candidates if column in reader.fieldnames), None)

        truth_column = find_column(truth_columns)
        prediction_column = find_column(prediction_columns)
        if not truth_column or not prediction_column or "comment_text" not in reader.fieldnames:
            raise ValueError(
                f"{path} must contain comment_text, one of {truth_columns}, "
                f"and one of {prediction_columns}; found {reader.fieldnames}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            truth = (row.get(truth_column) or "").strip().upper()
            prediction = (row.get(prediction_column) or "").strip().upper()
            comment = row.get("comment_text") or ""
            if truth not in LABELS or prediction not in LABELS:
                raise ValueError(f"Unexpected label in {path} at line {line_number}")
            if not comment.strip():
                raise ValueError(f"Blank comment_text in {path} at line {line_number}")
            # Keep text only for in-memory alignment; never include it in output.
            rows.append((comment, truth, prediction))

    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def score(rows):
    results = {}
    for label in LABELS:
        tp = sum(truth == label and prediction == label for truth, prediction in rows)
        fp = sum(truth != label and prediction == label for truth, prediction in rows)
        fn = sum(truth == label and prediction != label for truth, prediction in rows)
        support = sum(truth == label for truth, _ in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    accuracy = sum(truth == prediction for truth, prediction in rows) / len(rows)
    macro_f1 = sum(results[label]["f1"] for label in LABELS) / len(LABELS)
    matrix = {truth: {prediction: 0 for prediction in LABELS} for truth in LABELS}
    errors = {}
    for truth, prediction in rows:
        matrix[truth][prediction] += 1
        if truth != prediction:
            pair = f"{truth} -> {prediction}"
            errors[pair] = errors.get(pair, 0) + 1
    errors = dict(sorted(errors.items(), key=lambda item: (-item[1], item[0])))
    return {
        "rows": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": results,
        "confusion_matrix": matrix,
        "misclassification_pairs": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare existing SVM and RoBERTa test predictions without exposing comment text."
    )
    parser.add_argument("--svm", type=Path, default=Path("svm_baseline_results/test_predictions.csv"))
    parser.add_argument("--roberta", type=Path, default=Path("mahahate_roberta_results/test_predictions.csv"))
    parser.add_argument("--json-out", type=Path, help="Optional path for aggregate-only JSON output.")
    parser.add_argument(
        "--errors-out",
        type=Path,
        help="Optional CSV of misclassified row numbers and labels; comment text is never written.",
    )
    args = parser.parse_args()

    svm_rows = load_predictions(
        args.svm,
        ["final_category", "actual_label"],
        ["predicted_category", "predicted_label"],
    )
    roberta_rows = load_predictions(
        args.roberta,
        ["actual_label", "final_category"],
        ["predicted_label", "predicted_category"],
    )

    if len(svm_rows) != len(roberta_rows):
        raise ValueError(f"Row counts differ: SVM={len(svm_rows)}, RoBERTa={len(roberta_rows)}")

    aligned = []
    for index, (svm, roberta) in enumerate(zip(svm_rows, roberta_rows), start=1):
        if svm[0] != roberta[0] or svm[1] != roberta[1]:
            raise ValueError(
                f"Prediction files do not align at row {index}; "
                "use outputs generated from the same ordered test split."
            )
        aligned.append((svm[1], svm[2], roberta[2]))

    svm_scored = score([(truth, prediction) for truth, prediction, _ in aligned])
    roberta_scored = score([(truth, prediction) for truth, _, prediction in aligned])
    svm_only = sum(svm == truth and roberta != truth for truth, svm, roberta in aligned)
    roberta_only = sum(roberta == truth and svm != truth for truth, svm, roberta in aligned)
    both_wrong = sum(svm != truth and roberta != truth for truth, svm, roberta in aligned)
    same_prediction = sum(svm == roberta for _, svm, roberta in aligned)

    report = {
        "rows": len(aligned),
        "alignment": "verified by comment_text and actual label, in row order",
        "svm": svm_scored,
        "roberta": roberta_scored,
        "comparison": {
            "same_prediction": same_prediction,
            "svm_only_correct": svm_only,
            "roberta_only_correct": roberta_only,
            "both_wrong": both_wrong,
            "accuracy_delta_roberta_minus_svm": roberta_scored["accuracy"] - svm_scored["accuracy"],
            "macro_f1_delta_roberta_minus_svm": roberta_scored["macro_f1"] - svm_scored["macro_f1"],
        },
        "note": "Metrics describe these saved predictions on this labelled split only.",
    }

    print(f"Aligned test rows: {len(aligned)}")
    print(f"{'Model':<12} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"{'SVM':<12} {svm_scored['accuracy']:>10.4f} {svm_scored['macro_f1']:>10.4f}")
    print(f"{'RoBERTa':<12} {roberta_scored['accuracy']:>10.4f} {roberta_scored['macro_f1']:>10.4f}")
    print("\nPer-class F1 (precision / recall / support)")
    for label in LABELS:
        svm_class = svm_scored["per_class"][label]
        roberta_class = roberta_scored["per_class"][label]
        print(
            f"{label:<5} SVM {svm_class['f1']:.3f} "
            f"({svm_class['precision']:.3f}/{svm_class['recall']:.3f}, n={svm_class['support']}) | "
            f"RoBERTa {roberta_class['f1']:.3f} "
            f"({roberta_class['precision']:.3f}/{roberta_class['recall']:.3f})"
        )

    for name, results in (("SVM", svm_scored), ("RoBERTa", roberta_scored)):
        print(f"\n{name} confusion matrix (rows=actual, columns=predicted)")
        print("actual\\pred " + " ".join(f"{label:>6}" for label in LABELS))
        for actual in LABELS:
            counts = results["confusion_matrix"][actual]
            print(f"{actual:>10} " + " ".join(f"{counts[label]:>6}" for label in LABELS))

    print("\nPrediction comparison")
    print(f"Same prediction: {same_prediction}")
    print(f"SVM-only correct: {svm_only}")
    print(f"RoBERTa-only correct: {roberta_only}")
    print(f"Both wrong: {both_wrong}")
    for name, results in (("SVM", svm_scored), ("RoBERTa", roberta_scored)):
        print(f"\n{name} most common error pairs")
        pairs = list(results["misclassification_pairs"].items())[:5]
        if pairs:
            for pair, count in pairs:
                print(f"  {pair}: {count}")
        else:
            print("  none")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nAggregate report saved to {args.json_out}")

    if args.errors_out:
        args.errors_out.parent.mkdir(parents=True, exist_ok=True)
        with args.errors_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "prediction_row",
                "actual_label",
                "svm_prediction",
                "roberta_prediction",
                "svm_correct",
                "roberta_correct",
                "models_disagree",
            ])
            for row_number, (truth, svm, roberta) in enumerate(aligned, start=1):
                svm_correct = svm == truth
                roberta_correct = roberta == truth
                if not svm_correct or not roberta_correct:
                    writer.writerow([
                        row_number,
                        truth,
                        svm,
                        roberta,
                        svm_correct,
                        roberta_correct,
                        svm != roberta,
                    ])
        print(f"Misclassified row index saved to {args.errors_out} (comment text omitted).")


if __name__ == "__main__":
    main()

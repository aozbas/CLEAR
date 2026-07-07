"""Metric helpers for experimental classifier evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from ml.evaluation.schema import HAM10000_LABELS, validate_label


def confusion_matrix(truth: Sequence[str], predictions: Sequence[str]) -> list[list[int]]:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length.")

    label_to_index = {label: index for index, label in enumerate(HAM10000_LABELS)}
    matrix = [[0 for _ in HAM10000_LABELS] for _ in HAM10000_LABELS]
    for actual, predicted in zip(truth, predictions, strict=True):
        validate_label(actual)
        validate_label(predicted)
        matrix[label_to_index[actual]][label_to_index[predicted]] += 1
    return matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def per_class_metrics(confusion: list[list[int]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(HAM10000_LABELS):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = _safe_divide(true_positive, predicted)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        metrics[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def summarize_metrics(truth: Sequence[str], predictions: Sequence[str]) -> dict[str, object]:
    confusion = confusion_matrix(truth, predictions)
    per_class = per_class_metrics(confusion)
    total = len(truth)
    correct = sum(confusion[index][index] for index in range(len(HAM10000_LABELS)))

    precision_values = [per_class[label]["precision"] for label in HAM10000_LABELS]
    recall_values = [per_class[label]["recall"] for label in HAM10000_LABELS]
    f1_values = [per_class[label]["f1"] for label in HAM10000_LABELS]

    return {
        "accuracy": _safe_divide(correct, total),
        "balanced_accuracy": sum(recall_values) / len(recall_values),
        "macro_precision": sum(precision_values) / len(precision_values),
        "macro_recall": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "labels": list(HAM10000_LABELS),
    }

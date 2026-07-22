"""Metric helpers for experimental classifier evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS, validate_label


def confusion_matrix(
    truth: Sequence[str],
    predictions: Sequence[str],
    *,
    labels: tuple[str, ...] = PAD_UFES_NATIVE_LABELS,
) -> list[list[int]]:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length.")

    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(truth, predictions, strict=True):
        validate_label(actual, labels=labels)
        validate_label(predicted, labels=labels)
        matrix[label_to_index[actual]][label_to_index[predicted]] += 1
    return matrix


def _validate_confusion_shape(confusion: list[list[int]], labels: tuple[str, ...]) -> None:
    if len(confusion) != len(labels) or any(len(row) != len(labels) for row in confusion):
        raise ValueError("confusion matrix shape must match labels.")


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def per_class_metrics(
    confusion: list[list[int]],
    *,
    labels: tuple[str, ...] = PAD_UFES_NATIVE_LABELS,
) -> dict[str, dict[str, float | int]]:
    _validate_confusion_shape(confusion, labels)
    metrics: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
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


def summarize_metrics(
    truth: Sequence[str],
    predictions: Sequence[str],
    *,
    labels: tuple[str, ...] = PAD_UFES_NATIVE_LABELS,
) -> dict[str, object]:
    confusion = confusion_matrix(truth, predictions, labels=labels)
    per_class = per_class_metrics(confusion, labels=labels)
    total = len(truth)
    correct = sum(confusion[index][index] for index in range(len(labels)))

    precision_values = [per_class[label]["precision"] for label in labels]
    recall_values = [per_class[label]["recall"] for label in labels]
    f1_values = [per_class[label]["f1"] for label in labels]
    covered_labels = [label for label in labels if int(per_class[label]["support"]) > 0]
    covered_precision_values = [per_class[label]["precision"] for label in covered_labels]
    covered_recall_values = [per_class[label]["recall"] for label in covered_labels]
    covered_f1_values = [per_class[label]["f1"] for label in covered_labels]

    return {
        "accuracy": _safe_divide(correct, total),
        "balanced_accuracy": sum(recall_values) / len(recall_values),
        "macro_precision": sum(precision_values) / len(precision_values),
        "macro_recall": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
        "covered_labels": covered_labels,
        "covered_label_balanced_accuracy": _safe_mean(covered_recall_values),
        "covered_label_macro_precision": _safe_mean(covered_precision_values),
        "covered_label_macro_recall": _safe_mean(covered_recall_values),
        "covered_label_macro_f1": _safe_mean(covered_f1_values),
        "prediction_distribution": {
            label: {
                "count": int(per_class[label]["predicted"]),
                "fraction": _safe_divide(float(per_class[label]["predicted"]), total),
            }
            for label in labels
        },
        "per_class": per_class,
        "confusion_matrix": confusion,
        "labels": list(labels),
    }


def _safe_mean(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)

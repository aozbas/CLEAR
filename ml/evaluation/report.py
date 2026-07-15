"""Report writing for local experimental evaluation runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ml.evaluation.schema import HAM10000_LABELS, EvaluationExample, ModelMetadata, ModelPrediction


def write_report(
    output_dir: Path,
    *,
    model_metadata: ModelMetadata,
    examples: Sequence[EvaluationExample],
    predictions: Sequence[ModelPrediction],
    metrics: Mapping[str, object],
    dataset_metadata: Mapping[str, object] | None = None,
) -> None:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": asdict(model_metadata),
    }
    if dataset_metadata is not None:
        metadata["dataset"] = dict(dataset_metadata)

    (output_dir / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_predictions_csv(output_dir / "predictions.csv", examples, predictions)
    (output_dir / "summary.md").write_text(
        _summary_markdown(model_metadata, metrics, len(examples), dataset_metadata),
        encoding="utf-8",
    )


def _write_predictions_csv(
    output_path: Path,
    examples: Sequence[EvaluationExample],
    predictions: Sequence[ModelPrediction],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "truth", "prediction", "confidence", "latency_ms"],
        )
        writer.writeheader()
        for example, prediction in zip(examples, predictions, strict=True):
            writer.writerow(
                {
                    "image_path": str(example.image_path),
                    "truth": example.label,
                    "prediction": prediction.label,
                    "confidence": prediction.confidence,
                    "latency_ms": "" if prediction.latency_ms is None else prediction.latency_ms,
                }
            )


def _summary_markdown(
    model_metadata: ModelMetadata,
    metrics: Mapping[str, object],
    sample_count: int,
    dataset_metadata: Mapping[str, object] | None = None,
) -> str:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]
    rows = []
    for name in metric_names:
        if name in metrics:
            value = metrics[name]
            rows.append(f"| {name} | {_format_value(value)} |")

    metric_table = "\n".join(["| Metric | Value |", "|---|---|", *rows])
    lines = [
        f"# Evaluation Summary: {model_metadata.name}",
        "",
        "This report records experimental classification metrics only. "
        "It is not a medical diagnosis.",
        "",
        f"Samples evaluated: {sample_count}",
    ]
    if dataset_metadata is not None:
        lines.extend(
            [
                "",
                f"Dataset: {dataset_metadata['name']}",
                f"Split type: {dataset_metadata['split_type']}",
            ]
        )
        contamination = dataset_metadata.get("contamination_notes", [])
        if contamination:
            lines.extend(["", "Contamination notes:"])
            lines.extend(f"- {note}" for note in contamination)

    lines.extend(["", metric_table, ""])
    covered_metrics = _covered_metrics_markdown(metrics)
    if covered_metrics:
        lines.extend(covered_metrics)

    prediction_distribution = _prediction_distribution_markdown(metrics)
    if prediction_distribution:
        lines.extend(prediction_distribution)

    phone_stress = _phone_stress_markdown(metrics)
    if phone_stress:
        lines.extend(phone_stress)

    return "\n".join(lines)


def _covered_metrics_markdown(metrics: Mapping[str, object]) -> list[str]:
    covered_labels = metrics.get("covered_labels")
    if not isinstance(covered_labels, list):
        return []

    metric_names = [
        "covered_label_balanced_accuracy",
        "covered_label_macro_precision",
        "covered_label_macro_recall",
        "covered_label_macro_f1",
    ]
    rows = [
        f"| {name} | {_format_value(metrics[name])} |" for name in metric_names if name in metrics
    ]
    if not rows:
        return []

    return [
        f"Covered labels: {', '.join(str(label) for label in covered_labels)}",
        "",
        "Covered-label metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        *rows,
        "",
    ]


def _prediction_distribution_markdown(metrics: Mapping[str, object]) -> list[str]:
    distribution = metrics.get("prediction_distribution")
    if not isinstance(distribution, Mapping):
        return []

    labels = metrics.get("labels")
    if not isinstance(labels, list):
        labels = list(HAM10000_LABELS)

    rows = []
    for label in labels:
        label_distribution = distribution.get(str(label))
        if not isinstance(label_distribution, Mapping):
            continue
        rows.append(
            "| "
            f"{label} | "
            f"{_format_value(label_distribution.get('count', 0))} | "
            f"{_format_value(label_distribution.get('fraction', 0.0))} |"
        )
    if not rows:
        return []

    return [
        "Prediction distribution",
        "",
        "| Label | Count | Fraction |",
        "|---|---:|---:|",
        *rows,
        "",
    ]


def _phone_stress_markdown(metrics: Mapping[str, object]) -> list[str]:
    phone_stress = metrics.get("phone_stress")
    if not isinstance(phone_stress, Mapping):
        return []

    rows = []
    aggregate = phone_stress.get("aggregate")
    if isinstance(aggregate, Mapping):
        rows.append(_phone_stress_row("aggregate", aggregate))

    variants = phone_stress.get("variants")
    if isinstance(variants, Mapping):
        for variant_name in sorted(variants):
            variant_metrics = variants[variant_name]
            if isinstance(variant_metrics, Mapping):
                rows.append(_phone_stress_row(str(variant_name), variant_metrics))

    if not rows:
        return []

    return [
        "Phone-photo stress tests",
        "",
        "| Variant | Samples | Accuracy | Covered macro F1 | Melanoma recall | p95 latency ms |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ]


def _phone_stress_row(name: str, metrics: Mapping[str, object]) -> str:
    return (
        f"| {name} | "
        f"{_format_value(metrics.get('sample_count', ''))} | "
        f"{_format_value(metrics.get('accuracy', ''))} | "
        f"{_format_value(metrics.get('covered_label_macro_f1', ''))} | "
        f"{_format_value(_melanoma_recall(metrics))} | "
        f"{_format_value(metrics.get('latency_p95_ms', ''))} |"
    )


def _melanoma_recall(metrics: Mapping[str, object]) -> object:
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        return ""
    melanoma = per_class.get("melanoma")
    if not isinstance(melanoma, Mapping):
        return ""
    return melanoma.get("recall", "")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        formatted = f"{value:.4f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    return str(value)

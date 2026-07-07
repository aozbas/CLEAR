"""Report writing for local experimental evaluation runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ml.evaluation.schema import EvaluationExample, ModelMetadata, ModelPrediction


def write_report(
    output_dir: Path,
    *,
    model_metadata: ModelMetadata,
    examples: Sequence[EvaluationExample],
    predictions: Sequence[ModelPrediction],
    metrics: Mapping[str, object],
) -> None:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "model": asdict(model_metadata),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_predictions_csv(output_dir / "predictions.csv", examples, predictions)
    (output_dir / "summary.md").write_text(
        _summary_markdown(model_metadata, metrics, len(examples)),
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
            rows.append(f"| {name} | {value} |")

    metric_table = "\n".join(["| Metric | Value |", "|---|---|", *rows])
    return "\n".join(
        [
            f"# Evaluation Summary: {model_metadata.name}",
            "",
            "This report records experimental classification metrics only. "
            "It is not a medical diagnosis.",
            "",
            f"Samples evaluated: {sample_count}",
            "",
            metric_table,
            "",
        ]
    )

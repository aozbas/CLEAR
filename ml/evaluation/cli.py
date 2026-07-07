"""Command-line entry point for local experimental model evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ml.evaluation.adapters.baseline import BaselineAdapter
from ml.evaluation.dataset import load_examples
from ml.evaluation.metrics import summarize_metrics
from ml.evaluation.report import write_report
from ml.evaluation.schema import ModelPrediction

DEFAULT_SPLIT_CSV = Path("ml/data/splits/ham10000.csv")
DEFAULT_OUTPUT_DIR = Path("ml/runs/evaluation/baseline-test")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        adapter = _build_adapter(args.model, model_path=args.model_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    examples = load_examples(
        args.split_csv,
        args.split,
        max_samples=args.max_samples,
        samples_per_label=args.samples_per_label,
    )
    predictions = [adapter.predict_image(example.image_path) for example in examples]
    metrics = summarize_metrics(
        [example.label for example in examples],
        [prediction.label for prediction in predictions],
    )
    write_report(
        args.out,
        model_metadata=adapter.metadata,
        examples=examples,
        predictions=predictions,
        metrics={**metrics, **_latency_metrics(predictions)},
    )
    print(
        "Wrote experimental classification report to "
        f"{args.out} (macro_f1={metrics['macro_f1']}, "
        f"balanced_accuracy={metrics['balanced_accuracy']})"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-per-label", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    return parser


def _build_adapter(model_name: str, *, model_path: Path | None) -> BaselineAdapter:
    if model_name == "baseline":
        return BaselineAdapter(model_path=model_path)
    raise ValueError(f"Unsupported model: {model_name}")


def _latency_metrics(predictions: Sequence[ModelPrediction]) -> dict[str, float]:
    latencies = sorted(
        prediction.latency_ms for prediction in predictions if prediction.latency_ms is not None
    )
    if not latencies:
        return {"latency_mean_ms": 0.0, "latency_p95_ms": 0.0}

    p95_index = min(len(latencies) - 1, int(round((len(latencies) - 1) * 0.95)))
    return {
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_p95_ms": latencies[p95_index],
    }


if __name__ == "__main__":
    raise SystemExit(main())

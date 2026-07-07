"""Command-line entry point for local experimental model evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from ml.evaluation.adapters.baseline import BaselineAdapter
from ml.evaluation.adapters.huggingface_image_classifier import (
    HuggingFaceImageClassifierAdapter,
)
from ml.evaluation.candidates import get_candidate
from ml.evaluation.dataset import load_examples
from ml.evaluation.metrics import summarize_metrics
from ml.evaluation.report import write_report
from ml.evaluation.schema import ModelPrediction

DEFAULT_SPLIT_CSV = Path("ml/data/splits/ham10000.csv")
DEFAULT_OUTPUT_DIR = Path("ml/runs/evaluation/baseline-test")
DEFAULT_CACHE_DIR = Path("ml/model_cache/huggingface")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    if args.inspect_model is not None:
        _inspect_model(args.inspect_model, args.out)
        print(f"Wrote model inspection report to {args.out}")
        return 0

    if args.model is None:
        parser.print_usage(sys.stderr)
        print("error: --model is required unless --inspect-model is used", file=sys.stderr)
        return 2

    try:
        adapter = _build_adapter(
            args.model,
            model_path=args.model_path,
            cache_dir=args.cache_dir,
        )
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
    parser.add_argument("--inspect-model", default=None)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-per-label", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser


def _build_adapter(
    model_name: str,
    *,
    model_path: Path | None,
    cache_dir: Path,
) -> BaselineAdapter | HuggingFaceImageClassifierAdapter:
    candidate = get_candidate(model_name)
    if model_name == "baseline":
        return BaselineAdapter(model_path=model_path)
    if candidate.adapter_type == "huggingface_image_classifier":
        return HuggingFaceImageClassifierAdapter(
            model_id=candidate.name,
            revision=candidate.revision,
            label_map=candidate.label_map,
            cache_dir=cache_dir,
            license_name=candidate.license,
        )
    raise ValueError(
        f"Unsupported model: {model_name} has no runnable evaluation adapter."
    )


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


def _inspect_model(model_id: str, output_dir: Path) -> None:
    info = HfApi().model_info(model_id, files_metadata=True)
    card_data = getattr(info, "card_data", None)
    config = getattr(info, "config", None) or {}
    config_labels = config.get("id2label") or _download_config_labels(model_id, info)
    inspection = {
        "model_id": info.id,
        "revision": info.sha,
        "pipeline_tag": info.pipeline_tag,
        "library_name": info.library_name,
        "license": _card_value(card_data, "license"),
        "license_name": _card_value(card_data, "license_name"),
        "license_link": _card_value(card_data, "license_link"),
        "datasets": _card_value(card_data, "datasets"),
        "tags": info.tags,
        "config_labels": config_labels,
        "files": [
            {
                "name": sibling.rfilename,
                "size": getattr(sibling, "size", None),
            }
            for sibling in info.siblings
        ],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inspection.json").write_text(
        json.dumps(inspection, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _inspection_summary_markdown(inspection),
        encoding="utf-8",
    )


def _card_value(card_data: object, name: str) -> object:
    if card_data is None:
        return None
    return getattr(card_data, name, None)


def _download_config_labels(model_id: str, info: object) -> dict[str, str]:
    has_config = any(sibling.rfilename == "config.json" for sibling in info.siblings)
    if not has_config:
        return {}

    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=info.sha,
            cache_dir=DEFAULT_CACHE_DIR,
        )
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return {}

    labels = config.get("id2label") or {}
    return {str(key): str(value) for key, value in labels.items()}


def _inspection_summary_markdown(inspection: dict[str, object]) -> str:
    return "\n".join(
        [
            f"# Model Inspection: {inspection['model_id']}",
            "",
            "This inspection is for experimental classification evaluation only.",
            "",
            f"- Revision: {inspection['revision']}",
            f"- Library: {inspection['library_name']}",
            f"- Pipeline: {inspection['pipeline_tag']}",
            f"- License: {inspection['license']}",
            f"- Labels: {inspection['config_labels']}",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

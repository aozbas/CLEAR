"""Command-line entry point for local experimental model evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from ml.evaluation.adapters.baseline import BaselineAdapter
from ml.evaluation.adapters.huggingface_image_classifier import (
    HuggingFaceImageClassifierAdapter,
)
from ml.evaluation.adapters.keras_h5 import KerasH5Adapter
from ml.evaluation.adapters.zero_shot import OpenClipZeroShotAdapter, TransformersZeroShotAdapter
from ml.evaluation.candidates import get_candidate
from ml.evaluation.dataset import load_examples
from ml.evaluation.dataset_sources import (
    DATASET_SOURCES,
    contamination_notes,
    get_dataset_source,
)
from ml.evaluation.metrics import summarize_metrics
from ml.evaluation.report import write_report
from ml.evaluation.schema import HAM10000_LABELS, ModelPrediction
from ml.evaluation.stress import evaluate_phone_stress

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
        candidate = get_candidate(args.model)
        dataset_source = get_dataset_source(args.dataset_source)
        adapter_labels = _adapter_labels(dataset_source)
        adapter = _build_adapter(
            args.model,
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            labels=adapter_labels,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    examples = load_examples(
        args.split_csv,
        args.split,
        max_samples=args.max_samples,
        samples_per_label=args.samples_per_label,
        labels=_evaluation_labels(dataset_source, adapter.metadata.labels),
    )
    predictions = [adapter.predict_image(example.image_path) for example in examples]
    metrics = summarize_metrics(
        [example.label for example in examples],
        [prediction.label for prediction in predictions],
        labels=_evaluation_labels(dataset_source, adapter.metadata.labels),
    )
    report_metrics = {**metrics, **_latency_metrics(predictions)}
    if args.phone_stress:
        report_metrics["phone_stress"] = evaluate_phone_stress(
            adapter,
            examples,
            output_dir=args.out / "phone_stress_images",
        )

    write_report(
        args.out,
        model_metadata=adapter.metadata,
        examples=examples,
        predictions=predictions,
        metrics=report_metrics,
        dataset_metadata=_dataset_metadata(
            dataset_source.key,
            model_datasets=candidate.training_datasets,
            clean_dataset_sources=candidate.clean_dataset_sources,
        ),
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
    parser.add_argument(
        "--dataset-source",
        choices=sorted(DATASET_SOURCES),
        default="ham10000_internal",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-per-label", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--phone-stress",
        action="store_true",
        help="Run opt-in synthetic phone-photo stress transforms into the report output.",
    )
    return parser


def _build_adapter(
    model_name: str,
    *,
    model_path: Path | None,
    cache_dir: Path,
    labels: tuple[str, ...] = HAM10000_LABELS,
) -> (
    BaselineAdapter
    | HuggingFaceImageClassifierAdapter
    | KerasH5Adapter
    | OpenClipZeroShotAdapter
    | TransformersZeroShotAdapter
):
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
    if candidate.adapter_type == "keras_h5":
        if candidate.artifact_filename is None:
            raise ValueError(f"Unsupported model: {model_name} is missing an artifact filename.")
        return KerasH5Adapter(
            model_id=candidate.name,
            revision=candidate.revision,
            artifact_filename=candidate.artifact_filename,
            label_map=candidate.label_map,
            cache_dir=cache_dir,
            license_name=candidate.license,
        )
    if candidate.adapter_type == "open_clip_zero_shot":
        return OpenClipZeroShotAdapter(
            model_id=candidate.name,
            revision=candidate.revision,
            cache_dir=cache_dir,
            license_name=candidate.license,
            labels=labels,
        )
    if candidate.adapter_type == "transformers_zero_shot":
        return TransformersZeroShotAdapter(
            model_id=candidate.name,
            revision=candidate.revision,
            cache_dir=cache_dir,
            license_name=candidate.license,
            labels=labels,
        )
    if candidate.adapter_type == "embedding_linear_probe":
        raise ValueError(
            f"Unsupported direct evaluation model: {model_name} requires the embedding-probe "
            "workflow."
        )
    raise ValueError(f"Unsupported model: {model_name} has no runnable evaluation adapter.")


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


def _adapter_labels(dataset_source: object) -> tuple[str, ...]:
    partial_label_set = bool(getattr(dataset_source, "partial_label_set", True))
    if partial_label_set:
        return HAM10000_LABELS
    labels = getattr(dataset_source, "labels", HAM10000_LABELS)
    return tuple(str(label) for label in labels)


def _evaluation_labels(dataset_source: object, model_labels: list[str]) -> tuple[str, ...]:
    partial_label_set = bool(getattr(dataset_source, "partial_label_set", True))
    if partial_label_set:
        return tuple(model_labels)
    labels = getattr(dataset_source, "labels", model_labels)
    return tuple(str(label) for label in labels)


def _dataset_metadata(
    dataset_source_key: str,
    *,
    model_datasets: list[str],
    clean_dataset_sources: list[str],
) -> dict[str, object]:
    dataset_source = get_dataset_source(dataset_source_key)
    metadata = asdict(dataset_source)
    if dataset_source.key in clean_dataset_sources:
        metadata["contamination_notes"] = []
    else:
        metadata["contamination_notes"] = contamination_notes(
            dataset_source=dataset_source,
            model_datasets=model_datasets,
        )
    return metadata


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

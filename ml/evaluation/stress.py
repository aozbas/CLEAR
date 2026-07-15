"""Synthetic phone-photo stress evaluation for local model reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from ml.evaluation.adapters.base import LesionModelAdapter
from ml.evaluation.metrics import summarize_metrics
from ml.evaluation.schema import HAM10000_LABELS, EvaluationExample, ModelPrediction

PHONE_STRESS_VARIANTS: Mapping[str, str] = {
    "blur": "Gaussian blur simulating slight camera shake.",
    "jpeg_compression": "Low-quality JPEG compression.",
    "brightness_dark": "Darker exposure shift.",
    "brightness_bright": "Brighter exposure shift.",
    "crop_zoom": "Centered crop and resize simulating close framing.",
    "rotation": "Small in-plane rotation.",
    "low_resolution": "Downsample and resize simulating low-resolution capture.",
}


@dataclass(frozen=True)
class PhoneStressExample:
    variant_key: str
    variant_description: str
    original_image_path: Path
    example: EvaluationExample


def build_phone_stress_examples(
    examples: Sequence[EvaluationExample],
    output_dir: Path,
) -> list[PhoneStressExample]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _shared_labels(examples)

    stress_examples: list[PhoneStressExample] = []
    for index, example in enumerate(examples):
        with Image.open(example.image_path) as opened:
            source = opened.convert("RGB")

        for variant_key, description in PHONE_STRESS_VARIANTS.items():
            stressed = _apply_variant(source, variant_key)
            output_path = output_dir / f"{index:05d}-{example.label}-{variant_key}.jpg"
            _save_variant(stressed, output_path, variant_key)
            stress_examples.append(
                PhoneStressExample(
                    variant_key=variant_key,
                    variant_description=description,
                    original_image_path=example.image_path,
                    example=EvaluationExample(
                        image_path=output_path,
                        label=example.label,
                        split=example.split,
                        labels=labels,
                    ),
                )
            )

    return stress_examples


def evaluate_phone_stress(
    adapter: LesionModelAdapter,
    examples: Sequence[EvaluationExample],
    output_dir: Path,
) -> dict[str, object]:
    labels = _shared_labels(examples)
    stress_examples = build_phone_stress_examples(examples, output_dir)
    variant_truth: dict[str, list[str]] = defaultdict(list)
    variant_predictions: dict[str, list[ModelPrediction]] = defaultdict(list)
    all_truth: list[str] = []
    all_predictions: list[ModelPrediction] = []

    for stress_example in stress_examples:
        prediction = adapter.predict_image(stress_example.example.image_path)
        variant_truth[stress_example.variant_key].append(stress_example.example.label)
        variant_predictions[stress_example.variant_key].append(prediction)
        all_truth.append(stress_example.example.label)
        all_predictions.append(prediction)

    variants = {}
    for variant_key, description in PHONE_STRESS_VARIANTS.items():
        predictions = variant_predictions.get(variant_key, [])
        metrics = _summarize_predictions(
            variant_truth.get(variant_key, []),
            predictions,
            labels=labels,
        )
        metrics["description"] = description
        variants[variant_key] = metrics

    return {
        "aggregate": _summarize_predictions(all_truth, all_predictions, labels=labels),
        "variants": variants,
    }


def _summarize_predictions(
    truth: Sequence[str],
    predictions: Sequence[ModelPrediction],
    *,
    labels: tuple[str, ...],
) -> dict[str, object]:
    metrics = summarize_metrics(
        truth,
        [prediction.label for prediction in predictions],
        labels=labels,
    )
    metrics.update(_latency_metrics(predictions))
    metrics["sample_count"] = len(truth)
    return metrics


def _shared_labels(examples: Sequence[EvaluationExample]) -> tuple[str, ...]:
    if not examples:
        return HAM10000_LABELS

    labels = examples[0].labels
    if any(example.labels != labels for example in examples[1:]):
        raise ValueError("Phone-stress examples must use one shared label set.")
    return labels


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


def _apply_variant(image: Image.Image, variant_key: str) -> Image.Image:
    if variant_key == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if variant_key == "jpeg_compression":
        return image.copy()
    if variant_key == "brightness_dark":
        return ImageEnhance.Brightness(image).enhance(0.65)
    if variant_key == "brightness_bright":
        return ImageEnhance.Brightness(image).enhance(1.35)
    if variant_key == "crop_zoom":
        return _center_crop_zoom(image)
    if variant_key == "rotation":
        return image.rotate(
            12,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(0, 0, 0),
        )
    if variant_key == "low_resolution":
        return _low_resolution(image)
    raise ValueError(f"Unknown phone stress variant: {variant_key}")


def _save_variant(image: Image.Image, output_path: Path, variant_key: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality = 30 if variant_key == "jpeg_compression" else 92
    image.save(output_path, format="JPEG", quality=quality, optimize=True)


def _center_crop_zoom(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(width * 0.78))
    crop_height = max(1, int(height * 0.78))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def _low_resolution(image: Image.Image) -> Image.Image:
    width, height = image.size
    low_width = max(16, width // 4)
    low_height = max(16, height // 4)
    low = image.resize((low_width, low_height), Image.Resampling.BILINEAR)
    return low.resize((width, height), Image.Resampling.NEAREST)

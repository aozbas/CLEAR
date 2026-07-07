"""Hugging Face image-classification adapter for evaluation-only runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

from ml.evaluation.schema import HAM10000_LABELS, ModelMetadata, ModelPrediction, validate_label


class HuggingFaceImageClassifierAdapter:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str = "main",
        label_map: dict[str, str],
        cache_dir: Path | None = None,
        license_name: str | None = None,
    ) -> None:
        if not label_map:
            raise ValueError("label_map must include at least one candidate label.")
        for canonical_label in label_map.values():
            validate_label(canonical_label)

        self.model_id = model_id
        self.revision = revision
        self.label_map = label_map
        self.cache_dir = cache_dir
        image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        model = AutoModelForImageClassification.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        self._classifier = pipeline(
            "image-classification",
            model=model,
            image_processor=image_processor,
        )
        self.metadata = ModelMetadata(
            name=model_id,
            source=f"https://huggingface.co/{model_id}",
            adapter="huggingface_image_classifier",
            revision=self._resolved_revision() or revision,
            license=license_name,
            labels=self._canonical_labels(),
            notes=["Hugging Face image classifier used for experimental classification."],
        )

    def predict_image(self, image_path: Path) -> ModelPrediction:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            started = time.perf_counter()
            raw_outputs = self._classifier(rgb_image, top_k=None)
            latency_ms = (time.perf_counter() - started) * 1000

        probabilities = self._map_outputs(raw_outputs)
        if not probabilities:
            raise ValueError(f"Model returned no classification scores: {self.model_id}")

        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        return ModelPrediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
        )

    def _canonical_labels(self) -> list[str]:
        labels = set(self.label_map.values())
        return [label for label in HAM10000_LABELS if label in labels]

    def _resolved_revision(self) -> str | None:
        model = getattr(self._classifier, "model", None)
        config = getattr(model, "config", None)
        for attribute in ("_commit_hash", "commit_hash"):
            value = getattr(config, attribute, None)
            if value:
                return str(value)
        return None

    def _map_outputs(self, raw_outputs: Any) -> dict[str, float]:
        outputs = raw_outputs
        if outputs and isinstance(outputs, list) and isinstance(outputs[0], list):
            outputs = outputs[0]

        probabilities: dict[str, float] = {}
        for item in outputs:
            raw_label = str(item["label"])
            if raw_label not in self.label_map:
                raise ValueError(
                    f"Model {self.model_id} returned unmapped label: {raw_label}"
                )
            canonical_label = self.label_map[raw_label]
            score = float(item["score"])
            probabilities[canonical_label] = probabilities.get(canonical_label, 0.0) + score
        return probabilities

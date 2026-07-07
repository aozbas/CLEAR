"""Adapter for CLEAR's local HAM10000 baseline checkpoint."""

from __future__ import annotations

import time
from pathlib import Path

from ml.evaluation.schema import HAM10000_LABELS, ModelMetadata, ModelPrediction
from ml.inference.predict import predict


class BaselineAdapter:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.metadata = ModelMetadata(
            name="baseline",
            source="local ml/inference/predict.py",
            adapter="baseline",
            revision=None,
            license=None,
            labels=list(HAM10000_LABELS),
            notes=["Local CLEAR HAM10000 checkpoint for experimental classification."],
        )

    def predict_image(self, image_path: Path) -> ModelPrediction:
        image_bytes = Path(image_path).read_bytes()
        started = time.perf_counter()
        result = predict(image_bytes, model_path=self.model_path)
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelPrediction(
            label=str(result["label"]),
            confidence=float(result["confidence"]),
            probabilities=result.get("probabilities"),
            latency_ms=latency_ms,
        )

"""Common model adapter protocol for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ml.evaluation.schema import ModelMetadata, ModelPrediction


class LesionModelAdapter(Protocol):
    metadata: ModelMetadata

    def predict_image(self, image_path: Path) -> ModelPrediction:
        """Run one image through a model and return a canonical prediction."""

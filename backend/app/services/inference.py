"""Thin wrapper around the ML inference layer."""
from typing import Any

from ..config import settings
from ml.inference.predict import InvalidImageError, predict


def predict_lesion(image_bytes: bytes) -> dict[str, Any]:
    return predict(image_bytes, model_path=settings.resolved_model_path)

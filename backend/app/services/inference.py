"""Thin wrapper around the ML inference layer."""

from typing import Any

from ml.inference.predict import InvalidImageError, predict

from ..config import settings

__all__ = ["InvalidImageError", "predict_lesion"]


def predict_lesion(image_bytes: bytes) -> dict[str, Any]:
    return predict(image_bytes, model_path=settings.resolved_model_path)

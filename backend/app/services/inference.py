"""Thin wrapper around the ML inference layer."""
from typing import Any


def predict_lesion(image_bytes: bytes) -> dict[str, Any]:
    # TODO: load model once, run preprocess + forward pass
    return {"label": "unknown", "confidence": 0.0}

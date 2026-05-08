"""Thin wrapper around the ML inference layer."""
from typing import Any

from ..config import settings

_MODEL = None  # lazy-loaded in Phase 1 from settings.model_path


def predict_lesion(image_bytes: bytes) -> dict[str, Any]:
    # TODO (Phase 1): if _MODEL is None, load checkpoint from settings.model_path,
    # apply ml.preprocessing.get_transforms("val"), run forward pass.
    _ = settings.model_path
    return {"label": "unknown", "confidence": 0.0}

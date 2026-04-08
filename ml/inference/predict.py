"""Inference helper used by the backend service layer."""
from io import BytesIO

from PIL import Image


def load_image(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def predict(image_bytes: bytes) -> dict:
    # TODO: load checkpoint once, apply transforms, run model
    _ = load_image(image_bytes)
    return {"label": "unknown", "confidence": 0.0}

"""Inference helper used by the backend service layer."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from ml.models.classifier import build_model
from ml.preprocessing import get_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "lesion_classifier_binary.pt"
DEFAULT_LABELS = ["non_suspicious", "suspicious"]

_MODEL: torch.nn.Module | None = None
_MODEL_PATH: Path | None = None
_DEVICE: torch.device | None = None
_MODEL_LABELS: list[str] | None = None


class InvalidImageError(ValueError):
    """Raised when Pillow cannot decode uploaded image bytes."""


def load_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Unsupported image format (only PNG and JPEG allowed)") from exc


def resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is None:
        return DEFAULT_MODEL_PATH
    path = Path(model_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device is None and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_checkpoint_labels(checkpoint: Any) -> list[str]:
    if not isinstance(checkpoint, dict):
        return DEFAULT_LABELS

    labels = checkpoint.get("labels", DEFAULT_LABELS)
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("Checkpoint labels must be a non-empty list of strings.")
    return labels


def load_model(model_path: str | Path | None = None, device: str | torch.device | None = None) -> torch.nn.Module:
    global _MODEL, _MODEL_PATH, _DEVICE, _MODEL_LABELS

    resolved_path = resolve_model_path(model_path).resolve()
    resolved_device = get_device(device)
    if _MODEL is not None and _MODEL_PATH == resolved_path and _DEVICE == resolved_device:
        return _MODEL

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {resolved_path}. "
            "Train it first with `python -m ml.training.train`."
        )

    checkpoint = torch.load(resolved_path, map_location=resolved_device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    labels = get_checkpoint_labels(checkpoint)

    model = build_model(num_classes=len(labels)).to(resolved_device)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL = model
    _MODEL_PATH = resolved_path
    _DEVICE = resolved_device
    _MODEL_LABELS = labels
    return model


def predict(
    image_bytes: bytes,
    model_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    resolved_device = get_device(device)
    model = load_model(model_path, resolved_device)
    image = load_image(image_bytes)
    tensor = get_transforms("val")(image).unsqueeze(0).to(resolved_device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    confidence, index = torch.max(probabilities, dim=0)
    labels = _MODEL_LABELS or DEFAULT_LABELS
    label = labels[int(index.item())]
    return {"label": label, "confidence": float(confidence.item())}

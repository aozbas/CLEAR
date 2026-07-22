"""Inference helper used by the backend service layer."""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from ml.models.classifier import build_model
from ml.preprocessing import get_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "ml" / "models" / "pad_hiba_convnext_tiny_source_balanced_final_seed42.pt"
)
PREPROCESSING = "resize_224_imagenet_normalization"
CURRENT_MODEL_VERSION = "pad-hiba-convnext-tiny-source-balanced-final-2026-07-22"
CURRENT_TRAINING_PROTOCOL = "pad_hiba_source_balanced_full_development_final_fit_v1"
CURRENT_SOURCE_CLASS_WEIGHTING = "equal_total_effective_weight_per_source_class_cell"
CURRENT_HIBA_VIEW_WEIGHTING = "equal_total_mass_per_lesion"
CURRENT_CV_SUMMARY_SHA256 = "20dec905c9470dc34e467d95354ee78b5affaaa14d8ef4d7f13dad7f96a7da53"
CURRENT_MANIFEST_FINGERPRINT = "23d3f41f18fc6d1082434fc049b6a7b7af07785df70b668bfb5ec51115747c5d"
CURRENT_MANIFEST_IDENTITY_FINGERPRINT = (
    "cdce4bb2e59f3ab462ba402fdb13f58caecc0fcfadda269e474a49bfec663828"
)
SUPPORTED_INPUT_GATE_PROTOCOL = "pad_hiba_open_images_supported_input_gate_v1"
SUPPORTED_INPUT_GATE_METHOD = "logsumexp"
SUPPORTED_INPUT_GATE_THRESHOLD = 4.4970903396606445
SUPPORTED_INPUT_GATE_COHORT_FINGERPRINT = (
    "fe5cfd2dc03a79a40eed07fc2b7cc79e28e176a1a5e82d72dc0e572ab56ee1b2"
)
SUPPORTED_INPUT_GATE_REPORT_SHA256 = (
    "79f53e4ff3c76f56d3375aee38c728a739220e8194e5c2b0bbc3f278e621e6ee"
)
SUPPORTED_INPUT_GATE_VERSION = "pad-hiba-open-images-supported-input-gate-v1-79f53e4ff3c76f56"
HAM10000_LABELS = [
    "melanoma",
    "nevus",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
]
PAD_HIBA_LABELS = [
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "melanoma",
    "nevus",
    "squamous_cell_carcinoma",
    "seborrheic_keratosis",
]
DEFAULT_LABELS = HAM10000_LABELS

_MODEL: torch.nn.Module | None = None
_MODEL_PATH: Path | None = None
_DEVICE: torch.device | None = None
_MODEL_LABELS: list[str] | None = None
_MODEL_PREPROCESSING: str | None = None


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
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) for label in labels)
    ):
        raise ValueError("Checkpoint labels must be a non-empty list of strings.")
    return labels


def get_checkpoint_architecture(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        return "resnet18"
    architecture = checkpoint.get("architecture", "resnet18")
    if architecture not in {"resnet18", "convnext_tiny"}:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")
    return architecture


def get_checkpoint_preprocessing(checkpoint: Any, *, architecture: str) -> str:
    if not isinstance(checkpoint, dict):
        return PREPROCESSING
    preprocessing = checkpoint.get("preprocessing", PREPROCESSING)
    if preprocessing != PREPROCESSING:
        raise ValueError(f"Unsupported checkpoint preprocessing: {preprocessing!r}")
    if architecture == "convnext_tiny":
        labels = get_checkpoint_labels(checkpoint)
        if labels != PAD_HIBA_LABELS:
            raise ValueError("ConvNeXt-Tiny checkpoint labels do not match the demo label order.")
        expected_metadata = {
            "dataset": "pad_ufes_hiba",
            "dataset_role": "multisource_development_final_fit",
            "training_protocol": CURRENT_TRAINING_PROTOCOL,
            "model_version": CURRENT_MODEL_VERSION,
            "sources": ["pad_ufes", "hiba"],
            "source_class_weighting": CURRENT_SOURCE_CLASS_WEIGHTING,
            "hiba_view_weighting": CURRENT_HIBA_VIEW_WEIGHTING,
            "manifest_fingerprint": CURRENT_MANIFEST_FINGERPRINT,
            "manifest_identity_fingerprint": CURRENT_MANIFEST_IDENTITY_FINGERPRINT,
            "cv_summary_sha256": CURRENT_CV_SUMMARY_SHA256,
            "cv_decision_all_pass": False,
            "selection_status": "owner_selected_despite_failed_preregistered_gates",
            "pretrained_weights": "imagenet",
            "pretrained_weights_id": "IMAGENET1K_V1",
            "epoch": 11,
            "seed": 42,
            "source_total_raw_image_counts": {"pad_ufes": 2_298, "hiba": 309},
            "source_total_effective_unit_counts": {"pad_ufes": 2_298.0, "hiba": 308.0},
            "hyperparameters": {
                "epochs": 11,
                "epoch_rule": "median_of_locked_cv_selected_epochs",
                "locked_cv_selected_epochs": [15, 8, 10, 11, 13],
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "optimizer": "AdamW",
                "schedule": "none",
                "augmentation_profile": "baseline",
                "label_smoothing": 0.0,
                "sampling": "random_shuffle_without_replacement",
                "source_class_weighting": CURRENT_SOURCE_CLASS_WEIGHTING,
                "hiba_view_weighting": CURRENT_HIBA_VIEW_WEIGHTING,
            },
        }
        mismatches = [
            key for key, value in expected_metadata.items() if checkpoint.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "ConvNeXt-Tiny checkpoint metadata does not match the approved demo contract: "
                + ", ".join(mismatches)
            )
    return preprocessing


def load_model(
    model_path: str | Path | None = None, device: str | torch.device | None = None
) -> torch.nn.Module:
    global _MODEL, _MODEL_PATH, _DEVICE, _MODEL_LABELS, _MODEL_PREPROCESSING

    resolved_path = resolve_model_path(model_path).resolve()
    resolved_device = get_device(device)
    if _MODEL is not None and _MODEL_PATH == resolved_path and _DEVICE == resolved_device:
        return _MODEL

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {resolved_path}. "
            "Provision the owner-approved experimental checkpoint before starting the backend."
        )

    checkpoint = torch.load(resolved_path, map_location=resolved_device, weights_only=True)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    labels = get_checkpoint_labels(checkpoint)
    architecture = get_checkpoint_architecture(checkpoint)
    preprocessing = get_checkpoint_preprocessing(checkpoint, architecture=architecture)

    model = build_model(num_classes=len(labels), architecture=architecture).to(resolved_device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model
    _MODEL_PATH = resolved_path
    _DEVICE = resolved_device
    _MODEL_LABELS = labels
    _MODEL_PREPROCESSING = preprocessing
    return model


def supported_input_score(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("Supported-input scoring requires a non-empty logits batch.")
    if SUPPORTED_INPUT_GATE_METHOD != "logsumexp":
        raise RuntimeError("The configured supported-input gate method is unavailable.")
    return torch.logsumexp(logits, dim=1)


def predict(
    image_bytes: bytes,
    model_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    resolved_device = get_device(device)
    model = load_model(model_path, resolved_device)
    image = load_image(image_bytes)
    if _MODEL_PREPROCESSING != PREPROCESSING:
        raise RuntimeError("The loaded checkpoint preprocessing is unavailable.")
    tensor = get_transforms("val")(image).unsqueeze(0).to(resolved_device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        gate_score = float(supported_input_score(logits).squeeze(0).item())

    input_supported = math.isfinite(gate_score) and gate_score >= SUPPORTED_INPUT_GATE_THRESHOLD
    if not input_supported:
        return {
            "label": None,
            "confidence": None,
            "input_supported": False,
            "input_gate_version": SUPPORTED_INPUT_GATE_VERSION,
        }

    confidence, index = torch.max(probabilities, dim=0)
    labels = _MODEL_LABELS or DEFAULT_LABELS
    label = labels[int(index.item())]
    return {
        "label": label,
        "confidence": float(confidence.item()),
        "input_supported": True,
        "input_gate_version": SUPPORTED_INPUT_GATE_VERSION,
    }

"""Shared schema objects for experimental model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HAM10000_LABELS = (
    "melanoma",
    "nevus",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
)
VALID_SPLITS = ("train", "val", "test")


def validate_label(label: str) -> None:
    if label not in HAM10000_LABELS:
        raise ValueError(f"Unknown HAM10000 label: {label}")


def validate_probability(value: float, *, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class EvaluationExample:
    image_path: Path
    label: str
    split: str

    def __post_init__(self) -> None:
        validate_label(self.label)
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {self.split}")


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float] | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        validate_label(self.label)
        validate_probability(self.confidence, field_name="confidence")

        if self.probabilities is not None:
            for label, probability in self.probabilities.items():
                validate_label(label)
                validate_probability(probability, field_name=f"probability for {label}")

        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative.")


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    source: str
    adapter: str
    revision: str | None
    license: str | None
    labels: list[str]
    notes: list[str]

    def __post_init__(self) -> None:
        for label in self.labels:
            validate_label(label)

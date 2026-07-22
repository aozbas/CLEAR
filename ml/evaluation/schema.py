"""Shared schema objects for experimental model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PAD_UFES_NATIVE_LABELS = (
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "melanoma",
    "nevus",
    "squamous_cell_carcinoma",
    "seborrheic_keratosis",
)
LABEL_SETS = {
    "pad_ufes_native": PAD_UFES_NATIVE_LABELS,
}
KNOWN_LABELS = PAD_UFES_NATIVE_LABELS
VALID_SPLITS = ("train", "val", "test")


def labels_for_set(name: str) -> tuple[str, ...]:
    try:
        return LABEL_SETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown label set: {name}") from exc


def validate_label(label: str, *, labels: tuple[str, ...] | None = None) -> None:
    allowed_labels = labels or KNOWN_LABELS
    if label not in allowed_labels:
        raise ValueError(f"Unknown label: {label}")


def validate_probability(value: float, *, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class EvaluationExample:
    image_path: Path
    label: str
    split: str
    labels: tuple[str, ...] = KNOWN_LABELS

    def __post_init__(self) -> None:
        validate_label(self.label, labels=self.labels)
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {self.split}")


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float] | None = None
    latency_ms: float | None = None
    labels: tuple[str, ...] = KNOWN_LABELS

    def __post_init__(self) -> None:
        validate_label(self.label, labels=self.labels)
        validate_probability(self.confidence, field_name="confidence")

        if self.probabilities is not None:
            for label, probability in self.probabilities.items():
                validate_label(label, labels=self.labels)
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

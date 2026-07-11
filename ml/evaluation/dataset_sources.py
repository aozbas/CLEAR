"""Evaluation dataset source metadata and contamination checks."""

from __future__ import annotations

from dataclasses import dataclass

from ml.evaluation.schema import HAM10000_LABELS, validate_label


@dataclass(frozen=True)
class DatasetSource:
    key: str
    name: str
    source_url: str
    split_type: str
    labels: list[str]
    partial_label_set: bool
    known_training_datasets: list[str]
    notes: list[str]

    def __post_init__(self) -> None:
        for label in self.labels:
            validate_label(label)


DATASET_SOURCES: dict[str, DatasetSource] = {
    "ham10000_internal": DatasetSource(
        key="ham10000_internal",
        name="HAM10000 internal lesion-grouped test split",
        source_url="https://challenge.isic-archive.com/data/",
        split_type="internal_test",
        labels=list(HAM10000_LABELS),
        partial_label_set=False,
        known_training_datasets=["HAM10000"],
        notes=[
            "Fair for CLEAR baseline trained on CLEAR's train split.",
            "Not a clean external holdout for candidates trained on HAM10000.",
        ],
    ),
    "ph2_holdout": DatasetSource(
        key="ph2_holdout",
        name="PH2 external holdout",
        source_url="https://www.fc.up.pt/addi/ph2%20database.html",
        split_type="external_holdout",
        labels=["melanoma", "nevus"],
        partial_label_set=True,
        known_training_datasets=["PH2"],
        notes=[
            "External dermoscopy holdout with melanoma and nevus-like labels only.",
            "Use for leakage-resistant melanoma/nevus checks, not full seven-class scoring.",
        ],
    ),
    "derm7pt_holdout": DatasetSource(
        key="derm7pt_holdout",
        name="Derm7pt external holdout",
        source_url="https://github.com/jeremykawahara/derm7pt",
        split_type="external_holdout",
        labels=[
            "melanoma",
            "nevus",
            "basal_cell_carcinoma",
            "benign_keratosis",
            "dermatofibroma",
            "vascular_lesion",
        ],
        partial_label_set=True,
        known_training_datasets=["Derm7pt"],
        notes=[
            "External dermoscopy holdout prepared from the official Derm7pt release layout.",
            "Rows with diagnoses outside CLEAR's current label set are skipped.",
            "Use the official test split for benchmark comparisons.",
        ],
    ),
    "pad_ufes_clinical": DatasetSource(
        key="pad_ufes_clinical",
        name="PAD-UFES-20 clinical smartphone overlap-label split",
        source_url="https://data.mendeley.com/datasets/zr7vgbcyr2/1",
        split_type="clinical_phone_holdout",
        labels=[
            "melanoma",
            "nevus",
            "basal_cell_carcinoma",
            "actinic_keratosis",
        ],
        partial_label_set=True,
        known_training_datasets=["PAD-UFES-20"],
        notes=[
            "Clinical images collected with smartphone devices.",
            "Only CLEAR current-label overlap rows are included.",
            "SCC, Bowen's disease, and separate seborrheic keratosis are deferred.",
        ],
    ),
    "scin_user_submitted": DatasetSource(
        key="scin_user_submitted",
        name="SCIN user-submitted dermatology optional robustness split",
        source_url="https://github.com/google-research-datasets/scin",
        split_type="clinical_user_submitted_optional",
        labels=list(HAM10000_LABELS),
        partial_label_set=True,
        known_training_datasets=["SCIN"],
        notes=[
            "User-submitted dermatology images with dermatologist differential labels.",
            "Use for robustness and image-quality checks, not as biopsy-proven cancer truth.",
        ],
    ),
    "ddi_clinical": DatasetSource(
        key="ddi_clinical",
        name="Diverse Dermatology Images optional fairness split",
        source_url="https://ddi-dataset.github.io/",
        split_type="clinical_fairness_optional",
        labels=list(HAM10000_LABELS),
        partial_label_set=True,
        known_training_datasets=["DDI"],
        notes=[
            "Biopsy-proven clinical images with diverse skin-tone representation.",
            "Access and use terms restrict redistribution; keep local artifacts private.",
        ],
    ),
}


def get_dataset_source(key: str) -> DatasetSource:
    try:
        return DATASET_SOURCES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset source: {key}") from exc


def contamination_notes(
    *,
    dataset_source: DatasetSource,
    model_datasets: list[str],
) -> list[str]:
    source_datasets = {name.lower() for name in dataset_source.known_training_datasets}
    model_dataset_names = {name.lower() for name in model_datasets}
    overlap = sorted(source_datasets.intersection(model_dataset_names))
    if not overlap:
        return []

    overlap_text = ", ".join(overlap)
    return [
        "Known dataset overlap: "
        f"{dataset_source.name} overlaps with model training dataset(s): {overlap_text}. "
        "Treat metrics as possible train/test overlap, not external generalization."
    ]

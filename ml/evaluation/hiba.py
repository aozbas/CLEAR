"""Evaluate frozen PAD-UFES-native checkpoints on public HIBA clinical photographs.

The primary protocol is a six-class lesion-level evaluation over exact-mapped clinical images from
the Hospital Italiano de Buenos Aires collection. Dermoscopy and diagnoses outside CLEAR's
PAD-UFES-native taxonomy are excluded before model access.

Outputs contain aggregate experimental metrics only. They are not medical conclusions and must
remain on ignored paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.metrics import summarize_metrics
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.preprocessing import get_pad_ufes_transforms
from ml.training.train_pad_ufes import ARCHITECTURES, build_transfer_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "hiba"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "ml" / "models" / "pad_ufes_convnext_tiny_cv_seed42"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml" / "runs" / "evaluation" / "hiba-convnext-tiny-cv"
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_CALIBRATION_BINS = 10
DEFAULT_ARTIFACT_SAMPLES_PER_CLASS = 6
EXPECTED_PREPROCESSING = "resize_224_imagenet_normalization"
OFFICIAL_METADATA_SHA256 = "77060346ee2df2b691b455ad0a7060ae9ef8cdb513573de84b90ce4829cecce0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
INTERVAL_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "multiclass_brier_score",
    "top_label_ece",
)

REQUIRED_METADATA_COLUMNS = {
    "isic_id",
    "attribution",
    "copyright_license",
    "diagnosis_3",
    "diagnosis_confirm_type",
    "fitzpatrick_skin_type",
    "image_type",
    "lesion_id",
    "patient_id",
}
DIAGNOSIS_TO_PAD_UFES = {
    "Solar or actinic keratosis": "actinic_keratosis",
    "Basal cell carcinoma": "basal_cell_carcinoma",
    "Melanoma, NOS": "melanoma",
    "Nevus": "nevus",
    "Squamous cell carcinoma, NOS": "squamous_cell_carcinoma",
    "Seborrheic keratosis": "seborrheic_keratosis",
}
HIBA_CAPTURE_CONTEXT = {
    "source": "retrospective_hospital_dermatology_collection_argentina",
    "image_type": "clinical_smartphone_photographs",
    "capture_setting": "dermatology_clinic",
    "photographer": "dermatology_professional",
    "device_detail": "professional_respective_smartphone_models_not_released",
    "patient_taken_validation": False,
    "consumer_capture_validation": False,
    "dermoscopy_scored": False,
}
WORKFLOW_ARTIFACT_CONTEXT = {
    "pre_score_audit_status": "not_performed",
    "post_score_review": "deterministic_prediction_blind_sample_only",
    "not_used_for": ["filtering", "cropping", "rescoring", "tuning", "model_selection"],
}
EXPECTED_FROZEN_AUDIT = {
    "source_row_count": 1_616,
    "source_image_file_count": 1_616,
    "clinical_image_count": 346,
    "clinical_patient_count": 244,
    "clinical_lesion_count": 345,
    "primary_image_count": 309,
    "primary_lesion_count": 308,
    "primary_patient_count": 225,
    "primary_image_type_support": {"clinical: close-up": 6, "clinical: overview": 303},
    "primary_image_class_support": {
        "actinic_keratosis": 17,
        "basal_cell_carcinoma": 112,
        "melanoma": 59,
        "nevus": 53,
        "squamous_cell_carcinoma": 47,
        "seborrheic_keratosis": 21,
    },
    "primary_lesion_class_support": {
        "actinic_keratosis": 17,
        "basal_cell_carcinoma": 112,
        "melanoma": 58,
        "nevus": 53,
        "squamous_cell_carcinoma": 47,
        "seborrheic_keratosis": 21,
    },
    "primary_histopathology_image_count": 274,
    "primary_histopathology_lesion_count": 273,
    "primary_fitzpatrick_lesion_support": {
        "FST_I": 15,
        "FST_II": 258,
        "FST_III": 25,
        "missing": 10,
    },
    "repeated_lesion_count": 1,
    "excluded_clinical_diagnosis_support": {
        "<blank>": 10,
        "Dermatofibroma": 22,
        "Lichen planus like keratosis": 1,
        "Solar lentigo": 4,
    },
}


@dataclass(frozen=True)
class PreparedHiba:
    metadata_rows: pd.DataFrame
    clinical_rows: pd.DataFrame
    image_rows: pd.DataFrame
    lesion_rows: pd.DataFrame
    audit: dict[str, object]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen PAD-UFES-native checkpoints on HIBA clinical photographs."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="convnext_tiny")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--calibration-bins", type=int, default=DEFAULT_CALIBRATION_BINS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--artifact-contact-sheet", type=Path)
    parser.add_argument(
        "--artifact-samples-per-class",
        type=int,
        default=DEFAULT_ARTIFACT_SAMPLES_PER_CLASS,
    )
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_official_metadata(
    data_dir: Path,
    *,
    expected_sha256: str | None = OFFICIAL_METADATA_SHA256,
) -> tuple[Path, str]:
    data_dir = resolve_project_path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing HIBA data directory: {data_dir}")
    candidates = [
        path for path in data_dir.rglob("*.csv") if path.is_file() and not path.is_symlink()
    ]
    if not candidates:
        raise FileNotFoundError(f"No HIBA metadata CSV found under {data_dir}.")
    hashes = {path: _sha256(path) for path in candidates}
    if expected_sha256 is None:
        if len(candidates) != 1:
            raise ValueError("Synthetic HIBA source must contain exactly one metadata CSV.")
        path = candidates[0]
        return path, hashes[path]
    matches = [path for path, digest in hashes.items() if digest == expected_sha256]
    if len(matches) != 1:
        raise ValueError("HIBA official metadata SHA-256 did not match exactly one source CSV.")
    path = matches[0]
    return path, hashes[path]


def load_source(
    data_dir: Path,
    *,
    expected_metadata_sha256: str | None = OFFICIAL_METADATA_SHA256,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, str]]:
    """Load metadata and require one image file for every official ISIC ID."""
    data_dir = resolve_project_path(data_dir)
    metadata_path, metadata_sha256 = find_official_metadata(
        data_dir,
        expected_sha256=expected_metadata_sha256,
    )
    rows = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    missing = REQUIRED_METADATA_COLUMNS.difference(rows.columns)
    if missing:
        raise ValueError(f"HIBA metadata is missing columns: {', '.join(sorted(missing))}")
    if rows.empty:
        raise ValueError("HIBA metadata must not be empty.")
    for column in REQUIRED_METADATA_COLUMNS:
        rows[column] = rows[column].str.strip()
    if bool((rows["isic_id"] == "").any()):
        raise ValueError("HIBA metadata contains a blank isic_id.")
    if bool(rows["isic_id"].str.lower().duplicated().any()):
        raise ValueError("HIBA metadata contains duplicate isic_id values.")

    image_paths: dict[str, Path] = {}
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError("HIBA image source contains a symbolic link.")
        image_id = path.stem.lower()
        if image_id in image_paths:
            raise ValueError(f"HIBA source contains duplicate image ID: {path.stem}")
        image_paths[image_id] = path.resolve()

    expected_ids = set(rows["isic_id"].str.lower())
    actual_ids = set(image_paths)
    if expected_ids != actual_ids:
        raise ValueError(
            "HIBA image files do not match metadata: "
            f"missing={len(expected_ids - actual_ids)} extra={len(actual_ids - expected_ids)}."
        )
    if set(rows["copyright_license"]) != {"CC-BY"}:
        raise ValueError("HIBA metadata license drifted from CC-BY.")
    if set(rows["attribution"]) != {"Hospital Italiano de Buenos Aires"}:
        raise ValueError("HIBA metadata attribution drifted.")
    return rows.reset_index(drop=True), image_paths, {"metadata_csv": metadata_sha256}


def prepare_primary_cohort(
    metadata_rows: pd.DataFrame,
    image_paths: Mapping[str, Path],
    *,
    source_hashes: Mapping[str, str],
) -> PreparedHiba:
    clinical = metadata_rows[metadata_rows["image_type"].str.startswith("clinical:")].copy()
    clinical["label"] = clinical["diagnosis_3"].map(DIAGNOSIS_TO_PAD_UFES)
    scored = clinical[clinical["label"].notna()].copy()
    if scored.empty:
        raise ValueError("HIBA exact-mapped clinical cohort is empty.")
    for column in ("patient_id", "lesion_id"):
        if bool((scored[column] == "").any()):
            raise ValueError(f"HIBA scored cohort contains a blank {column}.")
    scored["image_path"] = scored["isic_id"].str.lower().map(image_paths)
    if bool(scored["image_path"].isna().any()):
        raise RuntimeError("HIBA scored images were lost after source validation.")
    scored = scored.sort_values("isic_id").reset_index(drop=True)
    scored["image_index"] = np.arange(len(scored), dtype=int)

    lesion_records = []
    for lesion_index, (lesion_id, group) in enumerate(
        scored.groupby("lesion_id", sort=True, dropna=False)
    ):
        patient_ids = set(group["patient_id"])
        labels = set(group["label"])
        fst_values = set(group["fitzpatrick_skin_type"])
        if len(patient_ids) != 1:
            raise ValueError("HIBA lesion maps to multiple patient IDs.")
        if len(labels) != 1:
            raise ValueError("HIBA lesion maps to multiple exact labels.")
        if len(fst_values) != 1:
            raise ValueError("HIBA lesion has inconsistent Fitzpatrick values.")
        lesion_records.append(
            {
                "lesion_index": lesion_index,
                "lesion_id": lesion_id,
                "patient_id": next(iter(patient_ids)),
                "label": next(iter(labels)),
                "fitzpatrick_group": _fitzpatrick_group(next(iter(fst_values))),
                "image_count": len(group),
                "histopathology_confirmed": bool(
                    (group["diagnosis_confirm_type"] == "histopathology").all()
                ),
            }
        )
    lesion_rows = pd.DataFrame.from_records(lesion_records)
    lesion_index_by_id = dict(
        zip(lesion_rows["lesion_id"], lesion_rows["lesion_index"], strict=True)
    )
    scored["lesion_index"] = scored["lesion_id"].map(lesion_index_by_id)
    image_rows = scored[
        [
            "image_index",
            "image_path",
            "isic_id",
            "lesion_index",
            "lesion_id",
            "patient_id",
            "label",
            "image_type",
            "diagnosis_confirm_type",
            "fitzpatrick_skin_type",
        ]
    ].copy()
    audit = build_dataset_audit(
        metadata_rows,
        clinical,
        image_rows,
        lesion_rows,
        source_hashes=source_hashes,
        source_image_file_count=len(image_paths),
    )
    return PreparedHiba(
        metadata_rows=metadata_rows,
        clinical_rows=clinical,
        image_rows=image_rows,
        lesion_rows=lesion_rows,
        audit=audit,
    )


def _fitzpatrick_group(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        return "missing"
    if normalized in {"I", "II", "III", "IV", "V", "VI"}:
        return f"FST_{normalized}"
    return "unknown_or_inconsistent"


def build_dataset_audit(
    metadata_rows: pd.DataFrame,
    clinical_rows: pd.DataFrame,
    image_rows: pd.DataFrame,
    lesion_rows: pd.DataFrame,
    *,
    source_hashes: Mapping[str, str],
    source_image_file_count: int,
) -> dict[str, object]:
    excluded = clinical_rows[clinical_rows["label"].isna()]["diagnosis_3"].map(
        lambda value: value if value else "<blank>"
    )
    image_class_support = Counter(image_rows["label"])
    lesion_class_support = Counter(lesion_rows["label"])
    fst_support = Counter(lesion_rows["fitzpatrick_group"])
    return {
        "dataset": "Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022)",
        "isic_collection": 251,
        "dataset_doi": "10.34970/587329",
        "study_doi": "10.1038/s41597-023-02630-0",
        "license": "CC-BY",
        "source_sha256": dict(source_hashes),
        "source_row_count": len(metadata_rows),
        "source_image_file_count": source_image_file_count,
        "clinical_image_count": len(clinical_rows),
        "clinical_patient_count": int(clinical_rows["patient_id"].nunique()),
        "clinical_lesion_count": int(clinical_rows["lesion_id"].nunique()),
        "diagnosis_mapping": dict(DIAGNOSIS_TO_PAD_UFES),
        "primary_image_count": len(image_rows),
        "primary_lesion_count": len(lesion_rows),
        "primary_patient_count": int(lesion_rows["patient_id"].nunique()),
        "primary_image_type_support": {
            key: int(value) for key, value in sorted(Counter(image_rows["image_type"]).items())
        },
        "primary_image_class_support": {
            label: int(image_class_support.get(label, 0)) for label in PAD_UFES_NATIVE_LABELS
        },
        "primary_lesion_class_support": {
            label: int(lesion_class_support.get(label, 0)) for label in PAD_UFES_NATIVE_LABELS
        },
        "primary_histopathology_image_count": int(
            (image_rows["diagnosis_confirm_type"] == "histopathology").sum()
        ),
        "primary_histopathology_lesion_count": int(lesion_rows["histopathology_confirmed"].sum()),
        "primary_fitzpatrick_lesion_support": {
            key: int(value) for key, value in sorted(fst_support.items())
        },
        "repeated_lesion_count": int((lesion_rows["image_count"] > 1).sum()),
        "excluded_clinical_diagnosis_support": {
            key: int(value) for key, value in sorted(Counter(excluded).items())
        },
        "capture_context": HIBA_CAPTURE_CONTEXT,
        "workflow_artifact_context": WORKFLOW_ARTIFACT_CONTEXT,
        "demographic_performance_reporting": (
            "withheld_no_FST_IV_V_VI_and_insufficient_balanced_subgroup_support"
        ),
        "report_scope": "aggregate_counts_only",
    }


def validate_frozen_audit(audit: Mapping[str, object]) -> None:
    mismatches = []
    for key, expected in EXPECTED_FROZEN_AUDIT.items():
        observed = audit.get(key)
        if observed != expected:
            mismatches.append(f"{key}={observed!r}")
    if mismatches:
        raise ValueError("HIBA frozen cohort audit drifted: " + ", ".join(mismatches))


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, object],
    *,
    architecture: str,
    seed: int,
) -> None:
    expected = {
        "architecture": architecture,
        "input_mode": "image_only",
        "dataset": "pad_ufes",
        "label_set": "pad_ufes_native",
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "preprocessing": EXPECTED_PREPROCESSING,
        "seed": seed,
    }
    mismatches = [
        f"{key}={checkpoint.get(key)!r}"
        for key, expected_value in expected.items()
        if checkpoint.get(key) != expected_value
    ]
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        mismatches.append("model_state_dict is missing")
    if mismatches:
        raise ValueError(
            "Checkpoint does not match the frozen PAD-UFES-native protocol: "
            + ", ".join(mismatches)
        )


def discover_checkpoint_paths(checkpoints_dir: Path, *, folds: int) -> list[Path]:
    checkpoints_dir = resolve_project_path(checkpoints_dir)
    if folds < 1:
        raise ValueError("folds must be positive.")
    paths = [checkpoints_dir / f"fold_{fold_index}.pt" for fold_index in range(folds)]
    missing_count = sum(not path.is_file() for path in paths)
    if missing_count:
        raise FileNotFoundError(
            f"Missing {missing_count} of {folds} frozen fold checkpoints in {checkpoints_dir}."
        )
    return paths


class HibaImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame) -> None:
        self.image_paths = [Path(path) for path in rows["image_path"]]
        self.transform = get_pad_ufes_transforms("val")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[index]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, index


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _predict_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    *,
    sample_count: int,
    architecture: str,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {checkpoint_path}")
    validate_checkpoint_metadata(checkpoint, architecture=architecture, seed=seed)

    model = build_transfer_model(architecture=architecture, weights="none")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    probabilities = np.empty((sample_count, len(PAD_UFES_NATIVE_LABELS)), dtype=np.float64)
    seen = np.zeros(sample_count, dtype=bool)
    with torch.inference_mode():
        for images, indices in loader:
            logits = model(images.to(device, non_blocking=True))
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            index_array = indices.numpy()
            probabilities[index_array] = batch_probabilities
            seen[index_array] = True
    if not seen.all():
        raise RuntimeError("Inference did not produce probabilities for every HIBA image.")

    del model
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities


def _validated_probability_matrix(probabilities: np.ndarray, *, expected_rows: int) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    expected_shape = (expected_rows, len(PAD_UFES_NATIVE_LABELS))
    if array.shape != expected_shape:
        raise ValueError(f"probabilities must have shape {expected_shape}.")
    if not np.isfinite(array).all():
        raise ValueError("probabilities must be finite.")
    if (array < -1e-7).any() or (array > 1.0 + 1e-7).any():
        raise ValueError("probabilities must be between zero and one.")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each probability row must sum to one.")
    return array


def aggregate_lesion_probabilities(
    probabilities: np.ndarray,
    image_rows: pd.DataFrame,
    lesion_rows: pd.DataFrame,
) -> np.ndarray:
    array = _validated_probability_matrix(probabilities, expected_rows=len(image_rows))
    aggregated = np.empty((len(lesion_rows), len(PAD_UFES_NATIVE_LABELS)), dtype=np.float64)
    image_lesion_indices = image_rows["lesion_index"].to_numpy()
    for lesion in lesion_rows.itertuples(index=False):
        mask = image_lesion_indices == lesion.lesion_index
        if not mask.any():
            raise ValueError("HIBA lesion is missing all source images.")
        aggregated[lesion.lesion_index] = array[mask].mean(axis=0)
    return aggregated


def multiclass_metrics(truth: Sequence[str], probabilities: np.ndarray) -> dict[str, object]:
    truth_values = list(truth)
    array = _validated_probability_matrix(probabilities, expected_rows=len(truth_values))
    predictions = [PAD_UFES_NATIVE_LABELS[index] for index in np.argmax(array, axis=1)]
    return summarize_metrics(truth_values, predictions, labels=PAD_UFES_NATIVE_LABELS)


def calibration_metrics(
    truth: Sequence[str],
    probabilities: np.ndarray,
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, object]:
    truth_values = list(truth)
    array = _validated_probability_matrix(probabilities, expected_rows=len(truth_values))
    if bins < 2:
        raise ValueError("calibration bins must be at least two.")
    label_to_index = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
    try:
        targets = np.asarray([label_to_index[label] for label in truth_values], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unknown HIBA calibration label: {exc.args[0]}") from exc
    row_indices = np.arange(len(targets))
    clipped_true_probabilities = np.clip(array[row_indices, targets], 1e-12, 1.0)
    one_hot = np.eye(len(PAD_UFES_NATIVE_LABELS), dtype=np.float64)[targets]
    predicted = np.argmax(array, axis=1)
    confidence = np.max(array, axis=1)
    correct = predicted == targets
    bin_indices = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability_bins = []
    ece = 0.0
    for bin_index in range(bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if count:
            accuracy = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            contribution = count / len(targets) * abs(accuracy - mean_confidence)
            ece += contribution
        else:
            accuracy = None
            mean_confidence = None
            contribution = 0.0
        reliability_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "ece_contribution": contribution,
            }
        )
    return {
        "negative_log_likelihood": float(-np.log(clipped_true_probabilities).mean()),
        "multiclass_brier_score": float(np.square(array - one_hot).sum(axis=1).mean()),
        "top_label_ece": float(ece),
        "bin_count": bins,
        "reliability_bins": reliability_bins,
        "calibration_fitted": False,
    }


def patient_cluster_bootstrap_intervals(
    truth: Sequence[str],
    probabilities: np.ndarray,
    patient_ids: Sequence[str],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, object]:
    truth_values = np.asarray(list(truth), dtype=object)
    array = _validated_probability_matrix(probabilities, expected_rows=len(truth_values))
    patient_values = np.asarray(list(patient_ids), dtype=object)
    if patient_values.ndim != 1 or len(patient_values) != len(truth_values):
        raise ValueError("patient_ids must align with truth and probabilities.")
    if any(not str(value).strip() for value in patient_values):
        raise ValueError("patient_ids must not contain blanks.")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100.")
    if set(truth_values) != set(PAD_UFES_NATIVE_LABELS):
        raise ValueError("patient-clustered intervals require all six PAD-UFES labels.")

    predictions = np.asarray(
        [PAD_UFES_NATIVE_LABELS[index] for index in np.argmax(array, axis=1)], dtype=object
    )
    clusters = sorted(set(patient_values.tolist()))
    cluster_indices = {cluster: np.flatnonzero(patient_values == cluster) for cluster in clusters}
    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in INTERVAL_METRICS}
    per_class_recall = {label: [] for label in PAD_UFES_NATIVE_LABELS}
    per_class_f1 = {label: [] for label in PAD_UFES_NATIVE_LABELS}
    for _ in range(samples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in selected])
        metrics = summarize_metrics(
            truth_values[indices].tolist(),
            predictions[indices].tolist(),
            labels=PAD_UFES_NATIVE_LABELS,
        )
        calibration = calibration_metrics(
            truth_values[indices].tolist(),
            array[indices],
            bins=calibration_bins,
        )
        distributions["accuracy"].append(float(metrics["accuracy"]))
        distributions["balanced_accuracy"].append(float(metrics["balanced_accuracy"]))
        distributions["macro_f1"].append(float(metrics["macro_f1"]))
        for metric in (
            "negative_log_likelihood",
            "multiclass_brier_score",
            "top_label_ece",
        ):
            distributions[metric].append(float(calibration[metric]))
        for label in PAD_UFES_NATIVE_LABELS:
            per_class_recall[label].append(float(metrics["per_class"][label]["recall"]))
            per_class_f1[label].append(float(metrics["per_class"][label]["f1"]))

    return {
        "method": "hiba_patient_cluster_percentile_bootstrap",
        "sampling_unit": "released_patient_id",
        "patient_count": len(clusters),
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "intervals": {
            metric: _percentile_interval(values) for metric, values in distributions.items()
        },
        "per_class_recall_intervals": {
            label: _percentile_interval(values) for label, values in per_class_recall.items()
        },
        "per_class_f1_intervals": {
            label: _percentile_interval(values) for label, values in per_class_f1.items()
        },
    }


def _percentile_interval(values: Sequence[float]) -> dict[str, float]:
    return {
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def _majority_reference(truth: Sequence[str]) -> dict[str, object]:
    support = Counter(truth)
    majority = max(
        PAD_UFES_NATIVE_LABELS,
        key=lambda label: (support[label], -PAD_UFES_NATIVE_LABELS.index(label)),
    )
    predictions = [majority] * len(truth)
    return {
        "description": (
            "Trivial prevalence reference; predicts the most common class for every lesion."
        ),
        "predicted_label": majority,
        "metrics": summarize_metrics(list(truth), predictions, labels=PAD_UFES_NATIVE_LABELS),
    }


def select_artifact_review_rows(
    image_rows: pd.DataFrame,
    *,
    samples_per_class: int = DEFAULT_ARTIFACT_SAMPLES_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    if samples_per_class < 1:
        raise ValueError("artifact samples per class must be positive.")
    selected = []
    for label in PAD_UFES_NATIVE_LABELS:
        candidates = image_rows[image_rows["label"] == label].copy()
        if len(candidates) < samples_per_class:
            raise ValueError(f"HIBA artifact sample lacks {samples_per_class} rows for {label}.")
        candidates["selection_hash"] = candidates["isic_id"].map(
            lambda image_id: hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
        )
        selected.append(
            candidates.sort_values(["selection_hash", "isic_id"]).head(samples_per_class)
        )
    return pd.concat(selected, ignore_index=True)


def write_artifact_contact_sheet(
    rows: pd.DataFrame,
    path: Path,
    *,
    samples_per_class: int,
) -> None:
    tile_size = 240
    caption_height = 34
    canvas = Image.new(
        "RGB",
        (samples_per_class * tile_size, len(PAD_UFES_NATIVE_LABELS) * (tile_size + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
        label_rows = rows[rows["label"] == label].reset_index(drop=True)
        for sample_index, row in label_rows.iterrows():
            with Image.open(row["image_path"]) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            x = sample_index * tile_size
            y = label_index * (tile_size + caption_height)
            image_x = x + (tile_size - thumbnail.width) // 2
            image_y = y + (tile_size - thumbnail.height) // 2
            canvas.paste(thumbnail, (image_x, image_y))
            draw.text(
                (x + 4, y + tile_size + 6), f"{label} sample {sample_index + 1}", fill="black"
            )
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=90)


def run_evaluation(args: argparse.Namespace) -> dict[str, object] | None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100.")
    if args.calibration_bins < 2:
        raise ValueError("calibration-bins must be at least two.")
    if args.audit_only and args.artifact_contact_sheet is not None:
        raise ValueError("The prediction-blind artifact sheet is generated only after scoring.")

    data_dir = resolve_project_path(args.data_dir)
    out_dir = resolve_project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows, image_paths, source_hashes = load_source(data_dir)
    prepared = prepare_primary_cohort(
        metadata_rows,
        image_paths,
        source_hashes=source_hashes,
    )
    validate_frozen_audit(prepared.audit)
    audit_path = out_dir / "dataset_audit.json"
    audit_path.write_text(json.dumps(prepared.audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote aggregate HIBA audit: {audit_path}")
    if args.audit_only:
        return None

    checkpoint_paths = discover_checkpoint_paths(args.checkpoints_dir, folds=args.folds)
    device = _resolve_device(args.device)
    dataset = HibaImageDataset(prepared.image_rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    truth = prepared.lesion_rows["label"].tolist()
    patient_ids = prepared.lesion_rows["patient_id"].tolist()

    fold_image_probabilities = []
    fold_reports = []
    for fold_index, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Evaluating fold_{fold_index}: {checkpoint_path}")
        image_probabilities = _predict_checkpoint(
            checkpoint_path,
            loader,
            sample_count=len(prepared.image_rows),
            architecture=args.architecture,
            seed=args.seed,
            device=device,
        )
        lesion_probabilities = aggregate_lesion_probabilities(
            image_probabilities,
            prepared.image_rows,
            prepared.lesion_rows,
        )
        fold_image_probabilities.append(image_probabilities)
        fold_reports.append(
            {
                "fold_index": fold_index,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "metrics": multiclass_metrics(truth, lesion_probabilities),
            }
        )

    ensemble_image_probabilities = np.mean(np.stack(fold_image_probabilities, axis=0), axis=0)
    ensemble_probabilities = aggregate_lesion_probabilities(
        ensemble_image_probabilities,
        prepared.image_rows,
        prepared.lesion_rows,
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "context": (
            "Experimental external HIBA clinical-smartphone classification evidence only; not "
            "patient-taken-photo validation, a medical diagnosis, a fairness result, or a "
            "deployment-readiness result."
        ),
        "dataset": prepared.audit,
        "model": {
            "candidate": "PAD-UFES-native ConvNeXt-Tiny grouped-CV fold ensemble",
            "architecture": args.architecture,
            "labels": list(PAD_UFES_NATIVE_LABELS),
            "fold_count": args.folds,
            "seed": args.seed,
            "input_mode": "image_only",
            "preprocessing": EXPECTED_PREPROCESSING,
        },
        "protocol": {
            "task": "HIBA exact six-class clinical-photo frozen external evaluation",
            "primary_unit": "lesion",
            "image_ensemble": "equal_weight_mean_of_five_fold_probability_vectors",
            "repeated_image_aggregation": "equal_weight_mean_within_lesion",
            "uncertainty": "released_patient_id_cluster_percentile_bootstrap",
            "calibration": "descriptive_only_no_calibrator_fitted",
            "dermoscopy": "excluded",
            "non_exact_diagnoses": "excluded_before_model_access",
            "demographic_performance": "withheld_due_to_missing_FST_IV_V_VI_and_imbalance",
            "training_calibration_threshold_tuning_filtering_or_model_selection_on_hiba": False,
            "capture_context": HIBA_CAPTURE_CONTEXT,
            "workflow_artifact_context": WORKFLOW_ARTIFACT_CONTEXT,
        },
        "reference_baselines": {"always_majority_class": _majority_reference(truth)},
        "folds": fold_reports,
        "ensemble": {
            "metrics": multiclass_metrics(truth, ensemble_probabilities),
            "calibration": calibration_metrics(
                truth,
                ensemble_probabilities,
                bins=args.calibration_bins,
            ),
            "confidence_intervals": patient_cluster_bootstrap_intervals(
                truth,
                ensemble_probabilities,
                patient_ids,
                samples=args.bootstrap_samples,
                seed=args.seed,
                calibration_bins=args.calibration_bins,
            ),
        },
        "artifact_review": {
            "status": (
                "deterministic_prediction_blind_contact_sheet_generated"
                if args.artifact_contact_sheet is not None
                else "not_generated"
            ),
            "selection": "SHA256(seed:isic_id)_first_n_per_exact_class",
            "samples_per_class": args.artifact_samples_per_class,
            "predictions_used_for_selection": False,
            "effect_on_metrics_or_selection": "none",
        },
        "privacy": {
            "report_scope": "aggregate_metrics_only",
            "per_image_predictions_written": False,
            "patient_lesion_or_image_identifiers_written": False,
            "raw_metadata_written": False,
        },
        "limitations": [
            "HIBA clinical images were captured by dermatology professionals, not patients.",
            "The exact cohort has no released FST IV-VI examples and cannot establish fairness.",
            "Thirty-five scored lesions lack a released histopathology confirmation type.",
            "Clinical workflow artifacts are reviewed only after scoring and never used to filter.",
            "This one external cohort cannot establish deployment or medical readiness.",
        ],
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote aggregate HIBA report: {report_path}")

    if args.artifact_contact_sheet is not None:
        selected_rows = select_artifact_review_rows(
            prepared.image_rows,
            samples_per_class=args.artifact_samples_per_class,
            seed=args.seed,
        )
        write_artifact_contact_sheet(
            selected_rows,
            args.artifact_contact_sheet,
            samples_per_class=args.artifact_samples_per_class,
        )
        print(f"Wrote ignored prediction-blind artifact sheet: {args.artifact_contact_sheet}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    run_evaluation(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

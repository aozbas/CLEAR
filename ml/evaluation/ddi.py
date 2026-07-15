"""Evaluate frozen PAD-UFES-native checkpoints on the authorized DDI holdout.

DDI is evaluated as a benign-versus-malignant stress/fairness benchmark. The
six PAD-UFES-native probabilities are collapsed into a predeclared malignancy
score; DDI is not treated as a six-class benchmark.

Outputs contain aggregate experimental metrics only. They are not medical
conclusions and must remain on ignored paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.preprocessing import get_pad_ufes_transforms
from ml.training.train_pad_ufes import ARCHITECTURES, build_transfer_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "ddi"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "ml" / "models" / "pad_ufes_convnext_tiny_cv_seed42"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml" / "runs" / "evaluation" / "ddi-convnext-tiny-cv"
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
BINARY_THRESHOLD = 0.5
EXPECTED_PREPROCESSING = "resize_224_imagenet_normalization"
REQUIRED_COLUMNS = {"DDI_ID", "DDI_file", "skin_tone", "malignant", "disease"}
EXPECTED_SKIN_TONES = (12, 34, 56)
SKIN_TONE_NAMES = {
    12: "FST_I_II",
    34: "FST_III_IV",
    56: "FST_V_VI",
}
MALIGNANT_PAD_UFES_LABELS = (
    "basal_cell_carcinoma",
    "melanoma",
    "squamous_cell_carcinoma",
)
NON_MALIGNANT_PAD_UFES_LABELS = (
    "actinic_keratosis",
    "nevus",
    "seborrheic_keratosis",
)
INTERVAL_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "sensitivity",
    "specificity",
    "roc_auc",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen PAD-UFES-native checkpoints on Stanford DDI."
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_ddi_metadata(data_dir: Path) -> pd.DataFrame:
    """Load and validate the authorized DDI layout without changing raw data."""
    data_dir = resolve_project_path(data_dir)
    metadata_path = data_dir / "ddi_metadata.csv"
    images_dir = data_dir / "images"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing DDI metadata: {metadata_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing DDI images directory: {images_dir}")

    rows = pd.read_csv(metadata_path)
    missing_columns = REQUIRED_COLUMNS.difference(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DDI metadata is missing columns: {missing}")
    if rows.empty:
        raise ValueError("DDI metadata is empty.")

    for column in REQUIRED_COLUMNS:
        if bool(rows[column].isna().any()):
            raise ValueError(f"DDI metadata contains blank {column} values.")

    for column in ("DDI_ID", "DDI_file", "disease"):
        rows[column] = rows[column].astype(str).str.strip()
        if bool((rows[column] == "").any()):
            raise ValueError(f"DDI metadata contains blank {column} values.")

    if bool(rows["DDI_ID"].duplicated().any()):
        raise ValueError("DDI metadata contains duplicate DDI_ID rows.")
    if bool(rows["DDI_file"].duplicated().any()):
        raise ValueError("DDI metadata contains duplicate DDI_file rows.")

    numeric_skin_tones = pd.to_numeric(rows["skin_tone"], errors="coerce")
    if bool(numeric_skin_tones.isna().any()):
        raise ValueError("DDI metadata contains non-numeric skin_tone values.")
    if bool((numeric_skin_tones % 1 != 0).any()):
        raise ValueError("DDI metadata contains non-integral skin_tone values.")
    rows["skin_tone"] = numeric_skin_tones.astype(int)
    unexpected_skin_tones = sorted(set(rows["skin_tone"]) - set(EXPECTED_SKIN_TONES))
    if unexpected_skin_tones:
        raise ValueError(f"DDI metadata contains unknown skin_tone values: {unexpected_skin_tones}")

    rows["malignant"] = rows["malignant"].map(_parse_malignant)

    invalid_filenames = [
        filename
        for filename in rows["DDI_file"]
        if Path(filename).is_absolute()
        or "/" in filename
        or "\\" in filename
        or Path(filename).suffix.lower() != ".png"
    ]
    if invalid_filenames:
        raise ValueError("DDI_file values must be plain PNG filenames.")

    expected_names = set(rows["DDI_file"])
    actual_names = {path.name for path in images_dir.glob("*.png") if path.is_file()}
    if expected_names != actual_names:
        missing_count = len(expected_names - actual_names)
        extra_count = len(actual_names - expected_names)
        raise ValueError(
            f"DDI image files do not match metadata: missing={missing_count} extra={extra_count}."
        )

    rows = rows.sort_values("DDI_file").reset_index(drop=True)
    rows["image_path"] = rows["DDI_file"].map(lambda name: images_dir / name)
    return rows


def _parse_malignant(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"DDI metadata contains invalid malignant value: {value!r}")


def build_dataset_audit(rows: pd.DataFrame) -> dict[str, object]:
    """Return aggregate-only DDI metadata counts."""
    binary_support = {
        "non_malignant": int((~rows["malignant"]).sum()),
        "malignant": int(rows["malignant"].sum()),
    }
    skin_tone_support = {}
    for code in EXPECTED_SKIN_TONES:
        group = rows[rows["skin_tone"] == code]
        skin_tone_support[SKIN_TONE_NAMES[code]] = {
            "skin_tone_code": code,
            "total": len(group),
            "non_malignant": int((~group["malignant"]).sum()),
            "malignant": int(group["malignant"].sum()),
        }
    return {
        "dataset": "Stanford Diverse Dermatology Images",
        "source_url": "https://ddi-dataset.github.io/",
        "image_count": len(rows),
        "unique_image_id_count": int(rows["DDI_ID"].nunique()),
        "unique_filename_count": int(rows["DDI_file"].nunique()),
        "diagnosis_count": int(rows["disease"].nunique()),
        "binary_support": binary_support,
        "skin_tone_support": skin_tone_support,
        "patient_grouping": "unavailable_in_supplied_metadata",
        "report_scope": "aggregate_counts_only",
    }


def collapse_native_probabilities(
    probabilities: np.ndarray,
    *,
    labels: Sequence[str] = PAD_UFES_NATIVE_LABELS,
) -> np.ndarray:
    """Collapse six native probabilities into the predeclared malignancy score."""
    array = np.asarray(probabilities, dtype=np.float64)
    label_order = tuple(labels)
    if len(label_order) != len(set(label_order)):
        raise ValueError("labels must be unique.")
    if set(label_order) != set(PAD_UFES_NATIVE_LABELS):
        raise ValueError("labels must contain the PAD-UFES-native six-class set.")
    if array.ndim != 2 or array.shape[1] != len(label_order):
        raise ValueError("probabilities must have one column per supplied label.")
    if not np.isfinite(array).all():
        raise ValueError("probabilities must be finite.")
    if (array < -1e-7).any() or (array > 1.0 + 1e-7).any():
        raise ValueError("probabilities must be between zero and one.")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each probability row must sum to one.")

    label_to_index = {label: index for index, label in enumerate(label_order)}
    malignant_indices = [label_to_index[label] for label in MALIGNANT_PAD_UFES_LABELS]
    return array[:, malignant_indices].sum(axis=1)


def binary_metrics(truth: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    truth_array, score_array = _validated_binary_arrays(truth, scores)
    predictions = score_array >= BINARY_THRESHOLD

    true_negative = int((~truth_array & ~predictions).sum())
    false_positive = int((~truth_array & predictions).sum())
    false_negative = int((truth_array & ~predictions).sum())
    true_positive = int((truth_array & predictions).sum())
    total = len(truth_array)

    sensitivity = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    malignant_precision = _safe_divide(true_positive, true_positive + false_positive)
    non_malignant_precision = _safe_divide(true_negative, true_negative + false_negative)
    malignant_f1 = _f1_from_counts(true_positive, false_positive, false_negative)
    non_malignant_f1 = _f1_from_counts(true_negative, false_negative, false_positive)

    return {
        "threshold": BINARY_THRESHOLD,
        "support": {
            "total": total,
            "non_malignant": int((~truth_array).sum()),
            "malignant": int(truth_array.sum()),
        },
        "predicted": {
            "non_malignant": int((~predictions).sum()),
            "malignant": int(predictions.sum()),
        },
        "confusion": {
            "true_non_malignant_pred_non_malignant": true_negative,
            "true_non_malignant_pred_malignant": false_positive,
            "true_malignant_pred_non_malignant": false_negative,
            "true_malignant_pred_malignant": true_positive,
        },
        "accuracy": _safe_divide(true_positive + true_negative, total),
        "balanced_accuracy": _mean_defined((sensitivity, specificity)),
        "macro_f1": _mean_defined((non_malignant_f1, malignant_f1)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "malignant_precision": malignant_precision,
        "non_malignant_precision": non_malignant_precision,
        "malignant_f1": malignant_f1,
        "non_malignant_f1": non_malignant_f1,
        "roc_auc": _roc_auc(truth_array, score_array),
    }


def _validated_binary_arrays(
    truth: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth_array = np.asarray(truth, dtype=bool)
    score_array = np.asarray(scores, dtype=np.float64)
    if truth_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("truth and scores must be one-dimensional.")
    if len(truth_array) != len(score_array):
        raise ValueError("truth and scores must have the same length.")
    if not len(truth_array):
        raise ValueError("truth and scores must not be empty.")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite.")
    if (score_array < 0.0).any() or (score_array > 1.0).any():
        raise ValueError("scores must be between zero and one.")
    return truth_array, score_array


def _safe_divide(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    denominator = 2 * true_positive + false_positive + false_negative
    return _safe_divide(2 * true_positive, denominator)


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return None if not defined else float(sum(defined)) / len(defined)


def _roc_auc(truth: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(truth.sum())
    negative_count = len(truth) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop

    positive_rank_sum = float(ranks[truth].sum())
    minimum_positive_rank_sum = positive_count * (positive_count + 1) / 2.0
    return (positive_rank_sum - minimum_positive_rank_sum) / (positive_count * negative_count)


def bootstrap_confidence_intervals(
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Compute deterministic image-level stratified percentile intervals."""
    truth_array, score_array = _validated_binary_arrays(truth, scores)
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100.")
    negative_indices = np.flatnonzero(~truth_array)
    positive_indices = np.flatnonzero(truth_array)
    if not len(negative_indices) or not len(positive_indices):
        raise ValueError("bootstrap intervals require both binary outcomes.")

    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in INTERVAL_METRICS}
    for _ in range(samples):
        sampled_indices = np.concatenate(
            [
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
            ]
        )
        metrics = binary_metrics(truth_array[sampled_indices], score_array[sampled_indices])
        for metric in INTERVAL_METRICS:
            value = metrics[metric]
            if value is None:
                raise ValueError(f"Bootstrap metric {metric} is undefined.")
            distributions[metric].append(float(value))

    return {
        "method": "image_level_stratified_percentile_bootstrap",
        "sampling_unit": "image",
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "intervals": {
            metric: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for metric, values in distributions.items()
        },
    }


def evaluate_binary_scores(
    truth: np.ndarray,
    scores: np.ndarray,
    skin_tones: np.ndarray,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    truth_array, score_array = _validated_binary_arrays(truth, scores)
    skin_tone_array = np.asarray(skin_tones, dtype=int)
    if skin_tone_array.ndim != 1 or len(skin_tone_array) != len(truth_array):
        raise ValueError("skin_tones must align with truth and scores.")
    observed_skin_tones = set(skin_tone_array.tolist())
    if observed_skin_tones != set(EXPECTED_SKIN_TONES):
        raise ValueError(
            "DDI evaluation requires skin-tone groups 12, 34, and 56; "
            f"observed={sorted(observed_skin_tones)}."
        )

    overall = _evaluate_group(
        truth_array,
        score_array,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    by_skin_tone = {}
    for code in EXPECTED_SKIN_TONES:
        mask = skin_tone_array == code
        by_skin_tone[SKIN_TONE_NAMES[code]] = {
            "skin_tone_code": code,
            **_evaluate_group(
                truth_array[mask],
                score_array[mask],
                bootstrap_samples=bootstrap_samples,
                seed=seed + code,
            ),
        }
    return {
        "overall": overall,
        "by_skin_tone": by_skin_tone,
        "descriptive_skin_tone_gaps": _descriptive_gaps(by_skin_tone),
    }


def _evaluate_group(
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    return {
        "metrics": binary_metrics(truth, scores),
        "confidence_intervals": bootstrap_confidence_intervals(
            truth,
            scores,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }


def _descriptive_gaps(by_skin_tone: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    gaps = {}
    for metric in ("balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc"):
        values = {
            group_name: float(group["metrics"][metric])
            for group_name, group in by_skin_tone.items()
            if group["metrics"][metric] is not None
        }
        lowest_group = min(values, key=values.get)
        highest_group = max(values, key=values.get)
        gaps[metric] = {
            "max_minus_min": values[highest_group] - values[lowest_group],
            "lowest_group": lowest_group,
            "highest_group": highest_group,
        }
    return gaps


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


class DdiImageDataset(Dataset):
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
        raise RuntimeError("Inference did not produce probabilities for every DDI image.")

    del model
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_evaluation(args: argparse.Namespace) -> dict[str, object] | None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100.")

    data_dir = resolve_project_path(args.data_dir)
    out_dir = resolve_project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_ddi_metadata(data_dir)
    audit = {
        **build_dataset_audit(rows),
        "metadata_sha256": _sha256(data_dir / "ddi_metadata.csv"),
    }
    audit_path = out_dir / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote aggregate DDI audit: {audit_path}")
    if args.audit_only:
        return None

    checkpoint_paths = discover_checkpoint_paths(args.checkpoints_dir, folds=args.folds)
    device = _resolve_device(args.device)
    dataset = DdiImageDataset(rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    truth = rows["malignant"].to_numpy(dtype=bool)
    skin_tones = rows["skin_tone"].to_numpy(dtype=int)

    fold_probabilities = []
    fold_reports = []
    for fold_index, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Evaluating fold_{fold_index}: {checkpoint_path}")
        probabilities = _predict_checkpoint(
            checkpoint_path,
            loader,
            sample_count=len(rows),
            architecture=args.architecture,
            seed=args.seed,
            device=device,
        )
        scores = collapse_native_probabilities(
            probabilities,
            labels=PAD_UFES_NATIVE_LABELS,
        )
        fold_probabilities.append(probabilities)
        fold_reports.append(
            {
                "fold_index": fold_index,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "metrics": binary_metrics(truth, scores),
            }
        )

    ensemble_probabilities = np.mean(np.stack(fold_probabilities, axis=0), axis=0)
    ensemble_scores = collapse_native_probabilities(
        ensemble_probabilities,
        labels=PAD_UFES_NATIVE_LABELS,
    )
    ensemble_report = evaluate_binary_scores(
        truth,
        ensemble_scores,
        skin_tones,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "context": (
            "Experimental external-holdout classification and fairness evidence only; "
            "not a medical diagnosis or deployment-readiness result."
        ),
        "dataset": audit,
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
            "task": "DDI benign-versus-malignant external stress/fairness evaluation",
            "malignant_probability": list(MALIGNANT_PAD_UFES_LABELS),
            "non_malignant_probability": list(NON_MALIGNANT_PAD_UFES_LABELS),
            "threshold": BINARY_THRESHOLD,
            "threshold_selection": "predeclared_without_DDI_tuning",
            "ensemble": "equal_weight_mean_of_five_fold_probability_vectors",
            "uncertainty": "image_level_stratified_percentile_bootstrap",
            "six_class_evaluation": "withheld_due_to_sparse_and_ambiguous_overlap",
        },
        "reference_baselines": {
            "always_non_malignant": {
                "description": (
                    "Trivial prevalence reference; predicts non-malignant for every image."
                ),
                "metrics": binary_metrics(truth, np.zeros(len(truth), dtype=np.float64)),
            }
        },
        "folds": fold_reports,
        "ensemble": ensemble_report,
        "limitations": [
            "The supplied metadata has no patient identifier, so intervals cannot account "
            "for repeated patients among the 656 images.",
            "The six-class closed-set model is forced into a binary score on diagnoses that "
            "extend beyond its PAD-UFES training taxonomy.",
            "Skin-tone comparisons are descriptive experimental evidence with finite subgroup "
            "support, not proof of clinical fairness.",
            "No DDI images were used for training, calibration, threshold selection, or model "
            "selection in this evaluation.",
        ],
        "artifact_scope": "aggregate_metrics_only_no_per_image_predictions",
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    metrics = ensemble_report["overall"]["metrics"]
    print(
        "Wrote DDI report: "
        f"{report_path}; balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} roc_auc={metrics['roc_auc']:.4f}"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    run_evaluation(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run locked validation-only PAD/HIBA partial-freeze ConvNeXt development.

PAD-UFES and HIBA are development data here. The former outer-test role is excluded from every
fold and is never scored. Outputs can justify a separately preregistered final fit, but they are
not external validation, deployment evidence, medical conclusions, or a production model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_hiba_cv import PROTOCOL as HIBA_PROTOCOL
from ml.training.prepare_pad_ufes import project_relative
from ml.training.prepare_pad_ufes_cv import DEFAULT_FOLDS
from ml.training.prepare_pad_ufes_cv import PROTOCOL as PAD_PROTOCOL
from ml.training.run_pad_hiba_convnext_cv import (
    ARCHITECTURE,
    AUGMENTATION_PROFILE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIBA_SPLITS_DIR,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PAD_SPLITS_DIR,
    DEFAULT_SEED,
    DEFAULT_TORCH_CACHE_DIR,
    DEFAULT_WEIGHT_DECAY,
    EXPECTED_SOURCE_EFFECTIVE_UNIT_COUNTS,
    EXPECTED_SOURCE_RAW_IMAGE_COUNTS,
    HIBA_VIEW_WEIGHTING,
    PREPROCESSING,
    PRETRAINED_WEIGHTS,
    SELECTION_METRIC,
    SOURCE_CLASS_WEIGHTING,
    SOURCE_ORDER,
    MultiSourceImageDataset,
    _add_confusions,
    _compact_metrics,
    _metrics_from_confusion,
    _parameter_count,
    _sha256,
    build_loader,
    load_multi_source_manifests,
    run_epoch,
)
from ml.training.train import get_device, resolve_project_path, set_seed
from ml.training.train_pad_ufes import build_transfer_model, pretrained_weights_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT / "ml" / "runs" / "training" / "pad_hiba_convnext_tiny-last-block-cv-seed42"
)
DEFAULT_CHECKPOINTS_DIR = (
    PROJECT_ROOT / "ml" / "models" / "pad_hiba_convnext_tiny_last_block_cv_seed42"
)
DEVELOPMENT_PROTOCOL = "pad_hiba_last_block_classifier_validation_only_cv"
TRAINABLE_SCOPE = "convnext_final_feature_block_and_classifier"
FORMER_TEST_ROLE_USE = "excluded_without_model_access_or_metrics"
MAX_TRAINABLE_PARAMETERS = 5_000_000
MAX_TRAINABLE_FRACTION = 0.2
MAX_INFERENCE_PARAMETERS = 30_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked validation-only PAD/HIBA partial-freeze ConvNeXt development."
    )
    parser.add_argument("--pad-splits-dir", type=Path, default=DEFAULT_PAD_SPLITS_DIR)
    parser.add_argument("--hiba-splits-dir", type=Path, default=DEFAULT_HIBA_SPLITS_DIR)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--torch-cache-dir", type=Path, default=DEFAULT_TORCH_CACHE_DIR)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def development_rows_for_fold(rows: pd.DataFrame) -> pd.DataFrame:
    """Return only locked train/validation rows and fail closed on role drift."""
    if set(rows["split"]) != {"train", "val", "test"}:
        raise ValueError("A PAD/HIBA fold must contain the locked train, val, and test roles.")
    development = rows[rows["split"].isin(("train", "val"))].copy().reset_index(drop=True)
    if development.empty or bool((development["split"] == "test").any()):
        raise ValueError("Former test-role rows must be absent from partial-freeze development.")
    for split in ("train", "val"):
        split_rows = development[development["split"] == split]
        if set(split_rows["source"]) != set(SOURCE_ORDER):
            raise ValueError(f"Partial-freeze {split} data is missing a locked source.")
        for source in SOURCE_ORDER:
            labels = set(split_rows.loc[split_rows["source"] == source, "label"])
            if labels != set(PAD_UFES_NATIVE_LABELS):
                raise ValueError(f"Partial-freeze {split}/{source} label coverage drifted.")
    return development


def configure_partial_freeze(model: nn.Module) -> dict[str, object]:
    """Freeze all parameters except the final ConvNeXt block and classifier."""
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    if features is None or classifier is None or len(features) == 0 or len(classifier) == 0:
        raise ValueError("ConvNeXt model does not expose the locked features/classifier structure.")
    final_stage = features[-1]
    if not hasattr(final_stage, "__len__") or len(final_stage) == 0:
        raise ValueError("ConvNeXt final feature stage does not contain a final block.")
    final_block = final_stage[-1]

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (final_block, classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True

    allowed_ids = {
        id(parameter) for module in (final_block, classifier) for parameter in module.parameters()
    }
    observed_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if not allowed_ids or observed_ids != allowed_ids:
        raise RuntimeError("Trainable ConvNeXt parameters differ from the locked module scope.")
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable_names:
        raise RuntimeError("Locked partial-freeze scope has no trainable parameters.")
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    inference_count = _parameter_count(model)
    trainable_fraction = trainable_count / inference_count
    name_digest = hashlib.sha256(("\n".join(trainable_names) + "\n").encode()).hexdigest()
    return {
        "profile": TRAINABLE_SCOPE,
        "trainable_parameter_count": trainable_count,
        "inference_parameter_count": inference_count,
        "trainable_parameter_fraction": trainable_fraction,
        "trainable_parameter_tensor_count": len(trainable_names),
        "trainable_parameter_names_sha256": name_digest,
    }


def build_trainable_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("Invalid locked optimizer hyperparameters.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Partial-freeze optimizer has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    expected_ids = {id(parameter) for parameter in parameters}
    if optimizer_ids != expected_ids:
        raise RuntimeError("Optimizer parameters differ from the locked trainable scope.")
    return optimizer


def _validate_checkpoint_metadata(
    checkpoint: Mapping[str, object],
    *,
    fold_index: int,
    manifest_fingerprint: str,
    scope: Mapping[str, object],
) -> None:
    expected = {
        "architecture": ARCHITECTURE,
        "input_mode": "image_only",
        "dataset": "pad_ufes_hiba",
        "dataset_role": "multisource_validation_only_development",
        "development_protocol": DEVELOPMENT_PROTOCOL,
        "outer_test_scored": False,
        "former_test_role_use": FORMER_TEST_ROLE_USE,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "pretrained_weights_id": pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS),
        "preprocessing": PREPROCESSING,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "selection_metric": SELECTION_METRIC,
        "trainable_scope": TRAINABLE_SCOPE,
        "trainable_parameter_count": scope["trainable_parameter_count"],
        "inference_parameter_count": scope["inference_parameter_count"],
        "trainable_parameter_names_sha256": scope["trainable_parameter_names_sha256"],
        "manifest_fingerprint": manifest_fingerprint,
        "fold_index": fold_index,
        "seed": DEFAULT_SEED,
    }
    mismatches = [
        f"{key}={checkpoint.get(key)!r}"
        for key, value in expected.items()
        if checkpoint.get(key) != value
    ]
    if mismatches:
        raise ValueError("Partial-freeze checkpoint provenance drifted: " + ", ".join(mismatches))


def train_fold(
    rows: pd.DataFrame,
    *,
    fold_index: int,
    manifest_fingerprint: str,
    source_total_raw_image_counts: dict[str, int],
    source_total_effective_unit_counts: dict[str, float],
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
    num_workers: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    pretrained: bool = True,
) -> dict[str, object]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("Invalid locked training hyperparameters.")
    development_rows = development_rows_for_fold(rows)
    excluded_rows = rows[rows["split"] == "test"]
    if excluded_rows.empty:
        raise ValueError("Every fold must retain a non-empty excluded former-test role.")
    set_seed(seed)
    datasets = {
        split: MultiSourceImageDataset(development_rows, split) for split in ("train", "val")
    }
    loaders = {
        split: build_loader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            seed=seed,
        )
        for split, dataset in datasets.items()
    }
    model = build_transfer_model(
        architecture=ARCHITECTURE,
        weights=PRETRAINED_WEIGHTS if pretrained else "none",
    ).to(device)
    scope = configure_partial_freeze(model)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = build_trainable_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW_trainable_parameters_only",
        "schedule": "none",
        "augmentation_profile": AUGMENTATION_PROFILE,
        "label_smoothing": 0.0,
        "sampling": "random_shuffle_without_replacement",
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "trainable_scope": TRAINABLE_SCOPE,
        "native_training_mode_behavior": True,
    }
    best_source_mean = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history = []
    checkpoint_path = resolve_project_path(Path(checkpoint_path))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        train_compact = _compact_metrics(train_metrics)
        val_compact = _compact_metrics(val_metrics)
        print(
            f"fold_{fold_index} epoch={epoch}/{epochs} "
            f"train_source_mean={train_compact['primary_source_mean_macro_f1']:.4f} "
            f"val_source_mean={val_compact['primary_source_mean_macro_f1']:.4f}"
        )
        history.append({"epoch": epoch, "train": train_compact, "val": val_compact})
        source_mean = float(val_metrics["primary_source_mean_macro_f1"])
        val_loss = float(val_metrics["loss"])
        improved = source_mean > best_source_mean or (
            math.isclose(source_mean, best_source_mean, abs_tol=1e-12) and val_loss < best_val_loss
        )
        if improved:
            best_source_mean = source_mean
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": ARCHITECTURE,
                    "input_mode": "image_only",
                    "dataset": "pad_ufes_hiba",
                    "dataset_role": "multisource_validation_only_development",
                    "development_protocol": DEVELOPMENT_PROTOCOL,
                    "outer_test_scored": False,
                    "former_test_role_use": FORMER_TEST_ROLE_USE,
                    "pretrained_weights": PRETRAINED_WEIGHTS if pretrained else "none",
                    "pretrained_weights_id": (
                        pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS)
                        if pretrained
                        else None
                    ),
                    "preprocessing": PREPROCESSING,
                    "augmentation_profile": AUGMENTATION_PROFILE,
                    "selection_metric": SELECTION_METRIC,
                    "trainable_scope": TRAINABLE_SCOPE,
                    "trainable_parameter_count": scope["trainable_parameter_count"],
                    "inference_parameter_count": scope["inference_parameter_count"],
                    "trainable_parameter_names_sha256": scope["trainable_parameter_names_sha256"],
                    "manifest_fingerprint": manifest_fingerprint,
                    "fold_index": fold_index,
                    "epoch": epoch,
                    "seed": seed,
                    "hyperparameters": hyperparameters,
                    "val_metrics": val_compact,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _validate_checkpoint_metadata(
        checkpoint,
        fold_index=fold_index,
        manifest_fingerprint=manifest_fingerprint,
        scope=scope,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    observed_scope = configure_partial_freeze(model)
    if observed_scope != scope:
        raise ValueError("Reloaded checkpoint trainable scope drifted.")
    train_evaluation_dataset = MultiSourceImageDataset(
        development_rows,
        "train",
        use_training_transform=False,
    )
    train_evaluation_loader = build_loader(
        train_evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    selected_train = run_epoch(model, train_evaluation_loader, criterion, device)
    selected_val = run_epoch(model, loaders["val"], criterion, device)
    report = {
        "context": (
            "Experimental validation-only PAD-UFES plus HIBA partial-freeze development; "
            "not external validation, medical certainty, or production evidence."
        ),
        "development_protocol": DEVELOPMENT_PROTOCOL,
        "outer_test_scored": False,
        "former_test_role_use": FORMER_TEST_ROLE_USE,
        "architecture": ARCHITECTURE,
        "input_mode": "image_only",
        "pretrained_weights": PRETRAINED_WEIGHTS if pretrained else "none",
        "pretrained_weights_id": (
            pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS) if pretrained else None
        ),
        "preprocessing": PREPROCESSING,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": seed,
        "fold_index": fold_index,
        "validation_outer_fold": (fold_index + 1) % DEFAULT_FOLDS,
        "excluded_outer_fold": fold_index,
        "pad_protocol": PAD_PROTOCOL,
        "hiba_protocol": HIBA_PROTOCOL,
        "hiba_role": "multisource_development",
        "manifest_fingerprint": manifest_fingerprint,
        "selection_metric": SELECTION_METRIC,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "primary_units": {"pad_ufes": "image", "hiba": "lesion"},
        "hyperparameters": hyperparameters,
        **scope,
        "source_total_raw_image_counts": source_total_raw_image_counts,
        "source_total_effective_unit_counts": source_total_effective_unit_counts,
        "included_raw_image_counts": {
            split: {
                source: int((datasets[split].rows["source"] == source).sum())
                for source in SOURCE_ORDER
            }
            for split in ("train", "val")
        },
        "included_effective_unit_counts": {
            split: {
                source: float(
                    datasets[split]
                    .rows.loc[datasets[split].rows["source"] == source, "view_mass"]
                    .sum()
                )
                for source in SOURCE_ORDER
            }
            for split in ("train", "val")
        },
        "excluded_raw_image_counts": {
            source: int((excluded_rows["source"] == source).sum()) for source in SOURCE_ORDER
        },
        "best_epoch": best_epoch,
        "best_val_primary_source_mean_macro_f1": best_source_mean,
        "history": history,
        "selected_train": selected_train,
        "selected_val": selected_val,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "privacy": {
            "aggregate_metrics_only": True,
            "per_image_predictions_written": False,
            "identifiers_or_paths_written": False,
        },
    }
    run_dir = resolve_project_path(Path(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def validate_report(
    report: Mapping[str, object],
    *,
    fold_index: int,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> None:
    expected = {
        "development_protocol": DEVELOPMENT_PROTOCOL,
        "outer_test_scored": False,
        "former_test_role_use": FORMER_TEST_ROLE_USE,
        "architecture": ARCHITECTURE,
        "input_mode": "image_only",
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "pretrained_weights_id": pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS),
        "preprocessing": PREPROCESSING,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": seed,
        "fold_index": fold_index,
        "validation_outer_fold": (fold_index + 1) % DEFAULT_FOLDS,
        "excluded_outer_fold": fold_index,
        "pad_protocol": PAD_PROTOCOL,
        "hiba_protocol": HIBA_PROTOCOL,
        "hiba_role": "multisource_development",
        "selection_metric": SELECTION_METRIC,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "primary_units": {"pad_ufes": "image", "hiba": "lesion"},
        "profile": TRAINABLE_SCOPE,
    }
    mismatches = [
        f"{key}={report.get(key)!r}" for key, value in expected.items() if report.get(key) != value
    ]
    if "test" in report:
        mismatches.append("forbidden test metrics object is present")
    expected_hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW_trainable_parameters_only",
        "schedule": "none",
        "augmentation_profile": AUGMENTATION_PROFILE,
        "label_smoothing": 0.0,
        "sampling": "random_shuffle_without_replacement",
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "trainable_scope": TRAINABLE_SCOPE,
        "native_training_mode_behavior": True,
    }
    hyperparameters = report.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        mismatches.append("hyperparameters are missing")
    else:
        mismatches.extend(
            f"hyperparameters.{key}={hyperparameters.get(key)!r}"
            for key, value in expected_hyperparameters.items()
            if hyperparameters.get(key) != value
        )
    fingerprint = report.get("manifest_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        mismatches.append(f"manifest_fingerprint={fingerprint!r}")
    count_fields = (
        "trainable_parameter_count",
        "inference_parameter_count",
        "trainable_parameter_tensor_count",
        "checkpoint_bytes",
    )
    for field in count_fields:
        if not isinstance(report.get(field), int) or int(report[field]) <= 0:
            mismatches.append(f"{field}={report.get(field)!r}")
    fraction = report.get("trainable_parameter_fraction")
    if not isinstance(fraction, (int, float)) or not 0.0 < float(fraction) <= 1.0:
        mismatches.append(f"trainable_parameter_fraction={fraction!r}")
    for field in ("trainable_parameter_names_sha256", "checkpoint_sha256"):
        value = report.get(field)
        if not isinstance(value, str) or len(value) != 64:
            mismatches.append(f"{field}={value!r}")
    for split_name in ("selected_train", "selected_val"):
        metrics = report.get(split_name)
        if not isinstance(metrics, dict):
            mismatches.append(f"{split_name} is missing")
            continue
        if set(metrics.get("by_source", {})) != set(SOURCE_ORDER):
            mismatches.append(f"{split_name}.by_source is invalid")
        if not isinstance(metrics.get("hiba_lesion"), dict):
            mismatches.append(f"{split_name}.hiba_lesion is invalid")
    if mismatches:
        raise ValueError(
            f"Partial-freeze report fold_{fold_index} violates the locked protocol: "
            + ", ".join(mismatches)
        )


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence.")
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _pooled_validation_confusion(
    reports: Sequence[dict[str, object]],
    *keys: str,
) -> list[list[int]]:
    confusions = []
    for report in reports:
        value: object = report["selected_val"]
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"Validation metrics are missing {'.'.join(keys)}.")
            value = value[key]
        if not isinstance(value, dict) or "confusion_matrix" not in value:
            raise ValueError(f"Validation metrics are missing {'.'.join(keys)} confusion.")
        confusions.append(value["confusion_matrix"])
    return _add_confusions(confusions)


def summarize_reports(
    reports_root: Path,
    out_path: Path,
    *,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    reports_root = resolve_project_path(Path(reports_root))
    reports = []
    for fold_index in range(num_folds):
        report_path = reports_root / f"fold_{fold_index}" / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing partial-freeze report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_report(report, fold_index=fold_index, seed=seed)
        reports.append(report)
    consistency_keys = (
        "manifest_fingerprint",
        "pretrained_weights_id",
        "trainable_parameter_count",
        "inference_parameter_count",
        "trainable_parameter_fraction",
        "trainable_parameter_tensor_count",
        "trainable_parameter_names_sha256",
        "source_total_raw_image_counts",
        "source_total_effective_unit_counts",
    )
    disagreements = [
        key
        for key in consistency_keys
        if len({json.dumps(report.get(key), sort_keys=True) for report in reports}) != 1
    ]
    if disagreements:
        raise ValueError(
            "Partial-freeze reports disagree on provenance: " + ", ".join(disagreements)
        )

    pooled_pad = _metrics_from_confusion(
        _pooled_validation_confusion(reports, "by_source", "pad_ufes")
    )
    pooled_hiba_image = _metrics_from_confusion(
        _pooled_validation_confusion(reports, "by_source", "hiba")
    )
    pooled_hiba_lesion = _metrics_from_confusion(
        _pooled_validation_confusion(reports, "hiba_lesion")
    )
    observed_support = {
        "pad_image": int(pooled_pad["total_support"]),
        "hiba_image": int(pooled_hiba_image["total_support"]),
        "hiba_lesion": int(pooled_hiba_lesion["total_support"]),
    }
    expected_support = {"pad_image": 2_298, "hiba_image": 309, "hiba_lesion": 308}
    if observed_support != expected_support:
        raise ValueError(
            "Validation coverage differs from one complete rotating pass: "
            f"observed={observed_support}, expected={expected_support}"
        )
    source_mean = statistics.fmean(
        [float(pooled_pad["macro_f1"]), float(pooled_hiba_lesion["macro_f1"])]
    )
    worst_source = min(float(pooled_pad["macro_f1"]), float(pooled_hiba_lesion["macro_f1"]))
    gaps = [
        float(report["selected_train"]["primary_source_mean_macro_f1"])
        - float(report["selected_val"]["primary_source_mean_macro_f1"])
        for report in reports
    ]
    trainable_count = int(reports[0]["trainable_parameter_count"])
    inference_count = int(reports[0]["inference_parameter_count"])
    trainable_fraction = float(reports[0]["trainable_parameter_fraction"])
    rules = {
        "mean_selected_train_val_primary_source_mean_macro_f1_gap_lte_0_2000": (
            statistics.fmean(gaps) <= 0.2
        ),
        "pooled_validation_pad_image_macro_f1_gte_0_6000": (float(pooled_pad["macro_f1"]) >= 0.6),
        "pooled_validation_hiba_lesion_macro_f1_gte_0_5000": (
            float(pooled_hiba_lesion["macro_f1"]) >= 0.5
        ),
        "pooled_validation_primary_source_mean_macro_f1_gte_0_5600": source_mean >= 0.56,
        "pooled_validation_primary_worst_source_macro_f1_gte_0_5000": worst_source >= 0.5,
        "pooled_validation_pad_melanoma_f1_gte_0_5000": (
            float(pooled_pad["per_class"]["melanoma"]["f1"]) >= 0.5
        ),
        "pooled_validation_pad_scc_f1_gte_0_2000": (
            float(pooled_pad["per_class"]["squamous_cell_carcinoma"]["f1"]) >= 0.2
        ),
        "pooled_validation_hiba_melanoma_f1_gte_0_3500": (
            float(pooled_hiba_lesion["per_class"]["melanoma"]["f1"]) >= 0.35
        ),
        "pooled_validation_hiba_scc_f1_gte_0_3000": (
            float(pooled_hiba_lesion["per_class"]["squamous_cell_carcinoma"]["f1"]) >= 0.3
        ),
        "trainable_parameter_count_lte_5000000": trainable_count <= MAX_TRAINABLE_PARAMETERS,
        "trainable_parameter_fraction_lte_0_2000": trainable_fraction <= MAX_TRAINABLE_FRACTION,
        "inference_parameter_count_lte_30000000": inference_count <= MAX_INFERENCE_PARAMETERS,
    }
    summary = {
        "context": (
            "Experimental validation-only PAD-UFES plus HIBA partial-freeze development; "
            "not external validation, medical certainty, or production evidence."
        ),
        "development_protocol": DEVELOPMENT_PROTOCOL,
        "outer_test_scored": False,
        "former_test_role_use": FORMER_TEST_ROLE_USE,
        "pad_protocol": PAD_PROTOCOL,
        "hiba_protocol": HIBA_PROTOCOL,
        "num_folds": num_folds,
        "seed": seed,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "architecture": ARCHITECTURE,
        "input_mode": "image_only",
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "pretrained_weights_id": reports[0]["pretrained_weights_id"],
        "preprocessing": PREPROCESSING,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "selection_metric": SELECTION_METRIC,
        "primary_units": {"pad_ufes": "image", "hiba": "lesion"},
        "manifest_fingerprint": reports[0]["manifest_fingerprint"],
        "trainable_scope": TRAINABLE_SCOPE,
        "trainable_parameter_count": trainable_count,
        "inference_parameter_count": inference_count,
        "trainable_parameter_fraction": trainable_fraction,
        "trainable_parameter_tensor_count": reports[0]["trainable_parameter_tensor_count"],
        "trainable_parameter_names_sha256": reports[0]["trainable_parameter_names_sha256"],
        "source_total_raw_image_counts": reports[0]["source_total_raw_image_counts"],
        "source_total_effective_unit_counts": reports[0]["source_total_effective_unit_counts"],
        "checkpoint_bytes": _distribution(
            [float(report["checkpoint_bytes"]) for report in reports]
        ),
        "selected_train_val_primary_source_mean_macro_f1_gap": _distribution(gaps),
        "pooled_validation_primary_by_source": {
            "pad_ufes_image": pooled_pad,
            "hiba_lesion": pooled_hiba_lesion,
        },
        "pooled_validation_hiba_image_secondary": pooled_hiba_image,
        "pooled_validation_primary_source_mean_macro_f1": source_mean,
        "pooled_validation_primary_worst_source_macro_f1": worst_source,
        "decision_rules": {**rules, "all_pass": all(rules.values())},
        "folds": [
            {
                "fold_index": fold_index,
                "best_epoch": int(report["best_epoch"]),
                "best_val_primary_source_mean_macro_f1": float(
                    report["best_val_primary_source_mean_macro_f1"]
                ),
                "selected_train_primary_source_mean_macro_f1": float(
                    report["selected_train"]["primary_source_mean_macro_f1"]
                ),
                "selected_val_primary_source_mean_macro_f1": float(
                    report["selected_val"]["primary_source_mean_macro_f1"]
                ),
                "selected_val_pad_image_macro_f1": float(
                    report["selected_val"]["by_source"]["pad_ufes"]["macro_f1"]
                ),
                "selected_val_hiba_lesion_macro_f1": float(
                    report["selected_val"]["hiba_lesion"]["macro_f1"]
                ),
            }
            for fold_index, report in enumerate(reports)
        ],
        "privacy": {
            "aggregate_metrics_only": True,
            "per_image_predictions_written": False,
            "identifiers_or_paths_written": False,
        },
        "caveat": (
            "PAD-UFES and HIBA are development data and no former outer-test role was scored. "
            "Passing can only justify a separately preregistered final fit and untouched external "
            "assessment; it cannot establish robustness, fairness, patient-self-photo behavior, "
            "deployment, diagnosis, or medical readiness."
        ),
    }
    if "test" in summary:
        raise RuntimeError("Validation-only summary contains a forbidden test metrics object.")
    out_path = resolve_project_path(Path(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    device = get_device(args.device)
    print(f"Using device: {device}")
    manifests = load_multi_source_manifests(args.pad_splits_dir, args.hiba_splits_dir)
    if manifests.fingerprint != "23d3f41f18fc6d1082434fc049b6a7b7af07785df70b668bfb5ec51115747c5d":
        raise ValueError("PAD/HIBA manifest fingerprint differs from the preregistered cohort.")
    if manifests.source_total_raw_image_counts != EXPECTED_SOURCE_RAW_IMAGE_COUNTS:
        raise ValueError("PAD/HIBA raw source counts drifted.")
    if manifests.source_total_effective_unit_counts != EXPECTED_SOURCE_EFFECTIVE_UNIT_COUNTS:
        raise ValueError("PAD/HIBA effective source counts drifted.")
    print(
        f"Validated {DEFAULT_FOLDS} PAD/HIBA rotating manifests for validation-only use; "
        f"fingerprint={manifests.fingerprint}"
    )
    torch_cache_dir = resolve_project_path(args.torch_cache_dir)
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_cache_dir))
    runs_root = resolve_project_path(args.runs_root)
    checkpoints_dir = resolve_project_path(args.checkpoints_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for fold_index in range(DEFAULT_FOLDS):
        run_dir = runs_root / f"fold_{fold_index}"
        report_path = run_dir / "report.json"
        if args.resume and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validate_report(report, fold_index=fold_index)
            print(f"fold_{fold_index}: validated report exists; skipping")
            continue
        print(f"fold_{fold_index}: starting validation-only partial-freeze ConvNeXt-Tiny")
        report = train_fold(
            manifests.folds[fold_index],
            fold_index=fold_index,
            manifest_fingerprint=manifests.fingerprint,
            source_total_raw_image_counts=manifests.source_total_raw_image_counts,
            source_total_effective_unit_counts=manifests.source_total_effective_unit_counts,
            checkpoint_path=checkpoints_dir / f"fold_{fold_index}.pt",
            run_dir=run_dir,
            device=device,
            num_workers=args.num_workers,
        )
        validate_report(report, fold_index=fold_index)
        print(
            f"fold_{fold_index}: best_epoch={report['best_epoch']} "
            f"selected_val_source_mean="
            f"{report['selected_val']['primary_source_mean_macro_f1']:.4f}; "
            "former test role unscored"
        )
    summary = summarize_reports(runs_root, runs_root / "summary.json")
    print(
        "PAD/HIBA validation-only partial-freeze development complete: "
        f"source_mean={summary['pooled_validation_primary_source_mean_macro_f1']:.4f} "
        f"worst_source={summary['pooled_validation_primary_worst_source_macro_f1']:.4f} "
        f"all_rules_pass={summary['decision_rules']['all_pass']} "
        f"summary={project_relative(runs_root / 'summary.json')}"
    )
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

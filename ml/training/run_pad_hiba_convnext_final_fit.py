"""Fit the owner-approved PAD-UFES + HIBA ConvNeXt-Tiny demo checkpoint.

This runner consumes the already completed, locked five-fold development summary and fits one
model on all approved PAD-UFES and HIBA development images for a fixed epoch count. It does not
evaluate MILK10k or create new medical-performance evidence. The resulting classifier remains an
experimental demonstration model and failed the preregistered cross-source promotion gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_pad_ufes import project_relative
from ml.training.run_pad_hiba_convnext_cv import (
    ARCHITECTURE,
    AUGMENTATION_PROFILE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIBA_SPLITS_DIR,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PAD_SPLITS_DIR,
    DEFAULT_SEED,
    DEFAULT_TORCH_CACHE_DIR,
    DEFAULT_WEIGHT_DECAY,
    HIBA_VIEW_WEIGHTING,
    PREPROCESSING,
    PRETRAINED_WEIGHTS,
    SOURCE_CLASS_WEIGHTING,
    SOURCE_ORDER,
    MultiSourceImageDataset,
    MultiSourceManifests,
    build_loader,
    load_multi_source_manifests,
    run_epoch,
)
from ml.training.train import get_device, resolve_project_path, set_seed
from ml.training.train_pad_ufes import build_transfer_model, pretrained_weights_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CV_SUMMARY = (
    PROJECT_ROOT / "ml" / "runs" / "training" / "pad_hiba_convnext_tiny-cv-seed42" / "summary.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "ml" / "models" / "pad_hiba_convnext_tiny_source_balanced_final_seed42.pt"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "ml"
    / "runs"
    / "training"
    / "pad_hiba_convnext_tiny-source-balanced-final-seed42"
    / "report.json"
)
EXPECTED_CV_SUMMARY_SHA256 = "20dec905c9470dc34e467d95354ee78b5affaaa14d8ef4d7f13dad7f96a7da53"
EXPECTED_MANIFEST_FINGERPRINT = "23d3f41f18fc6d1082434fc049b6a7b7af07785df70b668bfb5ec51115747c5d"
EXPECTED_SELECTED_EPOCHS = (15, 8, 10, 11, 13)
FINAL_EPOCHS = 11
TRAINING_PROTOCOL = "pad_hiba_source_balanced_full_development_final_fit_v1"
MODEL_VERSION = "pad-hiba-convnext-tiny-source-balanced-final-2026-07-22"
SOURCE_PROVENANCE = {
    "pad_ufes": {
        "name": "PAD-UFES-20",
        "doi": "10.1016/j.dib.2020.106221",
        "license": "CC-BY-4.0",
    },
    "hiba": {
        "name": "Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022)",
        "doi": "10.34970/587329",
        "isic_collection": 251,
        "license": "CC-BY",
    },
}
FAILED_GATE_KEYS = (
    "mean_selected_train_val_primary_source_mean_macro_f1_gap_lte_0_2000",
    "pooled_hiba_lesion_macro_f1_gte_0_5000",
    "pooled_primary_source_mean_macro_f1_gte_0_5600",
    "pooled_primary_worst_source_macro_f1_gte_0_5000",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the locked source-balanced PAD/HIBA ConvNeXt-Tiny demo checkpoint."
    )
    parser.add_argument("--pad-splits-dir", type=Path, default=DEFAULT_PAD_SPLITS_DIR)
    parser.add_argument("--hiba-splits-dir", type=Path, default=DEFAULT_HIBA_SPLITS_DIR)
    parser.add_argument("--cv-summary", type=Path, default=DEFAULT_CV_SUMMARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--torch-cache-dir", type=Path, default=DEFAULT_TORCH_CACHE_DIR)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_cv_summary(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_CV_SUMMARY_SHA256,
) -> dict[str, object]:
    path = resolve_project_path(Path(path))
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked PAD/HIBA development summary: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Locked PAD/HIBA development summary checksum drifted: "
            f"expected {expected_sha256}, observed {observed_sha256}."
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Locked PAD/HIBA development summary must be an object.")
    expected = {
        "architecture": ARCHITECTURE,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "pretrained_weights_id": "IMAGENET1K_V1",
        "preprocessing": PREPROCESSING,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": DEFAULT_SEED,
        "epochs": 15,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
    }
    mismatches = [
        f"{key}={summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    folds = summary.get("folds")
    selected_epochs = (
        tuple(int(fold["best_epoch"]) for fold in folds)
        if isinstance(folds, list) and all(isinstance(fold, dict) for fold in folds)
        else ()
    )
    if selected_epochs != EXPECTED_SELECTED_EPOCHS:
        mismatches.append(f"selected_epochs={selected_epochs!r}")
    decision_rules = summary.get("decision_rules")
    if not isinstance(decision_rules, dict) or decision_rules.get("all_pass") is not False:
        mismatches.append("decision_rules.all_pass did not remain false")
    elif any(decision_rules.get(key) is not False for key in FAILED_GATE_KEYS):
        mismatches.append("the four locked failed gate categories drifted")
    if mismatches:
        raise ValueError("Locked PAD/HIBA development evidence drifted: " + ", ".join(mismatches))
    return summary


def build_final_fit_rows(
    manifests: MultiSourceManifests,
    *,
    expected_manifest_fingerprint: str = EXPECTED_MANIFEST_FINGERPRINT,
) -> pd.DataFrame:
    if manifests.fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "PAD/HIBA manifest fingerprint drifted: "
            f"expected {expected_manifest_fingerprint}, observed {manifests.fingerprint}."
        )
    rows = manifests.folds[0].copy()
    if bool(rows["image_path"].astype(str).duplicated().any()):
        raise ValueError("Final-fit PAD/HIBA rows contain duplicate image paths.")
    raw_counts = {source: int((rows["source"] == source).sum()) for source in SOURCE_ORDER}
    effective_counts = {
        source: float(rows.loc[rows["source"] == source, "view_mass"].sum())
        for source in SOURCE_ORDER
    }
    if raw_counts != manifests.source_total_raw_image_counts:
        raise ValueError("Final-fit raw source counts do not match the locked manifests.")
    if effective_counts != manifests.source_total_effective_unit_counts:
        raise ValueError("Final-fit effective source counts do not match the locked manifests.")
    rows["split"] = "train"
    return rows.reset_index(drop=True)


def compact_training_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    by_source = metrics["by_source"]
    return {
        "loss": float(metrics["loss"]),
        "augmented_training_source_mean_macro_f1": float(metrics["primary_source_mean_macro_f1"]),
        "augmented_training_pad_image_macro_f1": float(by_source["pad_ufes"]["macro_f1"]),
        "augmented_training_hiba_lesion_macro_f1": float(metrics["hiba_lesion"]["macro_f1"]),
    }


def run_final_fit(args: argparse.Namespace) -> dict[str, object]:
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    cv_summary_path = resolve_project_path(args.cv_summary)
    cv_summary = load_locked_cv_summary(cv_summary_path)
    manifests = load_multi_source_manifests(args.pad_splits_dir, args.hiba_splits_dir)
    rows = build_final_fit_rows(manifests)
    device = get_device(args.device)
    set_seed(DEFAULT_SEED)
    torch_cache_dir = resolve_project_path(args.torch_cache_dir)
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_cache_dir))

    dataset = MultiSourceImageDataset(rows, "train")
    loader = build_loader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        seed=DEFAULT_SEED,
    )
    model = build_transfer_model(
        architecture=ARCHITECTURE,
        weights=PRETRAINED_WEIGHTS,
    ).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    hyperparameters = {
        "epochs": FINAL_EPOCHS,
        "epoch_rule": "median_of_locked_cv_selected_epochs",
        "locked_cv_selected_epochs": list(EXPECTED_SELECTED_EPOCHS),
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "optimizer": "AdamW",
        "schedule": "none",
        "augmentation_profile": AUGMENTATION_PROFILE,
        "label_smoothing": 0.0,
        "sampling": "random_shuffle_without_replacement",
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
    }
    history = []
    for epoch in range(1, FINAL_EPOCHS + 1):
        metrics = run_epoch(model, loader, criterion, device, optimizer=optimizer)
        compact = compact_training_metrics(metrics)
        history.append({"epoch": epoch, **compact})
        print(
            f"final_fit epoch={epoch}/{FINAL_EPOCHS} "
            f"loss={compact['loss']:.4f} "
            "metrics=augmented-training-sanity-only"
        )

    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": ARCHITECTURE,
        "input_mode": "image_only",
        "dataset": "pad_ufes_hiba",
        "dataset_role": "multisource_development_final_fit",
        "training_protocol": TRAINING_PROTOCOL,
        "model_version": MODEL_VERSION,
        "sources": list(SOURCE_ORDER),
        "source_provenance": SOURCE_PROVENANCE,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "pretrained_weights_id": pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS),
        "preprocessing": PREPROCESSING,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "manifest_fingerprint": manifests.fingerprint,
        "cv_summary_sha256": EXPECTED_CV_SUMMARY_SHA256,
        "cv_decision_all_pass": False,
        "selection_status": "owner_selected_despite_failed_preregistered_gates",
        "epoch": FINAL_EPOCHS,
        "seed": DEFAULT_SEED,
        "hyperparameters": hyperparameters,
        "source_total_raw_image_counts": manifests.source_total_raw_image_counts,
        "source_total_effective_unit_counts": manifests.source_total_effective_unit_counts,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    report = {
        "context": (
            "Owner-approved experimental demo final fit on PAD-UFES plus HIBA development data; "
            "not independent validation, diagnosis, clinical evidence, or medical readiness."
        ),
        "model_version": MODEL_VERSION,
        "architecture": ARCHITECTURE,
        "training_protocol": TRAINING_PROTOCOL,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "source_provenance": SOURCE_PROVENANCE,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "pretrained_weights_id": pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS),
        "preprocessing": PREPROCESSING,
        "manifest_fingerprint": manifests.fingerprint,
        "cv_summary": project_relative(cv_summary_path),
        "cv_summary_sha256": EXPECTED_CV_SUMMARY_SHA256,
        "cv_decision_rules": cv_summary["decision_rules"],
        "selection_status": "owner_selected_despite_failed_preregistered_gates",
        "hyperparameters": hyperparameters,
        "inference_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_total_raw_image_counts": manifests.source_total_raw_image_counts,
        "source_total_effective_unit_counts": manifests.source_total_effective_unit_counts,
        "training_history": history,
        "training_history_role": "augmented_training_sanity_only_not_performance_evidence",
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "independent_evaluation_performed": False,
        "milk10k_used": False,
        "privacy": {
            "aggregate_metrics_only": True,
            "per_image_predictions_written": False,
            "identifiers_or_paths_written": False,
        },
        "caveat": (
            "The source-balanced development experiment failed its four cross-source gate "
            "categories. Final fitting creates a runnable artifact but cannot improve or replace "
            "that evidence and does not establish consumer-photo generalization."
        ),
    }
    report_path = resolve_project_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Wrote experimental checkpoint sha256={checkpoint_sha256} "
        f"report={project_relative(report_path)}"
    )
    return report


def main() -> None:
    run_final_fit(parse_args())


if __name__ == "__main__":
    main()

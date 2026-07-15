"""Aggregate locked PAD-UFES grouped cross-validation reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ml.evaluation.metrics import per_class_metrics
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_pad_ufes import project_relative
from ml.training.prepare_pad_ufes_cv import DEFAULT_FOLDS, PROTOCOL
from ml.training.train import resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_ROOT = PROJECT_ROOT / "ml" / "runs" / "training" / "pad_ufes_resnet18-cv-seed42"
DEFAULT_OUT_PATH = DEFAULT_REPORTS_ROOT / "summary.json"
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate PAD-UFES grouped cross-validation reports."
    )
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--architecture", default="resnet18")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augmentation-profile", default="baseline")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr-schedule", default="none")
    parser.add_argument("--imbalance-strategy", default="inverse_frequency_loss")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def summarize_reports(
    reports_root: Path,
    out_path: Path,
    *,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = 42,
    architecture: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    augmentation_profile: str = "baseline",
    label_smoothing: float = 0.0,
    lr_schedule: str = "none",
    imbalance_strategy: str = "inverse_frequency_loss",
    weight_decay: float = 1e-4,
) -> dict[str, object]:
    reports_root = resolve_project_path(Path(reports_root))
    reports = []
    for fold_index in range(num_folds):
        report_path = reports_root / f"fold_{fold_index}" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing cross-validation report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _validate_report(
            report,
            fold_index=fold_index,
            num_folds=num_folds,
            seed=seed,
            architecture=architecture,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            augmentation_profile=augmentation_profile,
            label_smoothing=label_smoothing,
            lr_schedule=lr_schedule,
            imbalance_strategy=imbalance_strategy,
            weight_decay=weight_decay,
        )
        reports.append(report)

    expected_image_counts = {
        int(report["split_summary"]["cv_total_image_count"]) for report in reports
    }
    if len(expected_image_counts) != 1:
        raise ValueError("Cross-validation reports disagree on the total image count.")
    expected_image_count = expected_image_counts.pop()

    pooled_confusion = _pooled_confusion(reports)
    pooled_total = sum(sum(row) for row in pooled_confusion)
    if pooled_total != expected_image_count:
        raise ValueError(
            "Cross-validation test folds do not cover the dataset exactly once: "
            f"pooled_support={pooled_total} expected={expected_image_count}."
        )

    fold_metrics = {
        metric: _distribution([float(report["test"][metric]) for report in reports])
        for metric in METRIC_NAMES
    }
    per_class = {
        label: {
            metric: _distribution(
                [float(report["test"]["per_class"][label][metric]) for report in reports]
            )
            for metric in ("precision", "recall", "f1")
        }
        | {
            "support_by_fold": [
                int(report["test"]["per_class"][label]["support"]) for report in reports
            ],
            "total_support": sum(
                int(report["test"]["per_class"][label]["support"]) for report in reports
            ),
        }
        for label in PAD_UFES_NATIVE_LABELS
    }
    summary = {
        "context": (
            "Experimental PAD-UFES-native grouped cross-validation; not medical certainty."
        ),
        "protocol": PROTOCOL,
        "num_folds": num_folds,
        "seed": seed,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "architecture": architecture,
        "input_mode": "image_only",
        "pretrained_weights": "imagenet",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "augmentation_profile": augmentation_profile,
        "label_smoothing": label_smoothing,
        "lr_schedule": lr_schedule,
        "imbalance_strategy": imbalance_strategy,
        "weight_decay": weight_decay,
        "fold_metrics": fold_metrics,
        "best_validation_macro_f1": _distribution(
            [float(report["best_val_macro_f1"]) for report in reports]
        ),
        "per_class_fold_metrics": per_class,
        "pooled_test": _summarize_confusion(pooled_confusion),
        "folds": [
            {
                "fold_index": index,
                "best_epoch": int(report["best_epoch"]),
                "best_val_macro_f1": float(report["best_val_macro_f1"]),
                "test": {metric: float(report["test"][metric]) for metric in METRIC_NAMES},
            }
            for index, report in enumerate(reports)
        ],
        "caveat": (
            "Fold-level test outputs are aggregated only after the locked protocol completes; "
            "they must not be used for mid-protocol tuning."
        ),
    }

    out_path = resolve_project_path(Path(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _validate_report(
    report: dict[str, object],
    *,
    fold_index: int,
    num_folds: int,
    seed: int,
    architecture: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    augmentation_profile: str,
    label_smoothing: float,
    lr_schedule: str,
    imbalance_strategy: str,
    weight_decay: float,
) -> None:
    expected = {
        "architecture": architecture,
        "input_mode": "image_only",
        "pretrained_weights": "imagenet",
        "selection_metric": "val_macro_f1",
        "seed": seed,
    }
    mismatches = [
        f"{key}={report.get(key)!r}"
        for key, expected_value in expected.items()
        if report.get(key) != expected_value
    ]
    if report.get("labels") != list(PAD_UFES_NATIVE_LABELS):
        mismatches.append("labels do not match PAD-UFES-native order")

    hyperparameters = report.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}
    legacy_defaults = {
        "augmentation_profile": "baseline",
        "label_smoothing": 0.0,
        "lr_schedule": "none",
        "weight_decay": 1e-4,
        "imbalance_strategy": "inverse_frequency_loss",
        "epochs": 15,
        "batch_size": 32,
        "learning_rate": 1e-4,
    }
    training_expected = {
        "augmentation_profile": augmentation_profile,
        "label_smoothing": label_smoothing,
        "lr_schedule": lr_schedule,
        "weight_decay": weight_decay,
        "imbalance_strategy": imbalance_strategy,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    mismatches.extend(
        f"hyperparameters.{key}={hyperparameters.get(key, legacy_defaults[key])!r}"
        for key, expected_value in training_expected.items()
        if hyperparameters.get(key, legacy_defaults[key]) != expected_value
    )

    split_summary = report.get("split_summary")
    if not isinstance(split_summary, dict):
        mismatches.append("split_summary is missing")
    else:
        split_expected = {
            "protocol": PROTOCOL,
            "num_folds": num_folds,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": (fold_index + 1) % num_folds,
        }
        mismatches.extend(
            f"split_summary.{key}={split_summary.get(key)!r}"
            for key, expected_value in split_expected.items()
            if split_summary.get(key) != expected_value
        )

    if mismatches:
        raise ValueError(
            f"Cross-validation report fold_{fold_index} violates the locked protocol: "
            f"{', '.join(mismatches)}"
        )


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence.")
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _pooled_confusion(reports: list[dict[str, object]]) -> list[list[int]]:
    size = len(PAD_UFES_NATIVE_LABELS)
    pooled = [[0 for _ in range(size)] for _ in range(size)]
    for report in reports:
        confusion = report["test"]["confusion_matrix"]
        if len(confusion) != size or any(len(row) != size for row in confusion):
            raise ValueError("A fold confusion matrix does not match the native label count.")
        for row_index in range(size):
            for column_index in range(size):
                pooled[row_index][column_index] += int(confusion[row_index][column_index])
    return pooled


def _summarize_confusion(confusion: list[list[int]]) -> dict[str, object]:
    per_class = per_class_metrics(confusion, labels=PAD_UFES_NATIVE_LABELS)
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(PAD_UFES_NATIVE_LABELS)))
    precision_values = [float(per_class[label]["precision"]) for label in PAD_UFES_NATIVE_LABELS]
    recall_values = [float(per_class[label]["recall"]) for label in PAD_UFES_NATIVE_LABELS]
    f1_values = [float(per_class[label]["f1"]) for label in PAD_UFES_NATIVE_LABELS]
    return {
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": statistics.fmean(recall_values),
        "macro_precision": statistics.fmean(precision_values),
        "macro_recall": statistics.fmean(recall_values),
        "macro_f1": statistics.fmean(f1_values),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "total_support": total,
    }


def main() -> None:
    args = parse_args()
    summary = summarize_reports(
        args.reports_root,
        args.out,
        num_folds=args.folds,
        seed=args.seed,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augmentation_profile=args.augmentation_profile,
        label_smoothing=args.label_smoothing,
        lr_schedule=args.lr_schedule,
        imbalance_strategy=args.imbalance_strategy,
        weight_decay=args.weight_decay,
    )
    fold_macro_f1 = summary["fold_metrics"]["macro_f1"]
    print(
        "Wrote PAD-UFES CV summary to "
        f"{project_relative(resolve_project_path(args.out))}; "
        f"macro_f1_mean={fold_macro_f1['mean']:.4f} "
        f"population_std={fold_macro_f1['population_std']:.4f}"
    )


if __name__ == "__main__":
    main()

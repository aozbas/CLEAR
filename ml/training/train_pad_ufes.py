"""Train an image-only PAD-UFES-native transfer-learning baseline.

Run from the project root after preparing a patient-grouped split:
    python -m ml.training.train_pad_ufes

Outputs are experimental classification artifacts, not medical conclusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS, VALID_SPLITS
from ml.preprocessing import get_transforms
from ml.training.train import (
    build_loader,
    class_weights,
    get_device,
    resolve_project_path,
    run_epoch,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_CSV = (
    PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes_native_training.csv"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ml" / "models" / "pad_ufes_resnet18.pt"
DEFAULT_RUN_DIR = PROJECT_ROOT / "ml" / "runs" / "training" / "pad_ufes_resnet18"
DEFAULT_SEED = 42
REQUIRED_COLUMNS = {"split", "image_path", "label"}
WEIGHT_CHOICES = ("imagenet", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an image-only PAD-UFES-native ResNet18 baseline."
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--weights", choices=WEIGHT_CHOICES, default="imagenet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def load_training_split(split_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    split_csv = resolve_project_path(Path(split_csv))
    rows = pd.read_csv(split_csv)
    missing_columns = REQUIRED_COLUMNS.difference(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"PAD-UFES training split is missing columns: {missing}")

    unknown_splits = sorted(set(rows["split"]) - set(VALID_SPLITS))
    if unknown_splits:
        raise ValueError(f"PAD-UFES training split has unknown split values: {unknown_splits}")
    unknown_labels = sorted(set(rows["label"]) - set(PAD_UFES_NATIVE_LABELS))
    if unknown_labels:
        raise ValueError(f"PAD-UFES training split has non-native labels: {unknown_labels}")

    duplicate_splits = rows.groupby("image_path")["split"].nunique()
    if not duplicate_splits.empty and int(duplicate_splits.max()) > 1:
        raise ValueError("A PAD-UFES image appears in multiple training splits.")
    if bool(rows["image_path"].duplicated().any()):
        raise ValueError("A PAD-UFES image appears more than once in the training split.")

    coverage = pd.crosstab(rows["split"], rows["label"]).reindex(
        index=VALID_SPLITS,
        columns=PAD_UFES_NATIVE_LABELS,
        fill_value=0,
    )
    missing_coverage = [
        f"{split}/{label}"
        for split in VALID_SPLITS
        for label in PAD_UFES_NATIVE_LABELS
        if int(coverage.loc[split, label]) == 0
    ]
    if missing_coverage:
        raise ValueError(
            f"PAD-UFES training split has missing label coverage: {', '.join(missing_coverage)}"
        )

    summary_path = split_csv.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing grouped-split summary: {summary_path}. "
            "Prepare the CSV with ml.training.prepare_pad_ufes --split-strategy patient."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_split_summary(summary, rows)
    return rows, summary


def _validate_split_summary(summary: dict[str, object], rows: pd.DataFrame) -> None:
    expected_values = {
        "dataset": "pad_ufes",
        "label_mode": "native",
        "split_strategy": "patient",
        "group_key": "patient_id",
        "patient_overlap_count": 0,
        "patient_lesion_overlap_count": 0,
    }
    mismatches = [
        f"{key}={summary.get(key)!r}"
        for key, expected in expected_values.items()
        if summary.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "PAD-UFES split summary is not a verified patient-grouped native split: "
            f"{', '.join(mismatches)}"
        )
    if summary.get("image_count") != len(rows):
        raise ValueError("PAD-UFES split summary image_count does not match the split CSV.")

    expected_split_counts = {split: int((rows["split"] == split).sum()) for split in VALID_SPLITS}
    if summary.get("images_by_split") != expected_split_counts:
        raise ValueError("PAD-UFES split summary counts do not match the split CSV.")


def sample_rows(rows: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples < len(PAD_UFES_NATIVE_LABELS):
        raise ValueError(
            f"max_samples={max_samples} is too small for {len(PAD_UFES_NATIVE_LABELS)} labels."
        )
    required = []
    remaining = rows
    for offset, label in enumerate(PAD_UFES_NATIVE_LABELS):
        label_rows = rows[rows["label"] == label]
        sample = label_rows.sample(n=1, random_state=seed + offset)
        required.append(sample)
        remaining = remaining.drop(index=sample.index)

    remaining_count = max_samples - len(required)
    if remaining_count > 0:
        required.append(remaining.sample(n=remaining_count, random_state=seed))
    return pd.concat(required).sort_values("image_path").reset_index(drop=True)


class PadUfesDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        split: str,
        *,
        max_samples: int | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        split_rows = rows[rows["split"] == split].copy()
        if max_samples is not None and max_samples < len(split_rows):
            split_rows = sample_rows(split_rows, max_samples, seed)

        self.rows = split_rows.reset_index(drop=True)
        self.label_to_idx = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
        self.transform = get_transforms("train" if split == "train" else "val")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        image_path = resolve_project_path(Path(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.label_to_idx[row["label"]]

    def labels(self) -> list[int]:
        return [self.label_to_idx[label] for label in self.rows["label"].tolist()]


def build_transfer_model(*, weights: str = "imagenet") -> nn.Module:
    if weights == "imagenet":
        torchvision_weights = models.ResNet18_Weights.DEFAULT
    elif weights == "none":
        torchvision_weights = None
    else:
        raise ValueError(f"Unknown ResNet18 weights: {weights}")

    model = models.resnet18(weights=torchvision_weights)
    model.fc = nn.Linear(model.fc.in_features, len(PAD_UFES_NATIVE_LABELS))
    return model


def add_macro_metrics(metrics: dict[str, object]) -> dict[str, object]:
    per_class = metrics["per_class"]
    if not isinstance(per_class, dict):
        raise TypeError("per_class metrics must be a dictionary.")
    precisions = [float(per_class[label]["precision"]) for label in PAD_UFES_NATIVE_LABELS]
    recalls = [float(per_class[label]["recall"]) for label in PAD_UFES_NATIVE_LABELS]
    f1_scores = [float(per_class[label]["f1"]) for label in PAD_UFES_NATIVE_LABELS]
    return {
        **metrics,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_scores) / len(f1_scores),
    }


def print_metrics(split: str, metrics: dict[str, object]) -> None:
    print(
        f"{split}: loss={float(metrics['loss']):.4f} "
        f"accuracy={float(metrics['accuracy']):.4f} "
        f"balanced_accuracy={float(metrics['balanced_accuracy']):.4f} "
        f"macro_f1={float(metrics['macro_f1']):.4f}"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    epoch: int,
    val_metrics: dict[str, object],
    weights: str,
    seed: int,
    hyperparameters: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "resnet18",
            "input_mode": "image_only",
            "dataset": "pad_ufes",
            "label_set": "pad_ufes_native",
            "labels": list(PAD_UFES_NATIVE_LABELS),
            "label_to_idx": {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)},
            "pretrained_weights": weights,
            "pretrained_weights_id": (
                models.ResNet18_Weights.DEFAULT.name if weights == "imagenet" else None
            ),
            "preprocessing": "resize_224_imagenet_normalization",
            "selection_metric": "val_macro_f1",
            "epoch": epoch,
            "seed": seed,
            "hyperparameters": hyperparameters,
            "val_metrics": val_metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    split_csv = resolve_project_path(args.split_csv)
    checkpoint = resolve_project_path(args.checkpoint)
    run_dir = resolve_project_path(args.run_dir)
    rows, split_summary = load_training_split(split_csv)

    train_dataset = PadUfesDataset(
        rows,
        "train",
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_dataset = PadUfesDataset(
        rows,
        "val",
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    test_dataset = PadUfesDataset(
        rows,
        "test",
        max_samples=args.max_test_samples,
        seed=args.seed,
    )
    print(f"Using device: {device}")
    print(
        f"Dataset sizes: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )

    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_transfer_model(weights=args.weights).to(device)
    loss_weights = class_weights(train_dataset.labels(), len(PAD_UFES_NATIVE_LABELS), device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "class_weighting": "inverse_frequency_from_train_split",
        "class_weights": {
            label: float(loss_weights[index].item())
            for index, label in enumerate(PAD_UFES_NATIVE_LABELS)
        },
        "optimizer": "AdamW",
    }

    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = add_macro_metrics(
            run_epoch(
                model,
                train_loader,
                criterion,
                device,
                list(PAD_UFES_NATIVE_LABELS),
                optimizer,
            )
        )
        val_metrics = add_macro_metrics(
            run_epoch(
                model,
                val_loader,
                criterion,
                device,
                list(PAD_UFES_NATIVE_LABELS),
            )
        )
        print_metrics("train", train_metrics)
        print_metrics("val", val_metrics)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        macro_f1 = float(val_metrics["macro_f1"])
        val_loss = float(val_metrics["loss"])
        improved = macro_f1 > best_macro_f1 or (
            macro_f1 == best_macro_f1 and val_loss < best_val_loss
        )
        if improved:
            best_macro_f1 = macro_f1
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                checkpoint,
                model,
                epoch=epoch,
                val_metrics=val_metrics,
                weights=args.weights,
                seed=args.seed,
                hyperparameters=hyperparameters,
            )
            print(f"Saved checkpoint: {checkpoint}")

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    test_metrics = add_macro_metrics(
        run_epoch(
            model,
            test_loader,
            criterion,
            device,
            list(PAD_UFES_NATIVE_LABELS),
        )
    )
    print(f"\nBest epoch: {best_epoch}")
    print_metrics("test", test_metrics)

    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "context": "Experimental PAD-UFES-native image classification; not medical certainty.",
        "architecture": "resnet18",
        "input_mode": "image_only",
        "pretrained_weights": args.weights,
        "pretrained_weights_id": (
            models.ResNet18_Weights.DEFAULT.name if args.weights == "imagenet" else None
        ),
        "preprocessing": "resize_224_imagenet_normalization",
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "seed": args.seed,
        "split_csv": str(split_csv),
        "split_summary": split_summary,
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "hyperparameters": hyperparameters,
        "best_epoch": best_epoch,
        "selection_metric": "val_macro_f1",
        "best_val_macro_f1": best_macro_f1,
        "history": history,
        "test": test_metrics,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

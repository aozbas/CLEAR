"""Train an image-only PAD-UFES-native transfer-learning classifier.

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS, VALID_SPLITS
from ml.preprocessing import PAD_UFES_AUGMENTATION_PROFILES, get_pad_ufes_transforms
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
ARCHITECTURES = (
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "convnext_tiny",
)
LR_SCHEDULES = ("none", "cosine")
IMBALANCE_STRATEGIES = (
    "inverse_frequency_loss",
    "unweighted_loss",
    "balanced_sampler",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an image-only PAD-UFES-native transfer-learning classifier."
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument("--weights", choices=WEIGHT_CHOICES, default="imagenet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--augmentation-profile",
        choices=PAD_UFES_AUGMENTATION_PROFILES,
        default="baseline",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="none")
    parser.add_argument(
        "--imbalance-strategy",
        choices=IMBALANCE_STRATEGIES,
        default="inverse_frequency_loss",
    )
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
        augmentation_profile: str = "baseline",
    ) -> None:
        split_rows = rows[rows["split"] == split].copy()
        if max_samples is not None and max_samples < len(split_rows):
            split_rows = sample_rows(split_rows, max_samples, seed)

        self.rows = split_rows.reset_index(drop=True)
        self.label_to_idx = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
        self.transform = get_pad_ufes_transforms(
            "train" if split == "train" else "val",
            augmentation_profile=augmentation_profile,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        image_path = resolve_project_path(Path(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.label_to_idx[row["label"]]

    def labels(self) -> list[int]:
        return [self.label_to_idx[label] for label in self.rows["label"].tolist()]


def build_balanced_sampler(
    labels: list[int],
    num_classes: int,
    seed: int,
) -> WeightedRandomSampler:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError(
            f"Every class needs at least one training example. Counts: {counts.tolist()}"
        )
    per_example_weights = counts.reciprocal()[torch.tensor(labels)]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=per_example_weights,
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def build_train_loader(
    dataset: PadUfesDataset,
    batch_size: int,
    *,
    imbalance_strategy: str,
    num_workers: int,
    seed: int,
) -> DataLoader:
    if imbalance_strategy == "balanced_sampler":
        sampler = build_balanced_sampler(
            dataset.labels(),
            len(PAD_UFES_NATIVE_LABELS),
            seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    if imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy: {imbalance_strategy!r}")
    return build_loader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def resolve_torchvision_weights(architecture: str, weights: str) -> object | None:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture!r}")
    if weights == "none":
        return None
    if weights != "imagenet":
        raise ValueError(f"Unknown pretrained weights: {weights!r}")
    weight_enums = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
    }
    return weight_enums[architecture]


def pretrained_weights_id(architecture: str, weights: str) -> str | None:
    resolved = resolve_torchvision_weights(architecture, weights)
    return resolved.name if resolved is not None else None


def build_transfer_model(
    *,
    architecture: str = "resnet18",
    weights: str = "imagenet",
) -> nn.Module:
    torchvision_weights = resolve_torchvision_weights(architecture, weights)
    num_classes = len(PAD_UFES_NATIVE_LABELS)
    if architecture == "resnet18":
        model = models.resnet18(weights=torchvision_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=torchvision_weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=torchvision_weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=torchvision_weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")
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
    augmentation_profile: str = "baseline",
    architecture: str = "resnet18",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": architecture,
            "input_mode": "image_only",
            "dataset": "pad_ufes",
            "label_set": "pad_ufes_native",
            "labels": list(PAD_UFES_NATIVE_LABELS),
            "label_to_idx": {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)},
            "pretrained_weights": weights,
            "pretrained_weights_id": pretrained_weights_id(architecture, weights),
            "preprocessing": "resize_224_imagenet_normalization",
            "augmentation_profile": augmentation_profile,
            "selection_metric": "val_macro_f1",
            "epoch": epoch,
            "seed": seed,
            "hyperparameters": hyperparameters,
            "val_metrics": val_metrics,
        },
        path,
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if schedule == "none":
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown learning-rate schedule: {schedule!r}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in the range [0, 1).")
    set_seed(args.seed)
    device = get_device(args.device)
    split_csv = resolve_project_path(args.split_csv)
    default_checkpoint = (
        DEFAULT_CHECKPOINT
        if args.architecture == "resnet18"
        else PROJECT_ROOT / "ml" / "models" / f"pad_ufes_{args.architecture}.pt"
    )
    default_run_dir = (
        DEFAULT_RUN_DIR
        if args.architecture == "resnet18"
        else PROJECT_ROOT / "ml" / "runs" / "training" / f"pad_ufes_{args.architecture}"
    )
    checkpoint = resolve_project_path(args.checkpoint or default_checkpoint)
    run_dir = resolve_project_path(args.run_dir or default_run_dir)
    rows, split_summary = load_training_split(split_csv)

    train_dataset = PadUfesDataset(
        rows,
        "train",
        max_samples=args.max_train_samples,
        seed=args.seed,
        augmentation_profile=args.augmentation_profile,
    )
    val_dataset = PadUfesDataset(
        rows,
        "val",
        max_samples=args.max_val_samples,
        seed=args.seed,
        augmentation_profile=args.augmentation_profile,
    )
    test_dataset = PadUfesDataset(
        rows,
        "test",
        max_samples=args.max_test_samples,
        seed=args.seed,
        augmentation_profile=args.augmentation_profile,
    )
    print(f"Using device: {device}")
    print(
        f"Dataset sizes: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )

    train_loader = build_train_loader(
        train_dataset,
        args.batch_size,
        imbalance_strategy=args.imbalance_strategy,
        num_workers=args.num_workers,
        seed=args.seed,
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

    model = build_transfer_model(
        architecture=args.architecture,
        weights=args.weights,
    ).to(device)
    loss_weights = (
        class_weights(train_dataset.labels(), len(PAD_UFES_NATIVE_LABELS), device)
        if args.imbalance_strategy == "inverse_frequency_loss"
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=args.lr_schedule,
        epochs=args.epochs,
    )
    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "augmentation_profile": args.augmentation_profile,
        "label_smoothing": args.label_smoothing,
        "lr_schedule": args.lr_schedule,
        "imbalance_strategy": args.imbalance_strategy,
        "class_weighting": (
            "inverse_frequency_from_train_split" if loss_weights is not None else "none"
        ),
        "class_weights": (
            {
                label: float(loss_weights[index].item())
                for index, label in enumerate(PAD_UFES_NATIVE_LABELS)
            }
            if loss_weights is not None
            else None
        ),
        "sampling": (
            "inverse_frequency_with_replacement"
            if args.imbalance_strategy == "balanced_sampler"
            else "random_shuffle_without_replacement"
        ),
        "optimizer": "AdamW",
    }

    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        learning_rate = float(optimizer.param_groups[0]["lr"])
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
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

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
                augmentation_profile=args.augmentation_profile,
                architecture=args.architecture,
            )
            print(f"Saved checkpoint: {checkpoint}")
        if scheduler is not None:
            scheduler.step()

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
        "architecture": args.architecture,
        "input_mode": "image_only",
        "pretrained_weights": args.weights,
        "pretrained_weights_id": pretrained_weights_id(args.architecture, args.weights),
        "preprocessing": "resize_224_imagenet_normalization",
        "augmentation_profile": args.augmentation_profile,
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

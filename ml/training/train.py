"""Train the Phase 1 binary HAM10000 classifier.

Run from the project root:
    python -m ml.training.train
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.models.classifier import build_model
from ml.preprocessing import get_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_CSV = PROJECT_ROOT / "ml" / "data" / "splits" / "ham10000.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ml" / "models" / "lesion_classifier_binary.pt"
DEFAULT_SEED = 42

LABELS = ["non_suspicious", "suspicious"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
SUSPICIOUS_CANONICAL_LABELS = {
    "melanoma",
    "basal_cell_carcinoma",
    "actinic_keratosis",
}
NON_SUSPICIOUS_CANONICAL_LABELS = {
    "nevus",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary HAM10000 classifier.")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def canonical_to_binary(label: str) -> str:
    if label in SUSPICIOUS_CANONICAL_LABELS:
        return "suspicious"
    if label in NON_SUSPICIOUS_CANONICAL_LABELS:
        return "non_suspicious"
    raise ValueError(f"Unknown canonical label for binary training: {label!r}")


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Ham10000BinaryDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        split: str,
        max_samples: int | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        rows = pd.read_csv(split_csv)
        rows = rows[rows["split"] == split].copy()
        if rows.empty:
            raise ValueError(f"No rows found for split={split!r} in {split_csv}")

        try:
            rows["binary_label"] = rows["label"].apply(canonical_to_binary)
        except ValueError as exc:
            raise ValueError(f"Unknown labels in split CSV {split_csv}") from exc

        if max_samples is not None and max_samples < len(rows):
            rows = rows.sample(n=max_samples, random_state=seed).sort_values("image_path")

        self.rows = rows.reset_index(drop=True)
        self.transform = get_transforms("train" if split == "train" else "val")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        image_path = resolve_project_path(Path(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        label = LABEL_TO_IDX[row["binary_label"]]
        return self.transform(image), label

    def labels(self) -> list[int]:
        return [LABEL_TO_IDX[label] for label in self.rows["binary_label"].tolist()]


def class_weights(labels: Iterable[int], device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(list(labels)), minlength=len(LABELS)).float()
    if (counts == 0).any():
        raise ValueError(f"Every class needs at least one training example. Counts: {counts.tolist()}")
    weights = counts.sum() / (len(LABELS) * counts)
    return weights.to(device)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total = 0
    confusion = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.long)

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * targets.size(0)
            total_correct += (predictions == targets).sum().item()
            total += targets.size(0)

            for truth, pred in zip(targets.cpu(), predictions.cpu(), strict=True):
                confusion[int(truth), int(pred)] += 1

    return metrics_from_confusion(total_loss / total, total_correct / total, confusion)


def metrics_from_confusion(loss: float, accuracy: float, confusion: torch.Tensor) -> dict:
    per_class = {}
    for idx, label in enumerate(LABELS):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum().item() - tp)
        fn = int(confusion[idx, :].sum().item() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[idx, :].sum().item()),
        }

    return {
        "loss": loss,
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def print_metrics(split: str, metrics: dict) -> None:
    print(f"{split}: loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f}")
    for label, class_metrics in metrics["per_class"].items():
        print(
            "  "
            f"{label}: precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} "
            f"f1={class_metrics['f1']:.4f} "
            f"support={class_metrics['support']}"
        )


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "labels": LABELS,
            "label_to_idx": LABEL_TO_IDX,
            "binary_groups": {
                "suspicious": sorted(SUSPICIOUS_CANONICAL_LABELS),
                "non_suspicious": sorted(NON_SUSPICIOUS_CANONICAL_LABELS),
            },
            "val_metrics": val_metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    split_csv = resolve_project_path(args.split_csv)
    checkpoint = resolve_project_path(args.checkpoint)

    train_dataset = Ham10000BinaryDataset(
        split_csv,
        "train",
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_dataset = Ham10000BinaryDataset(
        split_csv,
        "val",
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    test_dataset = Ham10000BinaryDataset(
        split_csv,
        "test",
        max_samples=args.max_test_samples,
        seed=args.seed,
    )

    print(
        "Dataset sizes: "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )

    train_loader = build_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dataset.labels(), device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        print_metrics("train", train_metrics)
        print_metrics("val", val_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(checkpoint, model, epoch, val_metrics)
            print(f"Saved checkpoint: {checkpoint}")

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device)

    print(f"\nBest epoch: {best_epoch}")
    print_metrics("test", test_metrics)


if __name__ == "__main__":
    main()

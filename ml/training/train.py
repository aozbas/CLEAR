"""Shared utilities used by the phone-photo training experiments."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 42


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


def class_weights(labels: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(list(labels)), minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError(
            f"Every class needs at least one training example. Counts: {counts.tolist()}"
        )
    weights = counts.sum() / (num_classes * counts)
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
    label_names: list[str],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total = 0
    confusion = torch.zeros((len(label_names), len(label_names)), dtype=torch.long)

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

            for truth, prediction in zip(targets.cpu(), predictions.cpu(), strict=True):
                confusion[int(truth), int(prediction)] += 1

    if total == 0:
        raise ValueError("Cannot compute training metrics for an empty loader.")
    return metrics_from_confusion(
        total_loss / total,
        total_correct / total,
        confusion,
        label_names,
    )


def metrics_from_confusion(
    loss: float,
    accuracy: float,
    confusion: torch.Tensor,
    label_names: list[str],
) -> dict[str, object]:
    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(label_names):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum().item() - true_positive)
        false_negative = int(confusion[index, :].sum().item() - true_positive)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum().item()),
        }

    return {
        "loss": loss,
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def print_metrics(split: str, metrics: dict[str, object]) -> None:
    loss = float(metrics["loss"])
    accuracy = float(metrics["accuracy"])
    print(f"{split}: loss={loss:.4f} accuracy={accuracy:.4f}")
    per_class = metrics["per_class"]
    if not isinstance(per_class, dict):
        raise ValueError("per_class metrics must be a dictionary.")
    for label, values in per_class.items():
        if not isinstance(values, dict):
            raise ValueError(f"Metrics for {label!r} must be a dictionary.")
        print(
            "  "
            f"{label}: precision={float(values['precision']):.4f} "
            f"recall={float(values['recall']):.4f} "
            f"f1={float(values['f1']):.4f} "
            f"support={int(values['support'])}"
        )

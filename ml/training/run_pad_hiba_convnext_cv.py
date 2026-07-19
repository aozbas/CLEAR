"""Run the locked source-balanced PAD-UFES + HIBA ConvNeXt-Tiny development experiment.

Both sources are development data in this workflow. The output can nominate one candidate for a
separately preregistered untouched evaluation, but it is not external validation, deployment
evidence, a medical conclusion, or a production model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.metrics import per_class_metrics
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS, VALID_SPLITS
from ml.preprocessing import get_pad_ufes_transforms
from ml.training.prepare_hiba_cv import PROTOCOL as HIBA_PROTOCOL
from ml.training.prepare_pad_ufes import project_relative
from ml.training.prepare_pad_ufes_cv import DEFAULT_FOLDS
from ml.training.prepare_pad_ufes_cv import PROTOCOL as PAD_PROTOCOL
from ml.training.train import get_device, resolve_project_path, set_seed
from ml.training.train_pad_ufes import build_transfer_model, pretrained_weights_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAD_SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes_native_cv"
DEFAULT_HIBA_SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "hiba_multisource_cv"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "ml" / "runs" / "training" / "pad_hiba_convnext_tiny-cv-seed42"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "ml" / "models" / "pad_hiba_convnext_tiny_cv_seed42"
DEFAULT_TORCH_CACHE_DIR = PROJECT_ROOT / "ml" / "model_cache" / "torch"
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
ARCHITECTURE = "convnext_tiny"
PRETRAINED_WEIGHTS = "imagenet"
PREPROCESSING = "resize_224_imagenet_normalization"
AUGMENTATION_PROFILE = "baseline"
SOURCE_ORDER = ("pad_ufes", "hiba")
SOURCE_CLASS_WEIGHTING = "equal_total_effective_weight_per_source_class_cell"
HIBA_VIEW_WEIGHTING = "equal_total_mass_per_lesion"
SELECTION_METRIC = "val_pad_image_hiba_lesion_source_mean_macro_f1"
EXPECTED_SOURCE_RAW_IMAGE_COUNTS = {"pad_ufes": 2_298, "hiba": 309}
EXPECTED_SOURCE_EFFECTIVE_UNIT_COUNTS = {"pad_ufes": 2_298.0, "hiba": 308.0}
PAD_REQUIRED_COLUMNS = {"split", "image_path", "label"}
HIBA_REQUIRED_COLUMNS = {
    "split",
    "source",
    "image_path",
    "label",
    "patient_id",
    "lesion_id",
    "isic_id",
}


@dataclass(frozen=True)
class MultiSourceManifests:
    folds: tuple[pd.DataFrame, ...]
    pad_fold_summaries: tuple[dict[str, object], ...]
    hiba_fold_summaries: tuple[dict[str, object], ...]
    fingerprint: str
    source_total_raw_image_counts: dict[str, int]
    source_total_effective_unit_counts: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked PAD/HIBA source-balanced ConvNeXt-Tiny development CV."
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_pad_unit_id(image_path: str) -> str:
    return "pad_" + hashlib.sha256(str(image_path).encode()).hexdigest()


def load_multi_source_manifests(
    pad_splits_dir: Path,
    hiba_splits_dir: Path,
    *,
    num_folds: int = DEFAULT_FOLDS,
    expected_raw_counts: Mapping[str, int] = EXPECTED_SOURCE_RAW_IMAGE_COUNTS,
    expected_effective_counts: Mapping[str, float] = EXPECTED_SOURCE_EFFECTIVE_UNIT_COUNTS,
) -> MultiSourceManifests:
    pad_splits_dir = resolve_project_path(Path(pad_splits_dir))
    hiba_splits_dir = resolve_project_path(Path(hiba_splits_dir))
    folds = []
    pad_summaries = []
    hiba_summaries = []
    for fold_index in range(num_folds):
        pad_path = pad_splits_dir / f"fold_{fold_index}.csv"
        hiba_path = hiba_splits_dir / f"fold_{fold_index}.csv"
        pad_summary_path = pad_path.with_suffix(".summary.json")
        hiba_summary_path = hiba_path.with_suffix(".summary.json")
        for path in (pad_path, hiba_path, pad_summary_path, hiba_summary_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing multi-source fold input: {path}")

        pad = pd.read_csv(pad_path, dtype=str, keep_default_na=False)
        hiba = pd.read_csv(hiba_path, dtype=str, keep_default_na=False)
        missing_pad = PAD_REQUIRED_COLUMNS.difference(pad.columns)
        missing_hiba = HIBA_REQUIRED_COLUMNS.difference(hiba.columns)
        if missing_pad:
            raise ValueError(f"PAD fold_{fold_index} is missing: {', '.join(sorted(missing_pad))}")
        if missing_hiba:
            raise ValueError(
                f"HIBA fold_{fold_index} is missing: {', '.join(sorted(missing_hiba))}"
            )
        pad_summary = json.loads(pad_summary_path.read_text(encoding="utf-8"))
        hiba_summary = json.loads(hiba_summary_path.read_text(encoding="utf-8"))
        _validate_split_summary(
            pad_summary,
            protocol=PAD_PROTOCOL,
            fold_index=fold_index,
            num_folds=num_folds,
            name="pad",
        )
        _validate_split_summary(
            hiba_summary,
            protocol=HIBA_PROTOCOL,
            fold_index=fold_index,
            num_folds=num_folds,
            name="hiba",
        )

        pad = pad[["split", "image_path", "label"]].copy()
        pad.insert(1, "source", "pad_ufes")
        pad["unit_id"] = pad["image_path"].map(_stable_pad_unit_id)
        pad["lesion_id"] = pad["unit_id"]
        pad["view_mass"] = 1.0
        hiba = hiba[
            [
                "split",
                "source",
                "image_path",
                "label",
                "patient_id",
                "lesion_id",
                "isic_id",
            ]
        ].copy()
        hiba["unit_id"] = "hiba_" + hiba["lesion_id"].map(
            lambda value: hashlib.sha256(str(value).encode()).hexdigest()
        )
        lesion_image_counts = hiba.groupby("lesion_id")["image_path"].transform("count")
        hiba["view_mass"] = 1.0 / lesion_image_counts.astype(float)
        hiba = hiba[["split", "source", "image_path", "label", "unit_id", "lesion_id", "view_mass"]]
        rows = pd.concat([pad, hiba], ignore_index=True)
        _validate_fold_rows(rows, fold_index=fold_index)
        folds.append(rows)
        pad_summaries.append(pad_summary)
        hiba_summaries.append(hiba_summary)

    _validate_rotating_coverage(folds, num_folds=num_folds)
    first = folds[0]
    raw_counts = {source: int((first["source"] == source).sum()) for source in SOURCE_ORDER}
    effective_counts = {
        source: float(first.loc[first["source"] == source, "view_mass"].sum())
        for source in SOURCE_ORDER
    }
    expected_raw = dict(expected_raw_counts)
    expected_effective = dict(expected_effective_counts)
    if raw_counts != expected_raw or effective_counts != expected_effective:
        raise ValueError(
            f"PAD/HIBA source counts drifted: raw={raw_counts!r} effective={effective_counts!r}."
        )
    fingerprint = _manifest_fingerprint(folds)
    return MultiSourceManifests(
        folds=tuple(folds),
        pad_fold_summaries=tuple(pad_summaries),
        hiba_fold_summaries=tuple(hiba_summaries),
        fingerprint=fingerprint,
        source_total_raw_image_counts=raw_counts,
        source_total_effective_unit_counts=effective_counts,
    )


def _validate_split_summary(
    summary: object,
    *,
    protocol: str,
    fold_index: int,
    num_folds: int,
    name: str,
) -> None:
    if not isinstance(summary, dict):
        raise ValueError(f"{name} fold summary must be an object.")
    expected = {
        "protocol": protocol,
        "num_folds": num_folds,
        "fold_index": fold_index,
        "test_outer_fold": fold_index,
        "validation_outer_fold": (fold_index + 1) % num_folds,
    }
    mismatches = [
        f"{key}={summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if summary.get("patient_overlap_count") != 0:
        mismatches.append(f"patient_overlap_count={summary.get('patient_overlap_count')!r}")
    if name == "hiba" and summary.get("lesion_overlap_count") != 0:
        mismatches.append(f"lesion_overlap_count={summary.get('lesion_overlap_count')!r}")
    if mismatches:
        raise ValueError(f"{name} fold_{fold_index} summary drifted: {', '.join(mismatches)}")


def _validate_fold_rows(rows: pd.DataFrame, *, fold_index: int) -> None:
    if set(rows["source"]) != set(SOURCE_ORDER):
        raise ValueError(f"fold_{fold_index} does not contain both locked sources.")
    if set(rows["split"]) != set(VALID_SPLITS):
        raise ValueError(f"fold_{fold_index} does not contain all split roles.")
    if set(rows["label"]) != set(PAD_UFES_NATIVE_LABELS):
        raise ValueError(f"fold_{fold_index} does not contain all native labels.")
    if bool(rows["image_path"].astype(str).duplicated().any()):
        raise ValueError(f"fold_{fold_index} contains duplicate image paths.")
    if bool((rows.groupby(["source", "unit_id"])["split"].nunique() != 1).any()):
        raise ValueError(f"fold_{fold_index} splits a source unit across roles.")
    for split in VALID_SPLITS:
        split_rows = rows[rows["split"] == split]
        for source in SOURCE_ORDER:
            labels = set(split_rows.loc[split_rows["source"] == source, "label"])
            if labels != set(PAD_UFES_NATIVE_LABELS):
                raise ValueError(f"fold_{fold_index} has incomplete {split}/{source} labels.")
    hiba = rows[rows["source"] == "hiba"]
    lesion_masses = hiba.groupby("unit_id")["view_mass"].sum().to_numpy()
    if not np.allclose(lesion_masses, 1.0):
        raise ValueError("Every HIBA lesion must have total effective view mass one.")


def _validate_rotating_coverage(folds: Sequence[pd.DataFrame], *, num_folds: int) -> None:
    if len(folds) != num_folds:
        raise ValueError(f"Expected {num_folds} multi-source folds, found {len(folds)}.")
    first_units = folds[0][["source", "image_path", "label", "unit_id"]].sort_values(
        ["source", "image_path"]
    )
    test_counts: dict[tuple[str, str], int] = {}
    validation_counts: dict[tuple[str, str], int] = {}
    for fold_index, rows in enumerate(folds):
        observed = rows[["source", "image_path", "label", "unit_id"]].sort_values(
            ["source", "image_path"]
        )
        if not first_units.reset_index(drop=True).equals(observed.reset_index(drop=True)):
            raise ValueError(f"fold_{fold_index} source/image/label/unit mapping drifted.")
        for row in rows.itertuples(index=False):
            key = (str(row.source), str(row.image_path))
            if row.split == "test":
                test_counts[key] = test_counts.get(key, 0) + 1
            elif row.split == "val":
                validation_counts[key] = validation_counts.get(key, 0) + 1
    expected = {(str(row.source), str(row.image_path)) for row in folds[0].itertuples(index=False)}
    if set(test_counts) != expected or any(count != 1 for count in test_counts.values()):
        raise ValueError("Every PAD/HIBA image must be outer-test data exactly once.")
    if set(validation_counts) != expected or any(
        count != 1 for count in validation_counts.values()
    ):
        raise ValueError("Every PAD/HIBA image must be validation data exactly once.")


def _manifest_fingerprint(folds: Sequence[pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for fold_index, rows in enumerate(folds):
        stable = rows[
            ["split", "source", "image_path", "label", "unit_id", "view_mass"]
        ].sort_values(["source", "image_path"])
        digest.update(f"fold={fold_index}\n".encode())
        digest.update(stable.to_csv(index=False, lineterminator="\n").encode())
    return digest.hexdigest()


def source_class_image_weights(rows: pd.DataFrame) -> torch.Tensor:
    if rows.empty:
        raise ValueError("Cannot weight an empty multi-source split.")
    expected_cells = {
        (source, label) for source in SOURCE_ORDER for label in PAD_UFES_NATIVE_LABELS
    }
    cells = list(zip(rows["source"].astype(str), rows["label"].astype(str), strict=True))
    if set(cells) != expected_cells:
        raise ValueError("Every split must contain all 12 source/class weighting cells.")
    view_masses = rows["view_mass"].astype(float).to_numpy()
    if not np.isfinite(view_masses).all() or (view_masses <= 0.0).any():
        raise ValueError("Every multi-source view mass must be finite and positive.")
    cell_masses = {
        cell: float(view_masses[np.asarray([row_cell == cell for row_cell in cells])].sum())
        for cell in expected_cells
    }
    if any(mass <= 0.0 for mass in cell_masses.values()):
        raise ValueError("Every source/class cell needs positive effective mass.")
    total_mass = float(view_masses.sum())
    cell_target = total_mass / len(expected_cells)
    weights = torch.tensor(
        [
            float(view_mass) * cell_target / cell_masses[cell]
            for view_mass, cell in zip(view_masses, cells, strict=True)
        ],
        dtype=torch.float32,
    )
    if not math.isclose(float(weights.sum()), total_mass, rel_tol=1e-5, abs_tol=1e-4):
        raise RuntimeError("Source/class weights do not preserve total effective mass.")
    for cell in expected_cells:
        mask = torch.tensor([row_cell == cell for row_cell in cells], dtype=torch.bool)
        if not math.isclose(float(weights[mask].sum()), cell_target, rel_tol=1e-5, abs_tol=1e-4):
            raise RuntimeError(f"Source/class cell {cell!r} does not have equal total weight.")
    return weights


class MultiSourceImageDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        split: str,
        *,
        use_training_transform: bool | None = None,
    ) -> None:
        self.rows = rows[rows["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Multi-source {split} split is empty.")
        self.label_to_index = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
        training_transform = split == "train"
        if use_training_transform is not None:
            training_transform = use_training_transform
        self.transform = get_pad_ufes_transforms(
            "train" if training_transform else "val",
            augmentation_profile=AUGMENTATION_PROFILE,
        )
        self.weights = source_class_image_weights(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float, int]:
        row = self.rows.iloc[index]
        image_path = resolve_project_path(Path(row["image_path"]))
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[str(row["label"])], float(self.weights[index]), index


def build_loader(
    dataset: MultiSourceImageDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _empty_confusion() -> list[list[int]]:
    size = len(PAD_UFES_NATIVE_LABELS)
    return [[0 for _ in range(size)] for _ in range(size)]


def _confusion_from_indices(truth: Sequence[int], predictions: Sequence[int]) -> list[list[int]]:
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction indices must align.")
    confusion = _empty_confusion()
    for actual, predicted in zip(truth, predictions, strict=True):
        if not 0 <= int(actual) < len(PAD_UFES_NATIVE_LABELS):
            raise ValueError("Truth index is outside the native label order.")
        if not 0 <= int(predicted) < len(PAD_UFES_NATIVE_LABELS):
            raise ValueError("Prediction index is outside the native label order.")
        confusion[int(actual)][int(predicted)] += 1
    return confusion


def _metrics_from_confusion(confusion: list[list[int]]) -> dict[str, object]:
    metrics = per_class_metrics(confusion, labels=PAD_UFES_NATIVE_LABELS)
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(PAD_UFES_NATIVE_LABELS)))
    precisions = [float(metrics[label]["precision"]) for label in PAD_UFES_NATIVE_LABELS]
    recalls = [float(metrics[label]["recall"]) for label in PAD_UFES_NATIVE_LABELS]
    f1_values = [float(metrics[label]["f1"]) for label in PAD_UFES_NATIVE_LABELS]
    return {
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": statistics.fmean(recalls),
        "macro_precision": statistics.fmean(precisions),
        "macro_recall": statistics.fmean(recalls),
        "macro_f1": statistics.fmean(f1_values),
        "per_class": metrics,
        "confusion_matrix": confusion,
        "total_support": total,
    }


def _add_confusions(confusions: Sequence[list[list[int]]]) -> list[list[int]]:
    pooled = _empty_confusion()
    for confusion in confusions:
        if len(confusion) != len(pooled) or any(len(row) != len(pooled) for row in confusion):
            raise ValueError("Confusion shape does not match the native label count.")
        for row_index in range(len(pooled)):
            for column_index in range(len(pooled)):
                pooled[row_index][column_index] += int(confusion[row_index][column_index])
    return pooled


def _group_probabilities(
    rows: pd.DataFrame,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_key: str,
) -> tuple[list[int], list[int]]:
    truths = []
    predictions = []
    for _, group in rows.groupby(group_key, sort=True):
        indices = torch.tensor(group.index.tolist(), dtype=torch.long)
        group_targets = targets.index_select(0, indices).unique()
        if len(group_targets) != 1:
            raise ValueError(f"A grouped {group_key} unit has multiple labels.")
        mean_probability = probabilities.index_select(0, indices).mean(dim=0)
        truths.append(int(group_targets.item()))
        predictions.append(int(mean_probability.argmax().item()))
    return truths, predictions


def metrics_from_probabilities(
    rows: pd.DataFrame,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    if len(rows) != len(probabilities) or len(rows) != len(targets):
        raise ValueError("Rows, probabilities, and targets must align.")
    if probabilities.shape != (len(rows), len(PAD_UFES_NATIVE_LABELS)):
        raise ValueError("Probability shape does not match rows and native labels.")
    predictions = probabilities.argmax(dim=1)
    by_source = {}
    for source in SOURCE_ORDER:
        indices = rows.index[rows["source"] == source].tolist()
        if not indices:
            raise ValueError(f"Metrics are missing source {source!r}.")
        index_tensor = torch.tensor(indices, dtype=torch.long)
        by_source[source] = _metrics_from_confusion(
            _confusion_from_indices(
                targets.index_select(0, index_tensor).tolist(),
                predictions.index_select(0, index_tensor).tolist(),
            )
        )

    hiba_rows = rows[rows["source"] == "hiba"].copy()
    hiba_indices = torch.tensor(hiba_rows.index.tolist(), dtype=torch.long)
    hiba_probabilities = probabilities.index_select(0, hiba_indices)
    hiba_targets = targets.index_select(0, hiba_indices)
    hiba_rows = hiba_rows.reset_index(drop=True)
    hiba_truth, hiba_predictions = _group_probabilities(
        hiba_rows,
        hiba_probabilities,
        hiba_targets,
        group_key="unit_id",
    )
    hiba_lesion = _metrics_from_confusion(_confusion_from_indices(hiba_truth, hiba_predictions))
    source_mean = statistics.fmean(
        [float(by_source["pad_ufes"]["macro_f1"]), float(hiba_lesion["macro_f1"])]
    )
    worst_source = min(float(by_source["pad_ufes"]["macro_f1"]), float(hiba_lesion["macro_f1"]))
    return {
        "by_source": by_source,
        "hiba_lesion": hiba_lesion,
        "primary_source_mean_macro_f1": source_mean,
        "primary_worst_source_macro_f1": worst_source,
        "primary_units": {"pad_ufes": "image", "hiba": "lesion"},
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    sample_count = len(loader.dataset)
    probabilities = torch.empty((sample_count, len(PAD_UFES_NATIVE_LABELS)), dtype=torch.float32)
    targets_out = torch.empty(sample_count, dtype=torch.long)
    seen = torch.zeros(sample_count, dtype=torch.bool)
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    for images, targets, weights, indices in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)
        weights = weights.to(device=device, dtype=torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            losses = criterion(logits, targets)
            batch_weight = weights.sum()
            loss = (losses * weights).sum() / batch_weight
            if training:
                loss.backward()
                optimizer.step()
        index_tensor = indices.to(dtype=torch.long)
        probabilities[index_tensor] = torch.softmax(logits.detach(), dim=1).cpu()
        targets_out[index_tensor] = targets.detach().cpu()
        seen[index_tensor] = True
        weighted_loss_sum += float((losses.detach() * weights).sum().item())
        weight_sum += float(batch_weight.item())
    if not bool(seen.all()) or weight_sum <= 0.0:
        raise RuntimeError("Epoch did not produce every multi-source prediction.")
    rows = loader.dataset.rows.reset_index(drop=True)
    return {
        "loss": weighted_loss_sum / weight_sum,
        **metrics_from_probabilities(rows, probabilities, targets_out),
    }


def _compact_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    by_source = metrics["by_source"]
    return {
        "loss": float(metrics["loss"]),
        "primary_source_mean_macro_f1": float(metrics["primary_source_mean_macro_f1"]),
        "primary_worst_source_macro_f1": float(metrics["primary_worst_source_macro_f1"]),
        "pad_image_macro_f1": float(by_source["pad_ufes"]["macro_f1"]),
        "hiba_image_macro_f1": float(by_source["hiba"]["macro_f1"]),
        "hiba_lesion_macro_f1": float(metrics["hiba_lesion"]["macro_f1"]),
    }


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_fold(
    rows: pd.DataFrame,
    *,
    fold_index: int,
    pad_summary: dict[str, object],
    hiba_summary: dict[str, object],
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
    set_seed(seed)
    datasets = {split: MultiSourceImageDataset(rows, split) for split in ("train", "val", "test")}
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
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    parameter_count = _parameter_count(model)
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "schedule": "none",
        "augmentation_profile": AUGMENTATION_PROFILE,
        "label_smoothing": 0.0,
        "sampling": "random_shuffle_without_replacement",
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
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
                    "dataset_role": "multisource_development",
                    "sources": list(SOURCE_ORDER),
                    "labels": list(PAD_UFES_NATIVE_LABELS),
                    "pretrained_weights": PRETRAINED_WEIGHTS if pretrained else "none",
                    "pretrained_weights_id": (
                        pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS)
                        if pretrained
                        else None
                    ),
                    "preprocessing": PREPROCESSING,
                    "augmentation_profile": AUGMENTATION_PROFILE,
                    "selection_metric": SELECTION_METRIC,
                    "source_class_weighting": SOURCE_CLASS_WEIGHTING,
                    "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
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
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    train_evaluation_dataset = MultiSourceImageDataset(
        rows,
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
    test_metrics = run_epoch(model, loaders["test"], criterion, device)
    report = {
        "context": (
            "Experimental PAD-UFES plus HIBA source-balanced development classification; "
            "not external validation, medical certainty, or production evidence."
        ),
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
        "pad_protocol": PAD_PROTOCOL,
        "hiba_protocol": HIBA_PROTOCOL,
        "hiba_role": "multisource_development",
        "manifest_fingerprint": manifest_fingerprint,
        "selection_metric": SELECTION_METRIC,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "primary_units": {"pad_ufes": "image", "hiba": "lesion"},
        "hyperparameters": hyperparameters,
        "inference_parameter_count": parameter_count,
        "source_total_raw_image_counts": source_total_raw_image_counts,
        "source_total_effective_unit_counts": source_total_effective_unit_counts,
        "split_raw_image_counts": {
            split: {
                source: int((datasets[split].rows["source"] == source).sum())
                for source in SOURCE_ORDER
            }
            for split in VALID_SPLITS
        },
        "split_effective_unit_counts": {
            split: {
                source: float(
                    datasets[split]
                    .rows.loc[datasets[split].rows["source"] == source, "view_mass"]
                    .sum()
                )
                for source in SOURCE_ORDER
            }
            for split in VALID_SPLITS
        },
        "pad_split_summary": pad_summary,
        "hiba_split_summary": hiba_summary,
        "best_epoch": best_epoch,
        "best_val_primary_source_mean_macro_f1": best_source_mean,
        "history": history,
        "selected_train": selected_train,
        "selected_val": selected_val,
        "test": test_metrics,
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
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def validate_report(
    report: Mapping[str, object],
    *,
    fold_index: int,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> None:
    expected = {
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
        "pad_protocol": PAD_PROTOCOL,
        "hiba_protocol": HIBA_PROTOCOL,
        "hiba_role": "multisource_development",
        "selection_metric": SELECTION_METRIC,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "primary_units": {"pad_ufes": "image", "hiba": "lesion"},
    }
    mismatches = [
        f"{key}={report.get(key)!r}" for key, value in expected.items() if report.get(key) != value
    ]
    hyperparameters = report.get("hyperparameters")
    expected_hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "schedule": "none",
        "augmentation_profile": AUGMENTATION_PROFILE,
        "label_smoothing": 0.0,
        "sampling": "random_shuffle_without_replacement",
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
    }
    if not isinstance(hyperparameters, dict):
        mismatches.append("hyperparameters are missing")
    else:
        mismatches.extend(
            f"hyperparameters.{key}={hyperparameters.get(key)!r}"
            for key, value in expected_hyperparameters.items()
            if hyperparameters.get(key) != value
        )
    for name, protocol in (
        ("pad_split_summary", PAD_PROTOCOL),
        ("hiba_split_summary", HIBA_PROTOCOL),
    ):
        summary = report.get(name)
        if not isinstance(summary, dict):
            mismatches.append(f"{name} is missing")
        else:
            summary_expected = {
                "protocol": protocol,
                "num_folds": num_folds,
                "fold_index": fold_index,
                "test_outer_fold": fold_index,
                "validation_outer_fold": (fold_index + 1) % num_folds,
            }
            mismatches.extend(
                f"{name}.{key}={summary.get(key)!r}"
                for key, value in summary_expected.items()
                if summary.get(key) != value
            )
    fingerprint = report.get("manifest_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        mismatches.append(f"manifest_fingerprint={fingerprint!r}")
    if not isinstance(report.get("inference_parameter_count"), int):
        mismatches.append("inference_parameter_count is invalid")
    if (
        not isinstance(report.get("checkpoint_sha256"), str)
        or len(str(report.get("checkpoint_sha256"))) != 64
    ):
        mismatches.append("checkpoint_sha256 is invalid")
    for split_name in ("selected_train", "selected_val", "test"):
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
            f"PAD/HIBA report fold_{fold_index} violates the locked protocol: "
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


def _pooled_report_confusion(
    reports: Sequence[dict[str, object]],
    *keys: str,
) -> list[list[int]]:
    confusions = []
    for report in reports:
        value: object = report["test"]
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"Report test metrics are missing {'.'.join(keys)}.")
            value = value[key]
        if not isinstance(value, dict) or "confusion_matrix" not in value:
            raise ValueError(f"Report test metrics are missing {'.'.join(keys)} confusion.")
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
            raise FileNotFoundError(f"Missing PAD/HIBA report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_report(report, fold_index=fold_index, num_folds=num_folds, seed=seed)
        reports.append(report)
    consistency_keys = (
        "manifest_fingerprint",
        "pretrained_weights_id",
        "inference_parameter_count",
        "source_total_raw_image_counts",
        "source_total_effective_unit_counts",
    )
    disagreements = [
        key
        for key in consistency_keys
        if len({json.dumps(report.get(key), sort_keys=True) for report in reports}) != 1
    ]
    if disagreements:
        raise ValueError("PAD/HIBA reports disagree on provenance: " + ", ".join(disagreements))

    pooled_pad = _metrics_from_confusion(_pooled_report_confusion(reports, "by_source", "pad_ufes"))
    pooled_hiba_image = _metrics_from_confusion(
        _pooled_report_confusion(reports, "by_source", "hiba")
    )
    pooled_hiba_lesion = _metrics_from_confusion(_pooled_report_confusion(reports, "hiba_lesion"))
    observed_support = {
        "pad_image": int(pooled_pad["total_support"]),
        "hiba_image": int(pooled_hiba_image["total_support"]),
        "hiba_lesion": int(pooled_hiba_lesion["total_support"]),
    }
    expected_support = {"pad_image": 2_298, "hiba_image": 309, "hiba_lesion": 308}
    if observed_support != expected_support:
        raise ValueError(
            f"Outer-test coverage differs: observed={observed_support}, expected={expected_support}"
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
    parameter_count = int(reports[0]["inference_parameter_count"])
    rules = {
        "mean_selected_train_val_primary_source_mean_macro_f1_gap_lte_0_2000": (
            statistics.fmean(gaps) <= 0.2
        ),
        "pooled_pad_image_macro_f1_gte_0_6200": float(pooled_pad["macro_f1"]) >= 0.62,
        "pooled_hiba_lesion_macro_f1_gte_0_5000": (float(pooled_hiba_lesion["macro_f1"]) >= 0.5),
        "pooled_primary_source_mean_macro_f1_gte_0_5600": source_mean >= 0.56,
        "pooled_primary_worst_source_macro_f1_gte_0_5000": worst_source >= 0.5,
        "pooled_pad_melanoma_f1_gte_0_5000": (
            float(pooled_pad["per_class"]["melanoma"]["f1"]) >= 0.5
        ),
        "pooled_pad_scc_f1_gte_0_2000": (
            float(pooled_pad["per_class"]["squamous_cell_carcinoma"]["f1"]) >= 0.2
        ),
        "pooled_hiba_melanoma_f1_gte_0_3500": (
            float(pooled_hiba_lesion["per_class"]["melanoma"]["f1"]) >= 0.35
        ),
        "pooled_hiba_scc_f1_gte_0_3000": (
            float(pooled_hiba_lesion["per_class"]["squamous_cell_carcinoma"]["f1"]) >= 0.3
        ),
        "inference_parameter_count_lte_30000000": parameter_count <= 30_000_000,
    }
    summary = {
        "context": (
            "Experimental PAD-UFES plus HIBA source-balanced development evidence only; "
            "not independent validation, medical certainty, or production evidence."
        ),
        "pad_protocol": PAD_PROTOCOL,
        "hiba_protocol": HIBA_PROTOCOL,
        "hiba_role": "multisource_development",
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
        "inference_parameter_count": parameter_count,
        "source_total_raw_image_counts": reports[0]["source_total_raw_image_counts"],
        "source_total_effective_unit_counts": reports[0]["source_total_effective_unit_counts"],
        "checkpoint_bytes": _distribution(
            [float(report["checkpoint_bytes"]) for report in reports]
        ),
        "selected_train_val_primary_source_mean_macro_f1_gap": _distribution(gaps),
        "pooled_primary_by_source": {
            "pad_ufes_image": pooled_pad,
            "hiba_lesion": pooled_hiba_lesion,
        },
        "pooled_hiba_image_secondary": pooled_hiba_image,
        "pooled_primary_source_mean_macro_f1": source_mean,
        "pooled_primary_worst_source_macro_f1": worst_source,
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
                "test_primary_source_mean_macro_f1": float(
                    report["test"]["primary_source_mean_macro_f1"]
                ),
                "test_pad_image_macro_f1": float(
                    report["test"]["by_source"]["pad_ufes"]["macro_f1"]
                ),
                "test_hiba_lesion_macro_f1": float(report["test"]["hiba_lesion"]["macro_f1"]),
            }
            for fold_index, report in enumerate(reports)
        ],
        "privacy": {
            "aggregate_metrics_only": True,
            "per_image_predictions_written": False,
            "identifiers_or_paths_written": False,
        },
        "caveat": (
            "Both PAD-UFES and HIBA participated in development. Passing only nominates this "
            "configuration for a separately preregistered untouched MILK10k assessment; it cannot "
            "establish robustness, fairness, patient-self-photo behavior, deployment, diagnosis, "
            "or medical readiness."
        ),
    }
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
    print(f"Validated {DEFAULT_FOLDS} PAD/HIBA rotating folds; fingerprint={manifests.fingerprint}")
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
            print(f"fold_{fold_index}: report exists; skipping")
            continue
        print(f"fold_{fold_index}: starting source-balanced ConvNeXt-Tiny")
        report = train_fold(
            manifests.folds[fold_index],
            fold_index=fold_index,
            pad_summary=manifests.pad_fold_summaries[fold_index],
            hiba_summary=manifests.hiba_fold_summaries[fold_index],
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
            "outer-test report held for the pooled five-fold summary"
        )
    summary = summarize_reports(runs_root, runs_root / "summary.json")
    print(
        "PAD/HIBA development complete: "
        f"source_mean={summary['pooled_primary_source_mean_macro_f1']:.4f} "
        f"worst_source={summary['pooled_primary_worst_source_macro_f1']:.4f} "
        f"all_rules_pass={summary['decision_rules']['all_pass']} "
        f"summary={project_relative(runs_root / 'summary.json')}"
    )
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

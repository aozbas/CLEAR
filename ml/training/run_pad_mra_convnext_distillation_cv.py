"""Distill frozen MedSigLIP image features into a PAD/MRA ConvNeXt-Tiny student.

This is a grouped multi-source development workflow for experimental classification only. PAD-UFES
is known MedSigLIP pretraining data and MRA-MIDAS is training data here, so the output is not an
independent validation, deployment result, diagnosis, or medical-readiness claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from ml.evaluation.metrics import per_class_metrics
from ml.evaluation.mra_midas import CLINICAL_DISTANCES
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.preprocessing import get_pad_ufes_transforms
from ml.training.prepare_mra_midas_cv import PROTOCOL as MRA_PROTOCOL
from ml.training.prepare_pad_ufes import SPLIT_ORDER, project_relative
from ml.training.prepare_pad_ufes_cv import PROTOCOL as PAD_PROTOCOL
from ml.training.run_pad_mra_medsiglip_probe_cv import (
    MRA_AGGREGATION,
    SOURCE_ORDER,
    MultiSourceManifests,
    load_multi_source_manifests,
    load_or_extract_image_embeddings,
)
from ml.training.run_pad_ufes_medsiglip_probe_cv import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    KNOWN_PRETRAINING_DATASETS,
)
from ml.training.run_pad_ufes_medsiglip_probe_cv import (
    PREPROCESSING as TEACHER_PREPROCESSING,
)
from ml.training.train import build_loader, get_device, resolve_project_path, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAD_SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes_native_cv"
DEFAULT_MRA_SPLITS_DIR = (
    PROJECT_ROOT / "ml" / "data" / "external_splits" / "mra_midas_multisource_cv"
)
DEFAULT_TEACHER_EMBEDDINGS_PATH = (
    PROJECT_ROOT / "ml" / "runs" / "embeddings" / "pad_mra_medsiglip_448_rev9cea28a.pt"
)
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "runs"
    / "training"
    / "pad_mra_convnext_tiny_medsiglip_distillation-cv-seed42"
)
DEFAULT_CHECKPOINTS_DIR = (
    PROJECT_ROOT / "ml" / "models" / "pad_mra_convnext_tiny_medsiglip_distillation_cv_seed42"
)
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "ml" / "model_cache" / "huggingface"
DEFAULT_TORCH_CACHE_DIR = PROJECT_ROOT / "ml" / "model_cache" / "torch"

DEFAULT_FOLDS = 5
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DISTILLATION_WEIGHT = 0.25
DEFAULT_SEED = 42

ARCHITECTURE = "convnext_tiny_medsiglip_feature_distillation"
STUDENT_ARCHITECTURE = "convnext_tiny"
STUDENT_WEIGHTS_ID = models.ConvNeXt_Tiny_Weights.DEFAULT.name
STUDENT_PREPROCESSING = "imagenet_224_pad_ufes_baseline"
STUDENT_IMAGE_LOADING = "shared_in_memory_normalized_224_tensor_cache"
STUDENT_FEATURE_DIM = 768
VIEW_WEIGHTING = "pad_image_one_mra_equal_distance_equal_lesion"
SOURCE_CLASS_WEIGHTING = "equal_total_effective_weight_per_source_class_cell"
SELECTION_METRIC = "val_pad_mra_raw_image_source_mean_macro_f1"
PRIMARY_MRA_VIEW = "raw_image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked PAD/MRA ConvNeXt-Tiny MedSigLIP distillation CV experiment."
    )
    parser.add_argument("--pad-splits-dir", type=Path, default=DEFAULT_PAD_SPLITS_DIR)
    parser.add_argument("--mra-splits-dir", type=Path, default=DEFAULT_MRA_SPLITS_DIR)
    parser.add_argument(
        "--teacher-embeddings",
        type=Path,
        default=DEFAULT_TEACHER_EMBEDDINGS_PATH,
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--torch-cache-dir", type=Path, default=DEFAULT_TORCH_CACHE_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _pad_unit_id(image_path: str) -> str:
    return "pad-" + hashlib.sha256(str(image_path).encode()).hexdigest()


def fold_image_rows(
    manifests: MultiSourceManifests,
    *,
    fold_index: int,
) -> pd.DataFrame:
    if not 0 <= fold_index < len(manifests.pad.fold_rows):
        raise ValueError(f"Unknown fold index: {fold_index}")
    pad = manifests.pad.fold_rows[fold_index][["split", "image_path", "label"]].copy()
    pad.insert(1, "source", "pad_ufes")
    pad["unit_id"] = pad["image_path"].astype(str).map(_pad_unit_id)
    pad["record_id"] = pad["unit_id"]
    pad["distance"] = "single"
    mra = manifests.mra_fold_rows[fold_index][
        [
            "split",
            "source",
            "unit_id",
            "record_id",
            "distance",
            "image_path",
            "label",
        ]
    ].copy()
    rows = pd.concat(
        [
            pad[
                [
                    "split",
                    "source",
                    "unit_id",
                    "record_id",
                    "distance",
                    "image_path",
                    "label",
                ]
            ],
            mra,
        ],
        ignore_index=True,
    )
    if bool(rows["image_path"].duplicated().any()):
        raise ValueError(f"fold_{fold_index} contains duplicate image paths.")
    if set(rows["source"]) != set(SOURCE_ORDER):
        raise ValueError(f"fold_{fold_index} does not contain both locked sources.")
    if bool((rows.groupby(["source", "record_id"])["split"].nunique() != 1).any()):
        raise ValueError(f"fold_{fold_index} splits a patient or record across roles.")
    rows = add_view_masses(rows)
    for split in SPLIT_ORDER:
        split_rows = rows.loc[rows["split"] == split]
        for source in SOURCE_ORDER:
            labels = set(split_rows.loc[split_rows["source"] == source, "label"])
            if labels != set(PAD_UFES_NATIVE_LABELS):
                raise ValueError(f"fold_{fold_index} has incomplete {split}/{source} labels.")
    return rows.sort_values(
        ["split", "source", "label", "unit_id", "distance", "image_path"]
    ).reset_index(drop=True)


def add_view_masses(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "unit_id", "distance", "image_path", "label"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Image rows are missing columns: {', '.join(sorted(missing))}")
    result = rows.copy()
    result["view_mass"] = 1.0
    mra = result.loc[result["source"] == "mra_midas"]
    for unit_id, unit_rows in mra.groupby("unit_id", sort=False):
        if set(unit_rows["distance"]) != set(CLINICAL_DISTANCES):
            raise ValueError(f"MRA unit {unit_id!r} is missing a locked distance.")
        if unit_rows["label"].nunique() != 1:
            raise ValueError(f"MRA unit {unit_id!r} has multiple labels.")
        for distance in CLINICAL_DISTANCES:
            indices = unit_rows.index[unit_rows["distance"] == distance]
            if len(indices) == 0:
                raise ValueError(f"MRA unit {unit_id!r} is missing distance {distance!r}.")
            result.loc[indices, "view_mass"] = 0.5 / len(indices)
        unit_mass = float(result.loc[unit_rows.index, "view_mass"].sum())
        if not math.isclose(unit_mass, 1.0, abs_tol=1e-12):
            raise RuntimeError(f"MRA unit {unit_id!r} view masses do not sum to one.")
    if bool((result["view_mass"] <= 0.0).any()):
        raise ValueError("Every image view mass must be positive.")
    return result


def source_class_image_weights(rows: pd.DataFrame) -> torch.Tensor:
    if rows.empty:
        raise ValueError("Cannot weight an empty image split.")
    if "view_mass" not in rows.columns:
        raise ValueError("Image rows must be assigned view masses before source/class weighting.")
    expected_cells = {
        (source, label) for source in SOURCE_ORDER for label in PAD_UFES_NATIVE_LABELS
    }
    cells = list(zip(rows["source"].astype(str), rows["label"].astype(str), strict=True))
    if set(cells) != expected_cells:
        raise ValueError(
            "Source-class weighting cells differ: "
            f"missing={sorted(expected_cells - set(cells))}, "
            f"unknown={sorted(set(cells) - expected_cells)}"
        )
    masses = rows["view_mass"].astype(float)
    if bool((masses <= 0.0).any()):
        raise ValueError("View masses must be positive.")
    cell_masses = {
        cell: float(masses.loc[[row_cell == cell for row_cell in cells]].sum())
        for cell in expected_cells
    }
    if any(mass <= 0.0 for mass in cell_masses.values()):
        raise ValueError("Every source/class cell needs positive effective mass.")
    total_mass = float(masses.sum())
    cell_target = total_mass / len(expected_cells)
    weights = torch.tensor(
        [
            float(row.view_mass) * cell_target / cell_masses[(str(row.source), str(row.label))]
            for row in rows.itertuples(index=False)
        ],
        dtype=torch.float32,
    )
    if not math.isclose(float(weights.sum()), total_mass, rel_tol=1e-5, abs_tol=1e-4):
        raise RuntimeError("Source-class image weights do not preserve total effective mass.")
    for cell in expected_cells:
        cell_total = float(
            weights[torch.tensor([row_cell == cell for row_cell in cells], dtype=torch.bool)].sum()
        )
        if not math.isclose(cell_total, cell_target, rel_tol=1e-5, abs_tol=1e-4):
            raise RuntimeError(f"Source-class cell {cell!r} does not have equal total weight.")
    return weights


def _validate_teacher_cache_for_rows(
    rows: pd.DataFrame,
    teacher_cache: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    paths = teacher_cache.get("image_paths")
    sources = teacher_cache.get("image_sources")
    labels = teacher_cache.get("image_labels")
    features = teacher_cache.get("features")
    if not isinstance(paths, list) or not isinstance(sources, list) or not isinstance(labels, list):
        raise ValueError("Teacher cache is missing ordered image provenance.")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("Teacher cache is missing a two-dimensional feature tensor.")
    if not (len(paths) == len(sources) == len(labels) == features.shape[0]):
        raise ValueError("Teacher cache provenance lengths differ from its feature tensor.")
    if len(set(map(str, paths))) != len(paths):
        raise ValueError("Teacher cache contains duplicate image paths.")
    index_by_path = {str(path): index for index, path in enumerate(paths)}
    indices: list[int] = []
    for row in rows.itertuples(index=False):
        path = str(row.image_path)
        if path not in index_by_path:
            raise ValueError("Teacher cache is missing a requested image.")
        index = index_by_path[path]
        if str(sources[index]) != str(row.source) or str(labels[index]) != str(row.label):
            raise ValueError("Teacher cache source or label differs from an image row.")
        indices.append(index)
    selected = features.index_select(0, torch.tensor(indices, dtype=torch.long)).cpu().float()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Teacher cache contains non-finite features.")
    norms = selected.norm(dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)):
        raise ValueError("Teacher target features must be L2 normalized.")
    return selected, torch.tensor(indices, dtype=torch.long)


def preload_student_image_tensors(rows: pd.DataFrame) -> dict[str, torch.Tensor]:
    if "image_path" not in rows.columns:
        raise ValueError("Student image rows are missing image_path.")
    image_paths = sorted(set(rows["image_path"].astype(str)))
    if not image_paths:
        raise ValueError("Student image tensor cache cannot be empty.")
    transform = get_pad_ufes_transforms("val", augmentation_profile="baseline")
    cache: dict[str, torch.Tensor] = {}
    for image_path in image_paths:
        path = resolve_project_path(Path(image_path))
        with Image.open(path) as source_image:
            image = transform(source_image.convert("RGB")).contiguous()
        if image.shape != (3, 224, 224):
            raise ValueError(f"Unexpected cached student image shape: {tuple(image.shape)}")
        if image.dtype != torch.float32 or not bool(torch.isfinite(image).all()):
            raise ValueError("Cached student images must be finite float32 tensors.")
        cache[image_path] = image
    return cache


def cached_student_tensor(image: torch.Tensor, *, training: bool) -> torch.Tensor:
    if image.shape != (3, 224, 224) or image.dtype != torch.float32:
        raise ValueError("Cached student image must be a float32 tensor with shape (3, 224, 224).")
    if training and bool(torch.rand(1) < 0.5):
        return image.flip(-1)
    return image


class DistillationImageDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        teacher_cache: dict[str, object],
        student_image_cache: dict[str, torch.Tensor],
        *,
        training: bool,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.teacher_features, self.teacher_indices = _validate_teacher_cache_for_rows(
            self.rows,
            teacher_cache,
        )
        label_to_index = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
        source_to_index = {source: index for index, source in enumerate(SOURCE_ORDER)}
        self.targets = torch.tensor(
            [label_to_index[str(label)] for label in self.rows["label"]],
            dtype=torch.long,
        )
        self.sources = torch.tensor(
            [source_to_index[str(source)] for source in self.rows["source"]],
            dtype=torch.long,
        )
        self.weights = source_class_image_weights(self.rows)
        self.student_image_cache = student_image_cache
        self.training = training
        missing_images = sorted(
            set(self.rows["image_path"].astype(str)) - set(self.student_image_cache)
        )
        if missing_images:
            raise ValueError("Student image cache is missing a requested image.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, float, int, int]:
        image_path = str(self.rows.iloc[index]["image_path"])
        image = cached_student_tensor(
            self.student_image_cache[image_path],
            training=self.training,
        )
        return (
            image,
            self.teacher_features[index],
            int(self.targets[index]),
            float(self.weights[index]),
            int(self.sources[index]),
            index,
        )


class DistilledConvNeXtTiny(nn.Module):
    def __init__(
        self,
        teacher_feature_dim: int,
        *,
        weights: object | None,
    ) -> None:
        super().__init__()
        if teacher_feature_dim <= 0:
            raise ValueError("teacher_feature_dim must be positive.")
        self.student = models.convnext_tiny(weights=weights)
        classifier = self.student.classifier
        if len(classifier) != 3 or not isinstance(classifier[-1], nn.Linear):
            raise RuntimeError("Unexpected torchvision ConvNeXt-Tiny classifier structure.")
        feature_dim = int(classifier[-1].in_features)
        if feature_dim != STUDENT_FEATURE_DIM:
            raise RuntimeError(f"Unexpected ConvNeXt-Tiny feature dimension: {feature_dim}")
        classifier[-1] = nn.Linear(feature_dim, len(PAD_UFES_NATIVE_LABELS))
        self.projection = nn.Linear(feature_dim, teacher_feature_dim)
        self.student_feature_dim = feature_dim
        self.teacher_feature_dim = teacher_feature_dim

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.student.features(images)
        features = self.student.avgpool(features)
        features = self.student.classifier[0](features)
        features = self.student.classifier[1](features)
        logits = self.student.classifier[2](features)
        projection = functional.normalize(self.projection(features), p=2, dim=-1)
        return logits, projection

    def inference_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.student.parameters())

    def projection_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.projection.parameters())

    def training_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_student_model(
    teacher_feature_dim: int,
    *,
    pretrained: bool = True,
) -> DistilledConvNeXtTiny:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    return DistilledConvNeXtTiny(teacher_feature_dim, weights=weights)


def _empty_confusion() -> list[list[int]]:
    size = len(PAD_UFES_NATIVE_LABELS)
    return [[0 for _ in range(size)] for _ in range(size)]


def _confusion_from_indices(
    truths: Sequence[int],
    predictions: Sequence[int],
) -> list[list[int]]:
    if len(truths) != len(predictions):
        raise ValueError("Truth and prediction lengths differ.")
    confusion = _empty_confusion()
    for truth, prediction in zip(truths, predictions, strict=True):
        if not 0 <= int(truth) < len(confusion) or not 0 <= int(prediction) < len(confusion):
            raise ValueError("A class index is outside the locked label space.")
        confusion[int(truth)][int(prediction)] += 1
    return confusion


def _metrics_from_confusion(confusion: list[list[int]]) -> dict[str, object]:
    size = len(PAD_UFES_NATIVE_LABELS)
    if len(confusion) != size or any(len(row) != size for row in confusion):
        raise ValueError("A confusion matrix has an invalid shape.")
    per_class = per_class_metrics(confusion, labels=PAD_UFES_NATIVE_LABELS)
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(size))
    precisions = [float(per_class[label]["precision"]) for label in PAD_UFES_NATIVE_LABELS]
    recalls = [float(per_class[label]["recall"]) for label in PAD_UFES_NATIVE_LABELS]
    f1_scores = [float(per_class[label]["f1"]) for label in PAD_UFES_NATIVE_LABELS]
    return {
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": statistics.fmean(recalls),
        "macro_precision": statistics.fmean(precisions),
        "macro_recall": statistics.fmean(recalls),
        "macro_f1": statistics.fmean(f1_scores),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "total_support": total,
    }


def _add_confusions(confusions: Sequence[list[list[int]]]) -> list[list[int]]:
    pooled = _empty_confusion()
    for confusion in confusions:
        if len(confusion) != len(pooled) or any(len(row) != len(pooled) for row in confusion):
            raise ValueError("A confusion matrix has an invalid shape.")
        for row_index in range(len(pooled)):
            for column_index in range(len(pooled)):
                pooled[row_index][column_index] += int(confusion[row_index][column_index])
    return pooled


def _group_logits(
    rows: pd.DataFrame,
    logits: torch.Tensor,
    group_columns: Sequence[str],
) -> tuple[list[int], list[int], pd.DataFrame, torch.Tensor]:
    if len(rows) != logits.shape[0]:
        raise ValueError("Rows and logits have different lengths.")
    label_to_index = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
    grouped_rows: list[dict[str, str]] = []
    grouped_logits: list[torch.Tensor] = []
    truths: list[int] = []
    predictions: list[int] = []
    for keys, group in rows.groupby(list(group_columns), sort=True):
        labels = set(group["label"].astype(str))
        if len(labels) != 1:
            raise ValueError("A metric aggregation group has multiple labels.")
        indices = group.index.tolist()
        mean_logits = logits.index_select(0, torch.tensor(indices, dtype=torch.long)).mean(dim=0)
        label = labels.pop()
        key_values = keys if isinstance(keys, tuple) else (keys,)
        grouped_rows.append(
            {
                **dict(zip(group_columns, map(str, key_values), strict=True)),
                "label": label,
            }
        )
        grouped_logits.append(mean_logits)
        truths.append(label_to_index[label])
        predictions.append(int(mean_logits.argmax().item()))
    if not grouped_logits:
        raise ValueError("Metric aggregation produced no groups.")
    return truths, predictions, pd.DataFrame(grouped_rows), torch.stack(grouped_logits)


def metrics_from_image_logits(
    rows: pd.DataFrame,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    if logits.ndim != 2 or logits.shape[1] != len(PAD_UFES_NATIVE_LABELS):
        raise ValueError("Image logits have an invalid shape.")
    if len(rows) != logits.shape[0] or targets.shape != (len(rows),):
        raise ValueError("Image rows, logits, and targets are not aligned.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Image logits contain non-finite values.")
    predictions = logits.argmax(dim=1)
    by_source: dict[str, dict[str, object]] = {}
    for source in SOURCE_ORDER:
        indices = rows.index[rows["source"] == source].tolist()
        if not indices:
            raise ValueError(f"Metrics are missing source {source!r}.")
        source_truths = targets.index_select(0, torch.tensor(indices, dtype=torch.long)).tolist()
        source_predictions = predictions.index_select(
            0, torch.tensor(indices, dtype=torch.long)
        ).tolist()
        by_source[source] = _metrics_from_confusion(
            _confusion_from_indices(source_truths, source_predictions)
        )
    primary_source_mean = statistics.fmean(
        float(by_source[source]["macro_f1"]) for source in SOURCE_ORDER
    )
    primary_worst_source = min(float(by_source[source]["macro_f1"]) for source in SOURCE_ORDER)

    mra_rows = rows.loc[rows["source"] == "mra_midas"].copy()
    mra_indices = mra_rows.index.tolist()
    mra_rows = mra_rows.reset_index(drop=True)
    mra_logits = logits.index_select(0, torch.tensor(mra_indices, dtype=torch.long))
    distance_truths, distance_predictions, distance_rows, distance_logits = _group_logits(
        mra_rows,
        mra_logits,
        ("unit_id", "distance"),
    )
    if set(distance_rows["distance"]) != set(CLINICAL_DISTANCES):
        raise ValueError("Distance-unit metrics do not contain both locked distances.")
    distance_metrics = _metrics_from_confusion(
        _confusion_from_indices(distance_truths, distance_predictions)
    )
    paired_truths, paired_predictions, _paired_rows, _paired_logits = _group_logits(
        distance_rows,
        distance_logits,
        ("unit_id",),
    )
    paired_metrics = _metrics_from_confusion(
        _confusion_from_indices(paired_truths, paired_predictions)
    )
    combined = _metrics_from_confusion(
        _add_confusions([by_source[source]["confusion_matrix"] for source in SOURCE_ORDER])
    )
    return {
        "by_source": by_source,
        "source_mean_macro_f1": primary_source_mean,
        "worst_source_macro_f1": primary_worst_source,
        "combined_primary_secondary": combined,
        "mra_distance_unit": distance_metrics,
        "mra_paired_lesion": paired_metrics,
    }


def run_student_epoch(
    model: DistilledConvNeXtTiny,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    distillation_weight: float = DEFAULT_DISTILLATION_WEIGHT,
) -> dict[str, object]:
    if distillation_weight < 0.0:
        raise ValueError("distillation_weight must be non-negative.")
    if not isinstance(loader.dataset, DistillationImageDataset):
        raise TypeError("Student loader must use DistillationImageDataset.")
    training = optimizer is not None
    model.train(training)
    dataset = loader.dataset
    all_logits = torch.empty(
        (len(dataset), len(PAD_UFES_NATIVE_LABELS)),
        dtype=torch.float32,
    )
    seen = torch.zeros(len(dataset), dtype=torch.bool)
    weighted_supervised_sum = 0.0
    weighted_distillation_sum = 0.0
    weight_sum = 0.0

    with torch.set_grad_enabled(training):
        for images, teacher_features, targets, weights, _sources, indices in loader:
            images = images.to(device)
            teacher_features = teacher_features.to(device)
            targets = targets.to(device)
            weights = weights.to(device=device, dtype=torch.float32)
            if not bool((weights > 0.0).all()):
                raise ValueError("Every batch loss weight must be positive.")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, student_projection = model(images)
            supervised_losses = functional.cross_entropy(logits, targets, reduction="none")
            distillation_losses = 1.0 - functional.cosine_similarity(
                student_projection,
                teacher_features,
                dim=1,
            )
            batch_weight_sum = weights.sum()
            loss = (
                (supervised_losses + distillation_weight * distillation_losses) * weights
            ).sum() / batch_weight_sum
            if training:
                loss.backward()
                optimizer.step()
            batch_indices = indices.long()
            if bool(seen.index_select(0, batch_indices).any()):
                raise RuntimeError("Student epoch visited an image more than once.")
            seen[batch_indices] = True
            all_logits.index_copy_(0, batch_indices, logits.detach().cpu().float())
            weighted_supervised_sum += float((supervised_losses * weights).sum().item())
            weighted_distillation_sum += float((distillation_losses * weights).sum().item())
            weight_sum += float(batch_weight_sum.item())
    if not bool(seen.all()) or weight_sum <= 0.0:
        raise RuntimeError("Student epoch did not visit every image exactly once.")
    metrics = metrics_from_image_logits(dataset.rows, all_logits, dataset.targets)
    supervised_loss = weighted_supervised_sum / weight_sum
    distillation_loss = weighted_distillation_sum / weight_sum
    return {
        "loss": supervised_loss + distillation_weight * distillation_loss,
        "supervised_loss": supervised_loss,
        "distillation_loss": distillation_loss,
        "mean_teacher_cosine_similarity": 1.0 - distillation_loss,
        **metrics,
    }


def _compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "loss": float(metrics["loss"]),
        "supervised_loss": float(metrics["supervised_loss"]),
        "distillation_loss": float(metrics["distillation_loss"]),
        "mean_teacher_cosine_similarity": float(metrics["mean_teacher_cosine_similarity"]),
        "source_mean_macro_f1": float(metrics["source_mean_macro_f1"]),
        "worst_source_macro_f1": float(metrics["worst_source_macro_f1"]),
        "by_source_macro_f1": {
            source: float(metrics["by_source"][source]["macro_f1"]) for source in SOURCE_ORDER
        },
        "mra_distance_unit_macro_f1": float(metrics["mra_distance_unit"]["macro_f1"]),
        "mra_paired_lesion_macro_f1": float(metrics["mra_paired_lesion"]["macro_f1"]),
    }


def _effective_unit_counts(rows: pd.DataFrame) -> dict[str, float]:
    return {
        source: float(rows.loc[rows["source"] == source, "view_mass"].sum())
        for source in SOURCE_ORDER
    }


def _raw_image_counts(rows: pd.DataFrame) -> dict[str, int]:
    return {source: int((rows["source"] == source).sum()) for source in SOURCE_ORDER}


def checkpoint_provenance(
    *,
    fold_index: int,
    manifest_fingerprint: str,
    teacher_feature_dim: int,
    seed: int,
) -> dict[str, object]:
    return {
        "architecture": ARCHITECTURE,
        "student_architecture": STUDENT_ARCHITECTURE,
        "student_weights": STUDENT_WEIGHTS_ID,
        "student_preprocessing": STUDENT_PREPROCESSING,
        "student_image_loading": STUDENT_IMAGE_LOADING,
        "teacher_model_id": DEFAULT_MODEL_ID,
        "teacher_model_revision": DEFAULT_MODEL_REVISION,
        "teacher_preprocessing": TEACHER_PREPROCESSING,
        "teacher_encoder_frozen": True,
        "teacher_feature_dim": teacher_feature_dim,
        "manifest_fingerprint": manifest_fingerprint,
        "pad_protocol": PAD_PROTOCOL,
        "mra_protocol": MRA_PROTOCOL,
        "fold_index": fold_index,
        "view_weighting": VIEW_WEIGHTING,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "selection_metric": SELECTION_METRIC,
        "seed": seed,
    }


def validate_checkpoint_payload(
    payload: dict[str, object],
    *,
    expected_provenance: dict[str, object],
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Student checkpoint must be a dictionary.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Student checkpoint is missing provenance.")
    mismatches = [
        f"{key}={provenance.get(key)!r}"
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    ]
    if payload.get("labels") != list(PAD_UFES_NATIVE_LABELS):
        mismatches.append(f"labels={payload.get('labels')!r}")
    if payload.get("sources") != list(SOURCE_ORDER):
        mismatches.append(f"sources={payload.get('sources')!r}")
    if not isinstance(payload.get("model_state_dict"), dict):
        mismatches.append("model_state_dict is missing")
    if not isinstance(payload.get("epoch"), int) or int(payload.get("epoch", 0)) <= 0:
        mismatches.append(f"epoch={payload.get('epoch')!r}")
    if mismatches:
        raise ValueError("Student checkpoint provenance mismatch: " + ", ".join(mismatches))


def train_distillation_fold(
    rows: pd.DataFrame,
    *,
    fold_index: int,
    pad_summary: dict[str, object],
    mra_summary: dict[str, object],
    teacher_cache: dict[str, object],
    student_image_cache: dict[str, torch.Tensor],
    manifest_fingerprint: str,
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    distillation_weight: float = DEFAULT_DISTILLATION_WEIGHT,
    num_workers: int = 0,
    seed: int = DEFAULT_SEED,
    pretrained: bool = True,
) -> dict[str, object]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if learning_rate <= 0.0 or weight_decay < 0.0 or distillation_weight < 0.0:
        raise ValueError("Invalid learning rate, weight decay, or distillation weight.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    set_seed(seed)
    split_rows = {split: rows.loc[rows["split"] == split].copy() for split in SPLIT_ORDER}
    train_dataset = DistillationImageDataset(
        split_rows["train"],
        teacher_cache,
        student_image_cache,
        training=True,
    )
    evaluation_datasets = {
        split: DistillationImageDataset(
            split_rows[split],
            teacher_cache,
            student_image_cache,
            training=False,
        )
        for split in SPLIT_ORDER
    }
    train_loader = build_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    evaluation_loaders = {
        split: build_loader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        for split, dataset in evaluation_datasets.items()
    }
    teacher_features = teacher_cache.get("features")
    if not isinstance(teacher_features, torch.Tensor) or teacher_features.ndim != 2:
        raise ValueError("Teacher cache is missing its feature tensor.")
    teacher_feature_dim = int(teacher_features.shape[1])
    model = build_student_model(teacher_feature_dim, pretrained=pretrained).to(device)
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Every student parameter must be trainable.")
    inference_parameter_count = model.inference_parameter_count()
    projection_parameter_count = model.projection_parameter_count()
    training_parameter_count = model.training_parameter_count()
    if training_parameter_count != inference_parameter_count + projection_parameter_count:
        raise RuntimeError("Student parameter accounting is inconsistent.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimized_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if optimized_parameter_ids != {id(parameter) for parameter in model.parameters()}:
        raise RuntimeError("Optimizer parameters differ from the complete student model.")

    checkpoint_path = resolve_project_path(Path(checkpoint_path))
    run_dir = resolve_project_path(Path(run_dir))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_checkpoint_provenance = checkpoint_provenance(
        fold_index=fold_index,
        manifest_fingerprint=manifest_fingerprint,
        teacher_feature_dim=teacher_feature_dim,
        seed=seed,
    )

    best_source_mean = -1.0
    best_val_supervised_loss = math.inf
    best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_student_epoch(
            model,
            train_loader,
            device,
            optimizer,
            distillation_weight=distillation_weight,
        )
        val_metrics = run_student_epoch(
            model,
            evaluation_loaders["val"],
            device,
            distillation_weight=distillation_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "train": _compact_metrics(train_metrics),
                "val": _compact_metrics(val_metrics),
            }
        )
        val_source_mean = float(val_metrics["source_mean_macro_f1"])
        val_supervised_loss = float(val_metrics["supervised_loss"])
        improved = val_source_mean > best_source_mean or (
            math.isclose(val_source_mean, best_source_mean, abs_tol=1e-12)
            and val_supervised_loss < best_val_supervised_loss
        )
        if improved:
            best_source_mean = val_source_mean
            best_val_supervised_loss = val_supervised_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "provenance": expected_checkpoint_provenance,
                    "labels": list(PAD_UFES_NATIVE_LABELS),
                    "sources": list(SOURCE_ORDER),
                    "epoch": epoch,
                    "val_source_mean_macro_f1": val_source_mean,
                    "val_supervised_loss": val_supervised_loss,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_payload(
        checkpoint,
        expected_provenance=expected_checkpoint_provenance,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics = {
        split: run_student_epoch(
            model,
            evaluation_loaders[split],
            device,
            distillation_weight=distillation_weight,
        )
        for split in SPLIT_ORDER
    }
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "supervised_loss_weight": 1.0,
        "distillation_loss_weight": distillation_weight,
        "optimizer": "AdamW",
        "schedule": "none",
        "augmentation_profile": "baseline",
        "view_weighting": VIEW_WEIGHTING,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
    }
    report: dict[str, object] = {
        "context": (
            "Experimental PAD/MRA ConvNeXt-Tiny MedSigLIP feature distillation; "
            "not medical certainty, independent validation, or production evidence."
        ),
        "architecture": ARCHITECTURE,
        "student_architecture": STUDENT_ARCHITECTURE,
        "student_weights": STUDENT_WEIGHTS_ID if pretrained else "none",
        "student_preprocessing": STUDENT_PREPROCESSING,
        "student_image_loading": STUDENT_IMAGE_LOADING,
        "teacher_model_id": DEFAULT_MODEL_ID,
        "teacher_model_revision": DEFAULT_MODEL_REVISION,
        "teacher_preprocessing": TEACHER_PREPROCESSING,
        "teacher_encoder_frozen": True,
        "teacher_encoder_trainable_parameter_count": teacher_cache.get("processor", {}).get(
            "encoder_trainable_parameter_count"
        ),
        "teacher_encoder_parameter_count": teacher_cache.get("processor", {}).get(
            "encoder_parameter_count"
        ),
        "teacher_feature_dim": teacher_feature_dim,
        "known_pad_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "mra_role": "authorized_multisource_development",
        "mra_training_input": "original_clinical_images",
        "mra_primary_metric_unit": PRIMARY_MRA_VIEW,
        "mra_secondary_aggregation": MRA_AGGREGATION,
        "manifest_fingerprint": manifest_fingerprint,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": seed,
        "pad_split_summary": pad_summary,
        "mra_split_summary": mra_summary,
        "source_total_raw_image_counts": _raw_image_counts(rows),
        "source_total_effective_unit_counts": _effective_unit_counts(rows),
        "dataset_raw_image_counts": {
            split: _raw_image_counts(split_rows[split]) for split in SPLIT_ORDER
        },
        "dataset_effective_unit_counts": {
            split: _effective_unit_counts(split_rows[split]) for split in SPLIT_ORDER
        },
        "inference_parameter_count": inference_parameter_count,
        "projection_parameter_count": projection_parameter_count,
        "training_parameter_count": training_parameter_count,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "hyperparameters": hyperparameters,
        "best_epoch": best_epoch,
        "selection_metric": SELECTION_METRIC,
        "best_val_source_mean_macro_f1": best_source_mean,
        "history": history,
        "selected_train": selected_metrics["train"],
        "selected_val": selected_metrics["val"],
        "test": selected_metrics["test"],
        "caveat": (
            "PAD-UFES-20 is known teacher pretraining data and MRA-MIDAS is student training "
            "data. Passing cannot establish external robustness, fairness, patient-self-photo, "
            "deployment, diagnosis, or medical readiness."
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _validate_split_summary(
    summary: object,
    *,
    protocol: str,
    fold_index: int,
    num_folds: int,
    name: str,
) -> list[str]:
    if not isinstance(summary, dict):
        return [f"{name} is missing"]
    expected = {
        "protocol": protocol,
        "num_folds": num_folds,
        "fold_index": fold_index,
        "test_outer_fold": fold_index,
        "validation_outer_fold": (fold_index + 1) % num_folds,
    }
    return [
        f"{name}.{key}={summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]


def validate_distillation_report(
    report: dict[str, object],
    *,
    fold_index: int,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    distillation_weight: float = DEFAULT_DISTILLATION_WEIGHT,
) -> None:
    expected = {
        "architecture": ARCHITECTURE,
        "student_architecture": STUDENT_ARCHITECTURE,
        "student_weights": STUDENT_WEIGHTS_ID,
        "student_preprocessing": STUDENT_PREPROCESSING,
        "student_image_loading": STUDENT_IMAGE_LOADING,
        "teacher_model_id": DEFAULT_MODEL_ID,
        "teacher_model_revision": DEFAULT_MODEL_REVISION,
        "teacher_preprocessing": TEACHER_PREPROCESSING,
        "teacher_encoder_frozen": True,
        "teacher_encoder_trainable_parameter_count": 0,
        "known_pad_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "mra_role": "authorized_multisource_development",
        "mra_training_input": "original_clinical_images",
        "mra_primary_metric_unit": PRIMARY_MRA_VIEW,
        "mra_secondary_aggregation": MRA_AGGREGATION,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": seed,
        "selection_metric": SELECTION_METRIC,
    }
    mismatches = [
        f"{key}={report.get(key)!r}" for key, value in expected.items() if report.get(key) != value
    ]
    mismatches.extend(
        _validate_split_summary(
            report.get("pad_split_summary"),
            protocol=PAD_PROTOCOL,
            fold_index=fold_index,
            num_folds=num_folds,
            name="pad_split_summary",
        )
    )
    mismatches.extend(
        _validate_split_summary(
            report.get("mra_split_summary"),
            protocol=MRA_PROTOCOL,
            fold_index=fold_index,
            num_folds=num_folds,
            name="mra_split_summary",
        )
    )
    hyperparameters = report.get("hyperparameters")
    hyperparameter_expected = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "supervised_loss_weight": 1.0,
        "distillation_loss_weight": distillation_weight,
        "optimizer": "AdamW",
        "schedule": "none",
        "augmentation_profile": "baseline",
        "view_weighting": VIEW_WEIGHTING,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
    }
    if not isinstance(hyperparameters, dict):
        mismatches.append("hyperparameters are missing")
    else:
        mismatches.extend(
            f"hyperparameters.{key}={hyperparameters.get(key)!r}"
            for key, value in hyperparameter_expected.items()
            if hyperparameters.get(key) != value
        )
    fingerprint = report.get("manifest_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        mismatches.append(f"manifest_fingerprint={fingerprint!r}")
    teacher_feature_dim = report.get("teacher_feature_dim")
    if not isinstance(teacher_feature_dim, int) or teacher_feature_dim <= 0:
        mismatches.append(f"teacher_feature_dim={teacher_feature_dim!r}")
    inference_count = report.get("inference_parameter_count")
    projection_count = report.get("projection_parameter_count")
    training_count = report.get("training_parameter_count")
    if not isinstance(inference_count, int) or inference_count <= 0:
        mismatches.append(f"inference_parameter_count={inference_count!r}")
    if not isinstance(projection_count, int) or projection_count <= 0:
        mismatches.append(f"projection_parameter_count={projection_count!r}")
    if (
        isinstance(inference_count, int)
        and isinstance(projection_count, int)
        and training_count != inference_count + projection_count
    ):
        mismatches.append(f"training_parameter_count={training_count!r}")
    if (
        isinstance(teacher_feature_dim, int)
        and teacher_feature_dim > 0
        and projection_count != (STUDENT_FEATURE_DIM + 1) * teacher_feature_dim
    ):
        mismatches.append(f"projection_parameter_count={projection_count!r}")
    if (
        not isinstance(report.get("checkpoint_bytes"), int)
        or int(report.get("checkpoint_bytes", 0)) <= 0
    ):
        mismatches.append(f"checkpoint_bytes={report.get('checkpoint_bytes')!r}")
    for split_name in ("selected_train", "selected_val", "test"):
        metrics = report.get(split_name)
        if not isinstance(metrics, dict):
            mismatches.append(f"{split_name} is missing")
            continue
        if set(metrics.get("by_source", {})) != set(SOURCE_ORDER):
            mismatches.append(f"{split_name}.by_source is invalid")
        for key in ("mra_distance_unit", "mra_paired_lesion"):
            if not isinstance(metrics.get(key), dict):
                mismatches.append(f"{split_name}.{key} is missing")
    if mismatches:
        raise ValueError(
            f"Distillation report fold_{fold_index} violates the locked protocol: "
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


def summarize_distillation_reports(
    reports_root: Path,
    out_path: Path,
    *,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    distillation_weight: float = DEFAULT_DISTILLATION_WEIGHT,
) -> dict[str, object]:
    reports_root = resolve_project_path(Path(reports_root))
    reports: list[dict[str, object]] = []
    for fold_index in range(num_folds):
        report_path = reports_root / f"fold_{fold_index}" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing distillation report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_distillation_report(
            report,
            fold_index=fold_index,
            num_folds=num_folds,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            distillation_weight=distillation_weight,
        )
        reports.append(report)

    consistency_keys = (
        "manifest_fingerprint",
        "student_weights",
        "teacher_feature_dim",
        "teacher_encoder_parameter_count",
        "teacher_encoder_trainable_parameter_count",
        "inference_parameter_count",
        "projection_parameter_count",
        "training_parameter_count",
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
            "Distillation reports disagree on locked provenance: " + ", ".join(disagreements)
        )

    pad_confusion = _pooled_report_confusion(reports, "by_source", "pad_ufes")
    mra_raw_confusion = _pooled_report_confusion(reports, "by_source", "mra_midas")
    mra_distance_confusion = _pooled_report_confusion(reports, "mra_distance_unit")
    mra_paired_confusion = _pooled_report_confusion(reports, "mra_paired_lesion")
    pooled_pad = _metrics_from_confusion(pad_confusion)
    pooled_mra_raw = _metrics_from_confusion(mra_raw_confusion)
    pooled_mra_distance = _metrics_from_confusion(mra_distance_confusion)
    pooled_mra_paired = _metrics_from_confusion(mra_paired_confusion)
    expected_raw_counts = reports[0]["source_total_raw_image_counts"]
    expected_effective_counts = reports[0]["source_total_effective_unit_counts"]
    expected_support = {
        "pad": int(expected_raw_counts["pad_ufes"]),
        "mra_raw": int(expected_raw_counts["mra_midas"]),
        "mra_distance": int(round(2 * float(expected_effective_counts["mra_midas"]))),
        "mra_paired": int(round(float(expected_effective_counts["mra_midas"]))),
    }
    observed_support = {
        "pad": int(pooled_pad["total_support"]),
        "mra_raw": int(pooled_mra_raw["total_support"]),
        "mra_distance": int(pooled_mra_distance["total_support"]),
        "mra_paired": int(pooled_mra_paired["total_support"]),
    }
    if observed_support != expected_support:
        raise ValueError(
            f"Outer-test coverage differs: observed={observed_support}, expected={expected_support}"
        )
    source_mean = statistics.fmean(
        [float(pooled_pad["macro_f1"]), float(pooled_mra_raw["macro_f1"])]
    )
    worst_source = min(float(pooled_pad["macro_f1"]), float(pooled_mra_raw["macro_f1"]))
    gaps = [
        float(report["selected_train"]["source_mean_macro_f1"])
        - float(report["selected_val"]["source_mean_macro_f1"])
        for report in reports
    ]
    fold_source_mean = [float(report["test"]["source_mean_macro_f1"]) for report in reports]
    inference_parameter_count = int(reports[0]["inference_parameter_count"])
    rules = {
        "mean_selected_train_val_primary_source_mean_macro_f1_gap_lte_0_1500": (
            statistics.fmean(gaps) <= 0.15
        ),
        "pooled_pad_image_macro_f1_gte_0_6100": float(pooled_pad["macro_f1"]) >= 0.61,
        "pooled_mra_raw_image_macro_f1_gte_0_4000": (float(pooled_mra_raw["macro_f1"]) >= 0.4),
        "pooled_primary_source_mean_macro_f1_gte_0_5200": source_mean >= 0.52,
        "pooled_primary_worst_source_macro_f1_gte_0_4000": worst_source >= 0.4,
        "pooled_mra_paired_lesion_macro_f1_gte_0_4500": (
            float(pooled_mra_paired["macro_f1"]) >= 0.45
        ),
        "pooled_pad_scc_f1_gte_0_3000": (
            float(pooled_pad["per_class"]["squamous_cell_carcinoma"]["f1"]) >= 0.3
        ),
        "pooled_mra_raw_image_melanoma_f1_gte_0_3000": (
            float(pooled_mra_raw["per_class"]["melanoma"]["f1"]) >= 0.3
        ),
        "pooled_mra_paired_lesion_melanoma_f1_gte_0_3500": (
            float(pooled_mra_paired["per_class"]["melanoma"]["f1"]) >= 0.35
        ),
        "pooled_mra_paired_lesion_scc_f1_gte_0_4500": (
            float(pooled_mra_paired["per_class"]["squamous_cell_carcinoma"]["f1"]) >= 0.45
        ),
        "inference_parameter_count_lte_30000000": inference_parameter_count <= 30_000_000,
    }
    summary: dict[str, object] = {
        "context": (
            "Experimental PAD/MRA ConvNeXt-Tiny MedSigLIP feature distillation; "
            "not medical certainty, independent validation, or production evidence."
        ),
        "pad_protocol": PAD_PROTOCOL,
        "mra_protocol": MRA_PROTOCOL,
        "num_folds": num_folds,
        "seed": seed,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "architecture": ARCHITECTURE,
        "student_architecture": STUDENT_ARCHITECTURE,
        "student_weights": STUDENT_WEIGHTS_ID,
        "student_preprocessing": STUDENT_PREPROCESSING,
        "student_image_loading": STUDENT_IMAGE_LOADING,
        "teacher_model_id": DEFAULT_MODEL_ID,
        "teacher_model_revision": DEFAULT_MODEL_REVISION,
        "teacher_preprocessing": TEACHER_PREPROCESSING,
        "teacher_encoder_frozen": True,
        "teacher_encoder_trainable_parameter_count": 0,
        "teacher_feature_dim": int(reports[0]["teacher_feature_dim"]),
        "known_pad_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "mra_role": "authorized_multisource_development",
        "mra_primary_metric_unit": PRIMARY_MRA_VIEW,
        "mra_secondary_aggregation": MRA_AGGREGATION,
        "view_weighting": VIEW_WEIGHTING,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "selection_metric": SELECTION_METRIC,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "supervised_loss_weight": 1.0,
        "distillation_loss_weight": distillation_weight,
        "inference_parameter_count": inference_parameter_count,
        "projection_parameter_count": int(reports[0]["projection_parameter_count"]),
        "training_parameter_count": int(reports[0]["training_parameter_count"]),
        "checkpoint_bytes": _distribution(
            [float(report["checkpoint_bytes"]) for report in reports]
        ),
        "source_total_raw_image_counts": expected_raw_counts,
        "source_total_effective_unit_counts": expected_effective_counts,
        "fold_primary_source_mean_macro_f1": _distribution(fold_source_mean),
        "selected_train_val_primary_source_mean_macro_f1_gap": _distribution(gaps),
        "pooled_primary_by_source": {
            "pad_ufes": pooled_pad,
            "mra_midas": pooled_mra_raw,
        },
        "pooled_primary_source_mean_macro_f1": source_mean,
        "pooled_primary_worst_source_macro_f1": worst_source,
        "pooled_mra_distance_unit": pooled_mra_distance,
        "pooled_mra_paired_lesion": pooled_mra_paired,
        "decision_rules": {**rules, "all_pass": all(rules.values())},
        "folds": [
            {
                "fold_index": fold_index,
                "best_epoch": int(report["best_epoch"]),
                "best_val_source_mean_macro_f1": float(report["best_val_source_mean_macro_f1"]),
                "selected_train_source_mean_macro_f1": float(
                    report["selected_train"]["source_mean_macro_f1"]
                ),
                "selected_val_source_mean_macro_f1": float(
                    report["selected_val"]["source_mean_macro_f1"]
                ),
                "test_source_mean_macro_f1": float(report["test"]["source_mean_macro_f1"]),
                "test_macro_f1_by_source": {
                    source: float(report["test"]["by_source"][source]["macro_f1"])
                    for source in SOURCE_ORDER
                },
                "test_mra_paired_lesion_macro_f1": float(
                    report["test"]["mra_paired_lesion"]["macro_f1"]
                ),
            }
            for fold_index, report in enumerate(reports)
        ],
        "caveat": (
            "PAD-UFES-20 is known teacher pretraining data and MRA-MIDAS is student training "
            "data. Passing cannot establish external robustness, fairness, patient-self-photo, "
            "deployment, diagnosis, or medical readiness."
        ),
    }
    out_path = resolve_project_path(Path(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.model_id != DEFAULT_MODEL_ID or args.revision != DEFAULT_MODEL_REVISION:
        raise ValueError(
            "The teacher model ID and revision are locked by the preregistered protocol."
        )
    if not 0 < args.embedding_batch_size <= DEFAULT_EMBEDDING_BATCH_SIZE:
        raise ValueError("embedding_batch_size may only be reduced from the locked value of 8.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    device = get_device(args.device)
    print(f"Using device: {device}")
    manifests = load_multi_source_manifests(
        args.pad_splits_dir,
        args.mra_splits_dir,
        num_folds=DEFAULT_FOLDS,
    )
    print(
        f"Validated {DEFAULT_FOLDS} PAD/MRA rotating folds over "
        f"{len(manifests.image_rows):,} source images; fingerprint={manifests.fingerprint}"
    )
    teacher_cache = load_or_extract_image_embeddings(
        manifests,
        embeddings_path=args.teacher_embeddings,
        model_id=args.model_id,
        revision=args.revision,
        model_cache_dir=args.model_cache_dir,
        device=device,
        batch_size=args.embedding_batch_size,
        num_workers=args.num_workers,
    )
    if teacher_cache.get("processor", {}).get("encoder_trainable_parameter_count") != 0:
        raise ValueError("Teacher cache records trainable encoder parameters.")
    print("Preloading deterministic 224-pixel student tensors")
    student_image_cache = preload_student_image_tensors(manifests.image_rows)
    print(f"Preloaded {len(student_image_cache):,} unique student image tensors")
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
        if args.resume and report_path.exists():
            print(f"fold_{fold_index}: report exists; skipping")
            continue
        rows = fold_image_rows(manifests, fold_index=fold_index)
        print(f"fold_{fold_index}: starting ConvNeXt-Tiny feature distillation")
        report = train_distillation_fold(
            rows,
            fold_index=fold_index,
            pad_summary=manifests.pad.fold_summaries[fold_index],
            mra_summary=manifests.mra_fold_summaries[fold_index],
            teacher_cache=teacher_cache,
            student_image_cache=student_image_cache,
            manifest_fingerprint=manifests.fingerprint,
            checkpoint_path=checkpoints_dir / f"fold_{fold_index}.pt",
            run_dir=run_dir,
            device=device,
            num_workers=args.num_workers,
        )
        validate_distillation_report(report, fold_index=fold_index)
        print(
            f"fold_{fold_index}: best_epoch={report['best_epoch']} "
            f"test_primary_source_mean_macro_f1="
            f"{report['test']['source_mean_macro_f1']:.4f}"
        )

    summary = summarize_distillation_reports(
        runs_root,
        runs_root / "summary.json",
    )
    print(
        "Distillation complete: "
        f"pooled_primary_source_mean_macro_f1="
        f"{summary['pooled_primary_source_mean_macro_f1']:.4f} "
        f"pooled_primary_worst_source_macro_f1="
        f"{summary['pooled_primary_worst_source_macro_f1']:.4f} "
        f"all_rules_pass={summary['decision_rules']['all_pass']} "
        f"summary={project_relative(runs_root / 'summary.json')}"
    )
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

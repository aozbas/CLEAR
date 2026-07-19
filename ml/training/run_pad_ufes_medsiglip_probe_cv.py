"""Run a frozen MedSigLIP linear probe on grouped PAD-UFES-native folds.

This is an experimental representation-learning workflow, not medical evidence. MedSigLIP lists
PAD-UFES-20 in its pretraining data, so these results are not independent external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.metrics import per_class_metrics
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_pad_ufes import project_relative
from ml.training.prepare_pad_ufes_cv import DEFAULT_FOLDS, PROTOCOL
from ml.training.train import (
    build_loader,
    class_weights,
    get_device,
    resolve_project_path,
    run_epoch,
    set_seed,
)
from ml.training.train_pad_ufes import add_macro_metrics, load_training_split, print_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "google/medsiglip-448"
DEFAULT_MODEL_REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes_native_cv"
DEFAULT_EMBEDDINGS_PATH = (
    PROJECT_ROOT / "ml" / "runs" / "embeddings" / "pad_ufes_medsiglip_448_rev9cea28a.pt"
)
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT / "ml" / "runs" / "training" / "pad_ufes_medsiglip_linear_probe-cv-seed42"
)
DEFAULT_CHECKPOINTS_DIR = (
    PROJECT_ROOT / "ml" / "models" / "pad_ufes_medsiglip_linear_probe_cv_seed42"
)
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "ml" / "model_cache" / "huggingface"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 100
DEFAULT_PROBE_BATCH_SIZE = 128
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_WEIGHT_DECAY = 1e-2
CACHE_SCHEMA_VERSION = 1
ARCHITECTURE = "medsiglip_frozen_linear_probe"
PREPROCESSING = "medsiglip_auto_image_processor_slow_native_448"
IMBALANCE_STRATEGY = "inverse_frequency_loss"
KNOWN_PRETRAINING_DATASETS = ("PAD-UFES-20", "SCIN")
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)

ProcessorLoader = Callable[..., Any]
ModelLoader = Callable[..., nn.Module]


@dataclass(frozen=True)
class CvManifests:
    fold_rows: tuple[pd.DataFrame, ...]
    fold_summaries: tuple[dict[str, object], ...]
    unique_rows: pd.DataFrame
    fingerprint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen MedSigLIP PAD-UFES-native grouped-CV linear probe."
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PROBE_BATCH_SIZE)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_cv_manifests(splits_dir: Path, *, num_folds: int) -> CvManifests:
    if num_folds < 3:
        raise ValueError("folds must be at least 3.")
    splits_dir = resolve_project_path(Path(splits_dir))
    fold_rows: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    reference_mapping: dict[str, str] | None = None

    for fold_index in range(num_folds):
        split_csv = splits_dir / f"fold_{fold_index}.csv"
        rows, summary = load_training_split(split_csv)
        _validate_fold_summary(summary, fold_index=fold_index, num_folds=num_folds)
        mapping = dict(zip(rows["image_path"].astype(str), rows["label"].astype(str), strict=True))
        if reference_mapping is None:
            reference_mapping = mapping
        elif mapping != reference_mapping:
            raise ValueError(
                f"PAD-UFES fold_{fold_index} image-to-label mapping differs from fold_0."
            )
        fold_rows.append(rows.copy())
        fold_summaries.append(summary)

    assert reference_mapping is not None
    test_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    for rows in fold_rows:
        test_counts.update(rows.loc[rows["split"] == "test", "image_path"].astype(str))
        validation_counts.update(rows.loc[rows["split"] == "val", "image_path"].astype(str))
    if set(test_counts) != set(reference_mapping) or any(
        count != 1 for count in test_counts.values()
    ):
        raise ValueError("Every PAD-UFES image must be outer-fold test data exactly once.")
    if set(validation_counts) != set(reference_mapping) or any(
        count != 1 for count in validation_counts.values()
    ):
        raise ValueError("Every PAD-UFES image must be validation data exactly once.")

    unique_rows = pd.DataFrame(
        sorted(reference_mapping.items()),
        columns=("image_path", "label"),
    )
    fingerprint = _manifest_fingerprint(fold_rows, fold_summaries)
    return CvManifests(
        fold_rows=tuple(fold_rows),
        fold_summaries=tuple(fold_summaries),
        unique_rows=unique_rows,
        fingerprint=fingerprint,
    )


def _validate_fold_summary(
    summary: dict[str, object],
    *,
    fold_index: int,
    num_folds: int,
) -> None:
    expected = {
        "protocol": PROTOCOL,
        "num_folds": num_folds,
        "fold_index": fold_index,
        "test_outer_fold": fold_index,
        "validation_outer_fold": (fold_index + 1) % num_folds,
        "cv_total_image_count": summary.get("image_count"),
    }
    mismatches = [
        f"{key}={summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"PAD-UFES fold_{fold_index} violates the rotating-CV protocol: {', '.join(mismatches)}"
        )


def _manifest_fingerprint(
    fold_rows: Sequence[pd.DataFrame],
    fold_summaries: Sequence[dict[str, object]],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"schema={CACHE_SCHEMA_VERSION}\nprotocol={PROTOCOL}\n".encode())
    for fold_index, (rows, summary) in enumerate(zip(fold_rows, fold_summaries, strict=True)):
        digest.update(
            json.dumps(
                {
                    "fold_index": fold_index,
                    "num_folds": summary.get("num_folds"),
                    "test_outer_fold": summary.get("test_outer_fold"),
                    "validation_outer_fold": summary.get("validation_outer_fold"),
                },
                sort_keys=True,
            ).encode()
        )
        for row in rows.sort_values("image_path").itertuples(index=False):
            digest.update(f"\n{fold_index}\0{row.split}\0{row.image_path}\0{row.label}".encode())
    return digest.hexdigest()


class _ImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame) -> None:
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[int, Image.Image]:
        image_path = resolve_project_path(Path(str(self.rows.iloc[index]["image_path"])))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing PAD-UFES image: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        return index, image


def _collate_images(batch: list[tuple[int, Image.Image]]) -> tuple[list[int], list[Image.Image]]:
    indices, images = zip(*batch, strict=True)
    return list(indices), list(images)


def extract_embeddings(
    rows: pd.DataFrame,
    *,
    model_id: str,
    revision: str,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    processor_loader: ProcessorLoader | None = None,
    model_loader: ModelLoader | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    processor_kwargs: dict[str, object] = {}
    if processor_loader is None or model_loader is None:
        from transformers import AutoImageProcessor, AutoModel

        if processor_loader is None:
            processor_loader = AutoImageProcessor.from_pretrained
            processor_kwargs["use_fast"] = False
        model_loader = model_loader or AutoModel.from_pretrained

    cache_dir = resolve_project_path(Path(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor = processor_loader(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        **processor_kwargs,
    )
    encoder = model_loader(model_id, revision=revision, cache_dir=cache_dir).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("MedSigLIP encoder parameters must remain frozen.")

    loader = DataLoader(
        _ImageDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_images,
    )
    feature_batches: list[torch.Tensor] = []
    expected_index = 0
    with torch.inference_mode():
        for indices, images in loader:
            if indices != list(range(expected_index, expected_index + len(indices))):
                raise RuntimeError("Embedding loader changed the deterministic image order.")
            expected_index += len(indices)
            inputs = processor(images=images, return_tensors="pt")
            if "pixel_values" not in inputs:
                raise ValueError("MedSigLIP processor did not return pixel_values.")
            pixel_values = inputs["pixel_values"].to(device)
            outputs = encoder.get_image_features(pixel_values=pixel_values)
            pooled = _pooled_features(outputs)
            if pooled.ndim != 2 or pooled.shape[0] != len(images):
                raise ValueError(
                    "MedSigLIP pooled image features must have shape [batch, feature_dim]."
                )
            normalized = functional.normalize(pooled.float(), p=2, dim=-1)
            if not bool(torch.isfinite(normalized).all()):
                raise ValueError("MedSigLIP produced non-finite image embeddings.")
            feature_batches.append(normalized.cpu())

    if expected_index != len(rows):
        raise RuntimeError("Embedding extraction did not cover the complete PAD-UFES manifest.")
    features = torch.cat(feature_batches, dim=0)
    image_processor = getattr(processor, "image_processor", processor)
    processor_metadata = {
        "class": processor.__class__.__name__,
        "image_processor_class": image_processor.__class__.__name__,
        "size": _json_safe(getattr(image_processor, "size", None)),
        "crop_size": _json_safe(getattr(image_processor, "crop_size", None)),
        "encoder_class": encoder.__class__.__name__,
        "encoder_parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
        "encoder_trainable_parameter_count": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
    }
    return features, processor_metadata


def _pooled_features(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    pooled = getattr(outputs, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
        pooled = outputs[1]
        if isinstance(pooled, torch.Tensor):
            return pooled
    raise ValueError("MedSigLIP did not return a pooled image embedding.")


def _json_safe(value: Any) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_cache_payload(
    *,
    rows: pd.DataFrame,
    features: torch.Tensor,
    model_id: str,
    revision: str,
    manifest_fingerprint: str,
    processor_metadata: dict[str, object],
) -> dict[str, object]:
    features = features.detach().cpu().float().contiguous()
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise ValueError("Embedding tensor shape does not match the unique manifest.")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Embedding tensor contains non-finite values.")
    norms = features.norm(dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)):
        raise ValueError("Embedding tensor must be L2 normalized.")
    label_to_index = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
    labels = rows["label"].astype(str).tolist()
    targets = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "context": "Frozen MedSigLIP PAD-UFES embeddings; experimental classification only.",
        "model_id": model_id,
        "model_revision": revision,
        "known_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "preprocessing": PREPROCESSING,
        "embedding_normalization": "l2",
        "manifest_fingerprint": manifest_fingerprint,
        "image_count": len(rows),
        "feature_dim": int(features.shape[1]),
        "image_paths": rows["image_path"].astype(str).tolist(),
        "image_labels": labels,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "targets": targets,
        "features": features,
        "processor": processor_metadata,
    }


def load_embedding_cache(
    cache_path: Path,
    *,
    rows: pd.DataFrame,
    model_id: str,
    revision: str,
    manifest_fingerprint: str,
) -> dict[str, object]:
    cache_path = resolve_project_path(Path(cache_path))
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Embedding cache payload must be a dictionary.")
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": revision,
        "known_pretraining_overlap": True,
        "preprocessing": PREPROCESSING,
        "embedding_normalization": "l2",
        "manifest_fingerprint": manifest_fingerprint,
        "image_count": len(rows),
        "image_paths": rows["image_path"].astype(str).tolist(),
        "image_labels": rows["label"].astype(str).tolist(),
        "labels": list(PAD_UFES_NATIVE_LABELS),
    }
    mismatches = [
        f"{key}={payload.get(key)!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"Embedding cache provenance mismatch: {', '.join(mismatches)}")
    features = payload.get("features")
    targets = payload.get("targets")
    if not isinstance(features, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError("Embedding cache is missing tensor features or targets.")
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise ValueError("Embedding cache feature shape does not match the manifest.")
    if payload.get("feature_dim") != int(features.shape[1]):
        raise ValueError("Embedding cache feature_dim does not match its tensor.")
    if tuple(targets.shape) != (len(rows),):
        raise ValueError("Embedding cache target shape does not match the manifest.")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Embedding cache contains non-finite features.")
    norms = features.norm(dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)):
        raise ValueError("Embedding cache features are not L2 normalized.")
    label_to_index = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
    expected_targets = torch.tensor(
        [label_to_index[label] for label in rows["label"].astype(str)],
        dtype=torch.long,
    )
    if not torch.equal(targets.cpu().long(), expected_targets):
        raise ValueError("Embedding cache targets do not match manifest labels.")
    payload["features"] = features.cpu().float()
    payload["targets"] = targets.cpu().long()
    return payload


def load_or_extract_embeddings(
    manifests: CvManifests,
    *,
    embeddings_path: Path,
    model_id: str,
    revision: str,
    model_cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    embeddings_path = resolve_project_path(Path(embeddings_path))
    if embeddings_path.exists():
        print(f"Validating cached embeddings: {project_relative(embeddings_path)}")
        return load_embedding_cache(
            embeddings_path,
            rows=manifests.unique_rows,
            model_id=model_id,
            revision=revision,
            manifest_fingerprint=manifests.fingerprint,
        )

    features, processor_metadata = extract_embeddings(
        manifests.unique_rows,
        model_id=model_id,
        revision=revision,
        cache_dir=model_cache_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    payload = build_cache_payload(
        rows=manifests.unique_rows,
        features=features,
        model_id=model_id,
        revision=revision,
        manifest_fingerprint=manifests.fingerprint,
        processor_metadata=processor_metadata,
    )
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = embeddings_path.with_suffix(embeddings_path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(embeddings_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote frozen embeddings: {project_relative(embeddings_path)}")
    return payload


class EmbeddingDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, cache: dict[str, object]) -> None:
        cache_paths = [str(path) for path in cache["image_paths"]]
        cache_labels = [str(label) for label in cache["image_labels"]]
        index_by_path = {path: index for index, path in enumerate(cache_paths)}
        label_by_path = dict(zip(cache_paths, cache_labels, strict=True))
        if len(index_by_path) != len(cache_paths):
            raise ValueError("Embedding cache contains duplicate image paths.")

        indices: list[int] = []
        for row in rows.itertuples(index=False):
            path = str(row.image_path)
            if path not in index_by_path:
                raise ValueError(f"Embedding cache is missing requested image: {path}")
            if label_by_path[path] != str(row.label):
                raise ValueError(f"Embedding cache label differs for image: {path}")
            indices.append(index_by_path[path])
        index_tensor = torch.tensor(indices, dtype=torch.long)
        self.features = cache["features"].index_select(0, index_tensor)
        self.targets = cache["targets"].index_select(0, index_tensor)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.features[index], int(self.targets[index])

    def labels(self) -> list[int]:
        return self.targets.tolist()


def train_probe_fold(
    rows: pd.DataFrame,
    *,
    split_summary: dict[str, object],
    cache: dict[str, object],
    checkpoint_path: Path,
    run_dir: Path,
    model_id: str,
    revision: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, object]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and probe batch size must be positive.")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative.")
    set_seed(seed)
    datasets = {
        split: EmbeddingDataset(rows.loc[rows["split"] == split], cache)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: build_loader(
            dataset,
            batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    feature_dim = int(cache["feature_dim"])
    head = nn.Linear(feature_dim, len(PAD_UFES_NATIVE_LABELS)).to(device)
    trainable_parameter_count = sum(parameter.numel() for parameter in head.parameters())
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            datasets["train"].labels(),
            len(PAD_UFES_NATIVE_LABELS),
            device,
        )
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    checkpoint_path = resolve_project_path(Path(checkpoint_path))
    run_dir = resolve_project_path(Path(run_dir))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    best_val_loss = math.inf
    best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = add_macro_metrics(
            run_epoch(
                head,
                loaders["train"],
                criterion,
                device,
                list(PAD_UFES_NATIVE_LABELS),
                optimizer,
            )
        )
        val_metrics = add_macro_metrics(
            run_epoch(
                head,
                loaders["val"],
                criterion,
                device,
                list(PAD_UFES_NATIVE_LABELS),
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train": _compact_metrics(train_metrics),
                "val": _compact_metrics(val_metrics),
            }
        )
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_loss = float(val_metrics["loss"])
        improved = val_macro_f1 > best_macro_f1 or (
            math.isclose(val_macro_f1, best_macro_f1, abs_tol=1e-12) and val_loss < best_val_loss
        )
        if improved:
            best_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": head.state_dict(),
                    "architecture": ARCHITECTURE,
                    "model_id": model_id,
                    "model_revision": revision,
                    "encoder_frozen": True,
                    "embedding_normalization": "l2",
                    "feature_dim": feature_dim,
                    "labels": list(PAD_UFES_NATIVE_LABELS),
                    "epoch": epoch,
                    "seed": seed,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics = {
        split: add_macro_metrics(
            run_epoch(
                head,
                loaders[split],
                criterion,
                device,
                list(PAD_UFES_NATIVE_LABELS),
            )
        )
        for split in ("train", "val", "test")
    }
    for split, metrics in selected_metrics.items():
        print_metrics(split, metrics)

    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "imbalance_strategy": IMBALANCE_STRATEGY,
    }
    report: dict[str, object] = {
        "context": (
            "Experimental frozen MedSigLIP PAD-UFES-native representation probe; "
            "not medical certainty or independent validation."
        ),
        "architecture": ARCHITECTURE,
        "model_id": model_id,
        "model_revision": revision,
        "encoder_frozen": True,
        "known_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "embedding_normalization": "l2",
        "preprocessing": PREPROCESSING,
        "manifest_fingerprint": cache.get("manifest_fingerprint"),
        "feature_dim": feature_dim,
        "trainable_parameter_count": trainable_parameter_count,
        "encoder_parameter_count": cache.get("processor", {}).get("encoder_parameter_count"),
        "encoder_trainable_parameter_count": cache.get("processor", {}).get(
            "encoder_trainable_parameter_count"
        ),
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "seed": seed,
        "split_summary": split_summary,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "hyperparameters": hyperparameters,
        "best_epoch": best_epoch,
        "selection_metric": "val_macro_f1",
        "best_val_macro_f1": best_macro_f1,
        "history": history,
        "selected_train": selected_metrics["train"],
        "selected_val": selected_metrics["val"],
        "test": selected_metrics["test"],
        "caveat": (
            "MedSigLIP lists PAD-UFES-20 in pretraining data; this is domain-fit evidence only."
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _compact_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        **{metric: float(metrics[metric]) for metric in METRIC_NAMES},
    }


def summarize_probe_reports(
    reports_root: Path,
    out_path: Path,
    *,
    num_folds: int = DEFAULT_FOLDS,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_PROBE_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> dict[str, object]:
    reports_root = resolve_project_path(Path(reports_root))
    reports: list[dict[str, object]] = []
    for fold_index in range(num_folds):
        report_path = reports_root / f"fold_{fold_index}" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing MedSigLIP probe report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _validate_probe_report(
            report,
            fold_index=fold_index,
            num_folds=num_folds,
            model_id=model_id,
            revision=revision,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        reports.append(report)

    expected_counts = {int(report["split_summary"]["cv_total_image_count"]) for report in reports}
    if len(expected_counts) != 1:
        raise ValueError("MedSigLIP probe reports disagree on the PAD-UFES image count.")
    expected_count = expected_counts.pop()
    pooled_confusion = _pooled_confusion(reports)
    if sum(sum(row) for row in pooled_confusion) != expected_count:
        raise ValueError("Probe outer-test folds do not cover PAD-UFES exactly once.")

    train_val_gaps = [
        float(report["selected_train"]["macro_f1"]) - float(report["selected_val"]["macro_f1"])
        for report in reports
    ]
    fold_metrics = {
        metric: _distribution([float(report["test"][metric]) for report in reports])
        for metric in METRIC_NAMES
    }
    pooled_test = _summarize_confusion(pooled_confusion)
    rules = {
        "mean_selected_train_val_macro_f1_gap_lte_0_2000": (
            statistics.fmean(train_val_gaps) <= 0.2
        ),
        "mean_fold_macro_f1_gte_0_6450": fold_metrics["macro_f1"]["mean"] >= 0.645,
        "pooled_macro_f1_gte_0_6500": float(pooled_test["macro_f1"]) >= 0.65,
        "pooled_balanced_accuracy_gte_0_6387": (float(pooled_test["balanced_accuracy"]) >= 0.6387),
        "pooled_scc_f1_gte_0_2443730": (
            float(pooled_test["per_class"]["squamous_cell_carcinoma"]["f1"]) >= 0.2443730
        ),
    }
    summary: dict[str, object] = {
        "context": (
            "Experimental frozen MedSigLIP PAD-UFES-native grouped cross-validation; "
            "not medical certainty or independent validation."
        ),
        "protocol": PROTOCOL,
        "num_folds": num_folds,
        "seed": seed,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "architecture": ARCHITECTURE,
        "model_id": model_id,
        "model_revision": revision,
        "encoder_frozen": True,
        "known_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "embedding_normalization": "l2",
        "preprocessing": PREPROCESSING,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "imbalance_strategy": IMBALANCE_STRATEGY,
        "fold_metrics": fold_metrics,
        "best_validation_macro_f1": _distribution(
            [float(report["best_val_macro_f1"]) for report in reports]
        ),
        "selected_train_val_macro_f1_gap": _distribution(train_val_gaps),
        "pooled_test": pooled_test,
        "decision_rules": {**rules, "all_pass": all(rules.values())},
        "folds": [
            {
                "fold_index": fold_index,
                "best_epoch": int(report["best_epoch"]),
                "best_val_macro_f1": float(report["best_val_macro_f1"]),
                "selected_train_macro_f1": float(report["selected_train"]["macro_f1"]),
                "selected_val_macro_f1": float(report["selected_val"]["macro_f1"]),
                "test": {metric: float(report["test"][metric]) for metric in METRIC_NAMES},
            }
            for fold_index, report in enumerate(reports)
        ],
        "caveat": (
            "PAD-UFES-20 is known MedSigLIP pretraining data. Passing internal rules cannot "
            "support external robustness, fairness, deployment, diagnosis, or "
            "medical-readiness claims."
        ),
    }
    out_path = resolve_project_path(Path(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _validate_probe_report(
    report: dict[str, object],
    *,
    fold_index: int,
    num_folds: int,
    model_id: str,
    revision: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    expected = {
        "architecture": ARCHITECTURE,
        "model_id": model_id,
        "model_revision": revision,
        "encoder_frozen": True,
        "known_pretraining_overlap": True,
        "embedding_normalization": "l2",
        "preprocessing": PREPROCESSING,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "seed": seed,
        "selection_metric": "val_macro_f1",
    }
    mismatches = [
        f"{key}={report.get(key)!r}" for key, value in expected.items() if report.get(key) != value
    ]
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
            for key, value in split_expected.items()
            if split_summary.get(key) != value
        )
    hyperparameters = report.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        mismatches.append("hyperparameters are missing")
    else:
        hyperparameter_expected = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "imbalance_strategy": IMBALANCE_STRATEGY,
        }
        mismatches.extend(
            f"hyperparameters.{key}={hyperparameters.get(key)!r}"
            for key, value in hyperparameter_expected.items()
            if hyperparameters.get(key) != value
        )
    if mismatches:
        raise ValueError(
            f"MedSigLIP probe report fold_{fold_index} violates the locked protocol: "
            f"{', '.join(mismatches)}"
        )


def _pooled_confusion(reports: Sequence[dict[str, object]]) -> list[list[int]]:
    size = len(PAD_UFES_NATIVE_LABELS)
    pooled = [[0 for _ in range(size)] for _ in range(size)]
    for report in reports:
        confusion = report["test"]["confusion_matrix"]
        if len(confusion) != size or any(len(row) != size for row in confusion):
            raise ValueError("A probe confusion matrix does not match the native label count.")
        for row_index in range(size):
            for column_index in range(size):
                pooled[row_index][column_index] += int(confusion[row_index][column_index])
    return pooled


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence.")
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


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


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.folds < 3:
        raise ValueError("folds must be at least 3.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.embedding_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive.")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    device = get_device(args.device)
    print(f"Using device: {device}")
    manifests = load_cv_manifests(args.splits_dir, num_folds=args.folds)
    print(
        f"Validated {args.folds} rotating folds over {len(manifests.unique_rows):,} images; "
        f"fingerprint={manifests.fingerprint}"
    )
    cache = load_or_extract_embeddings(
        manifests,
        embeddings_path=args.embeddings,
        model_id=args.model_id,
        revision=args.revision,
        model_cache_dir=args.model_cache_dir,
        device=device,
        batch_size=args.embedding_batch_size,
        num_workers=args.num_workers,
    )
    if cache.get("processor", {}).get("encoder_trainable_parameter_count") not in (None, 0):
        raise ValueError("Embedding cache records trainable encoder parameters.")

    runs_root = resolve_project_path(args.runs_root)
    checkpoints_dir = resolve_project_path(args.checkpoints_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for fold_index, (rows, summary) in enumerate(
        zip(manifests.fold_rows, manifests.fold_summaries, strict=True)
    ):
        run_dir = runs_root / f"fold_{fold_index}"
        report_path = run_dir / "report.json"
        if args.resume and report_path.exists():
            print(f"fold_{fold_index}: report exists; skipping")
            continue
        print(f"fold_{fold_index}: starting frozen linear probe")
        report = train_probe_fold(
            rows,
            split_summary=summary,
            cache=cache,
            checkpoint_path=checkpoints_dir / f"fold_{fold_index}.pt",
            run_dir=run_dir,
            model_id=args.model_id,
            revision=args.revision,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        print(
            f"fold_{fold_index}: best_epoch={report['best_epoch']} "
            f"test_macro_f1={report['test']['macro_f1']:.4f}"
        )

    summary = summarize_probe_reports(
        runs_root,
        runs_root / "summary.json",
        num_folds=args.folds,
        model_id=args.model_id,
        revision=args.revision,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        "MedSigLIP probe complete: "
        f"mean_fold_macro_f1={summary['fold_metrics']['macro_f1']['mean']:.4f} "
        f"pooled_macro_f1={summary['pooled_test']['macro_f1']:.4f} "
        f"all_rules_pass={summary['decision_rules']['all_pass']}"
    )
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

"""Run a frozen MedSigLIP linear probe over PAD-UFES and MRA-MIDAS development folds.

This workflow records internal multi-source experimental-classification evidence only. PAD-UFES
is known MedSigLIP pretraining data, and MRA-MIDAS is used for training here, so neither source is
an independent holdout in this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.metrics import per_class_metrics
from ml.evaluation.mra_midas import CLINICAL_DISTANCES
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_mra_midas_cv import PROTOCOL as MRA_PROTOCOL
from ml.training.prepare_pad_ufes import SPLIT_ORDER, project_relative
from ml.training.prepare_pad_ufes_cv import PROTOCOL as PAD_PROTOCOL
from ml.training.run_pad_ufes_medsiglip_probe_cv import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    KNOWN_PRETRAINING_DATASETS,
    PREPROCESSING,
    CvManifests,
    extract_embeddings,
    load_cv_manifests,
)
from ml.training.train import build_loader, get_device, resolve_project_path, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAD_SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes_native_cv"
DEFAULT_MRA_SPLITS_DIR = (
    PROJECT_ROOT / "ml" / "data" / "external_splits" / "mra_midas_multisource_cv"
)
DEFAULT_EMBEDDINGS_PATH = (
    PROJECT_ROOT / "ml" / "runs" / "embeddings" / "pad_mra_medsiglip_448_rev9cea28a.pt"
)
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT / "ml" / "runs" / "training" / "pad_mra_medsiglip_linear_probe-cv-seed42"
)
DEFAULT_CHECKPOINTS_DIR = (
    PROJECT_ROOT / "ml" / "models" / "pad_mra_medsiglip_linear_probe_cv_seed42"
)
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "ml" / "model_cache" / "huggingface"
DEFAULT_SEED = 42
DEFAULT_FOLDS = 5
DEFAULT_EPOCHS = 100
DEFAULT_PROBE_BATCH_SIZE = 128
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_WEIGHT_DECAY = 1e-2
CACHE_SCHEMA_VERSION = 1
ARCHITECTURE = "medsiglip_frozen_multisource_linear_probe"
SOURCE_ORDER = ("pad_ufes", "mra_midas")
MRA_AGGREGATION = "mean_within_distance_then_equal_distance_mean_then_l2"
WEIGHTING = "equal_total_weight_per_source_class_cell"
SELECTION_METRIC = "val_source_mean_macro_f1"
MRA_REQUIRED_COLUMNS = {
    "split",
    "source",
    "unit_id",
    "record_id",
    "distance",
    "image_path",
    "label",
}


@dataclass(frozen=True)
class MultiSourceManifests:
    pad: CvManifests
    mra_fold_rows: tuple[pd.DataFrame, ...]
    mra_fold_summaries: tuple[dict[str, object], ...]
    image_rows: pd.DataFrame
    fingerprint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen MedSigLIP PAD-UFES + MRA-MIDAS grouped-CV probe."
    )
    parser.add_argument("--pad-splits-dir", type=Path, default=DEFAULT_PAD_SPLITS_DIR)
    parser.add_argument("--mra-splits-dir", type=Path, default=DEFAULT_MRA_SPLITS_DIR)
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


def load_multi_source_manifests(
    pad_splits_dir: Path,
    mra_splits_dir: Path,
    *,
    num_folds: int,
) -> MultiSourceManifests:
    pad = load_cv_manifests(pad_splits_dir, num_folds=num_folds)
    mra_splits_dir = resolve_project_path(Path(mra_splits_dir))
    fold_rows: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    reference_mapping: dict[str, tuple[str, str, str, str]] | None = None

    for fold_index in range(num_folds):
        fold_path = mra_splits_dir / f"fold_{fold_index}.csv"
        summary_path = fold_path.with_suffix(".summary.json")
        if not fold_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing MRA-MIDAS development fold: {fold_path}")
        rows = pd.read_csv(fold_path, dtype=str, keep_default_na=False)
        missing = MRA_REQUIRED_COLUMNS.difference(rows.columns)
        if missing:
            raise ValueError(
                f"MRA-MIDAS fold_{fold_index} is missing columns: {', '.join(sorted(missing))}"
            )
        if rows.empty:
            raise ValueError(f"MRA-MIDAS fold_{fold_index} must not be empty.")
        if set(rows["source"]) != {"mra_midas"}:
            raise ValueError(f"MRA-MIDAS fold_{fold_index} has an invalid source value.")
        unknown_splits = sorted(set(rows["split"]) - set(SPLIT_ORDER))
        if unknown_splits:
            raise ValueError(
                f"MRA-MIDAS fold_{fold_index} has unknown split values: {unknown_splits}"
            )
        unknown_labels = sorted(set(rows["label"]) - set(PAD_UFES_NATIVE_LABELS))
        if unknown_labels:
            raise ValueError(f"MRA-MIDAS fold_{fold_index} has non-native labels: {unknown_labels}")
        if bool(rows["image_path"].duplicated().any()):
            raise ValueError(f"MRA-MIDAS fold_{fold_index} contains duplicate image paths.")
        _validate_mra_grouping(rows, fold_index=fold_index)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_mra_summary(
            summary,
            rows,
            fold_index=fold_index,
            num_folds=num_folds,
        )
        mapping = {
            str(row.image_path): (
                str(row.unit_id),
                str(row.record_id),
                str(row.distance),
                str(row.label),
            )
            for row in rows.itertuples(index=False)
        }
        if reference_mapping is None:
            reference_mapping = mapping
        elif mapping != reference_mapping:
            raise ValueError(f"MRA-MIDAS fold_{fold_index} image/unit mapping differs from fold_0.")
        fold_rows.append(rows.copy())
        fold_summaries.append(summary)

    assert reference_mapping is not None
    _validate_mra_rotating_coverage(fold_rows, reference_mapping)
    image_rows = _combined_image_rows(pad, reference_mapping)
    fingerprint = _multi_source_fingerprint(pad, fold_rows, fold_summaries)
    return MultiSourceManifests(
        pad=pad,
        mra_fold_rows=tuple(fold_rows),
        mra_fold_summaries=tuple(fold_summaries),
        image_rows=image_rows,
        fingerprint=fingerprint,
    )


def _validate_mra_grouping(rows: pd.DataFrame, *, fold_index: int) -> None:
    for column in ("unit_id", "record_id", "image_path", "distance", "label"):
        if bool(rows[column].astype(str).str.strip().eq("").any()):
            raise ValueError(f"MRA-MIDAS fold_{fold_index} contains blank {column} values.")
    if bool((rows.groupby("unit_id")["split"].nunique() != 1).any()):
        raise ValueError(f"MRA-MIDAS fold_{fold_index} splits a lesion across roles.")
    if bool((rows.groupby("record_id")["split"].nunique() != 1).any()):
        raise ValueError(f"MRA-MIDAS fold_{fold_index} splits a record across roles.")
    if bool((rows.groupby("unit_id")["label"].nunique() != 1).any()):
        raise ValueError(f"MRA-MIDAS fold_{fold_index} gives a lesion multiple labels.")
    distance_sets = rows.groupby("unit_id")["distance"].agg(lambda values: set(values))
    if any(values != set(CLINICAL_DISTANCES) for values in distance_sets):
        raise ValueError(f"MRA-MIDAS fold_{fold_index} has an incomplete paired lesion.")
    coverage = pd.crosstab(rows["split"], rows["label"]).reindex(
        index=SPLIT_ORDER,
        columns=PAD_UFES_NATIVE_LABELS,
        fill_value=0,
    )
    missing = [
        f"{split}/{label}"
        for split in SPLIT_ORDER
        for label in PAD_UFES_NATIVE_LABELS
        if int(coverage.loc[split, label]) == 0
    ]
    if missing:
        raise ValueError(
            f"MRA-MIDAS fold_{fold_index} has missing label coverage: {', '.join(missing)}"
        )


def _validate_mra_summary(
    summary: dict[str, object],
    rows: pd.DataFrame,
    *,
    fold_index: int,
    num_folds: int,
) -> None:
    units = rows[["unit_id", "record_id", "label", "split"]].drop_duplicates()
    expected = {
        "dataset": "mra_midas",
        "role": "authorized_multisource_development",
        "protocol": MRA_PROTOCOL,
        "num_folds": num_folds,
        "fold_index": fold_index,
        "test_outer_fold": fold_index,
        "validation_outer_fold": (fold_index + 1) % num_folds,
        "group_key": "midas_record_id",
        "feature_unit": "paired_distance_lesion_embedding",
        "distance_aggregation": MRA_AGGREGATION,
        "record_overlap_count": 0,
        "lesion_overlap_count": 0,
        "image_count": len(rows),
        "unit_count": len(units),
        "record_count": int(units["record_id"].nunique()),
    }
    mismatches = [
        f"{key}={summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"MRA-MIDAS fold_{fold_index} summary violates the locked protocol: "
            f"{', '.join(mismatches)}"
        )


def _validate_mra_rotating_coverage(
    fold_rows: Sequence[pd.DataFrame],
    reference_mapping: dict[str, tuple[str, str, str, str]],
) -> None:
    image_test_counts: Counter[str] = Counter()
    unit_test_counts: Counter[str] = Counter()
    record_test_counts: Counter[str] = Counter()
    all_units = {values[0] for values in reference_mapping.values()}
    all_records = {values[1] for values in reference_mapping.values()}
    for rows in fold_rows:
        test_rows = rows.loc[rows["split"] == "test"]
        image_test_counts.update(test_rows["image_path"].astype(str))
        unit_test_counts.update(test_rows["unit_id"].astype(str).unique())
        record_test_counts.update(test_rows["record_id"].astype(str).unique())
    if set(image_test_counts) != set(reference_mapping) or any(
        count != 1 for count in image_test_counts.values()
    ):
        raise ValueError("Every MRA-MIDAS image must be outer-fold test data exactly once.")
    if set(unit_test_counts) != all_units or any(count != 1 for count in unit_test_counts.values()):
        raise ValueError("Every MRA-MIDAS lesion must be outer-fold test data exactly once.")
    if set(record_test_counts) != all_records or any(
        count != 1 for count in record_test_counts.values()
    ):
        raise ValueError("Every MRA-MIDAS record must be outer-fold test data exactly once.")


def _combined_image_rows(
    pad: CvManifests,
    mra_mapping: dict[str, tuple[str, str, str, str]],
) -> pd.DataFrame:
    pad_rows = pad.unique_rows.copy()
    pad_rows.insert(0, "source", "pad_ufes")
    pad_rows["unit_id"] = pad_rows["image_path"].astype(str).map(_pad_unit_id)
    pad_rows["distance"] = "single"
    mra_rows = pd.DataFrame(
        [
            {
                "source": "mra_midas",
                "unit_id": unit_id,
                "distance": distance,
                "image_path": path,
                "label": label,
            }
            for path, (unit_id, _record_id, distance, label) in sorted(mra_mapping.items())
        ]
    )
    combined = pd.concat(
        [
            pad_rows[["source", "unit_id", "distance", "image_path", "label"]],
            mra_rows,
        ],
        ignore_index=True,
    ).sort_values(["source", "unit_id", "distance", "image_path"])
    combined = combined.reset_index(drop=True)
    if bool(combined["image_path"].duplicated().any()):
        raise ValueError("PAD-UFES and MRA-MIDAS manifests contain duplicate image paths.")
    return combined


def _pad_unit_id(image_path: str) -> str:
    return "pad-" + hashlib.sha256(str(image_path).encode()).hexdigest()


def _multi_source_fingerprint(
    pad: CvManifests,
    mra_rows: Sequence[pd.DataFrame],
    mra_summaries: Sequence[dict[str, object]],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"schema={CACHE_SCHEMA_VERSION}\npad_protocol={PAD_PROTOCOL}\n"
        f"mra_protocol={MRA_PROTOCOL}\npad_fingerprint={pad.fingerprint}\n".encode()
    )
    for fold_index, (rows, summary) in enumerate(zip(mra_rows, mra_summaries, strict=True)):
        digest.update(
            json.dumps(
                {
                    "fold_index": fold_index,
                    "num_folds": summary.get("num_folds"),
                    "test_outer_fold": summary.get("test_outer_fold"),
                    "validation_outer_fold": summary.get("validation_outer_fold"),
                    "distance_aggregation": summary.get("distance_aggregation"),
                },
                sort_keys=True,
            ).encode()
        )
        for row in rows.sort_values("image_path").itertuples(index=False):
            digest.update(
                f"\n{fold_index}\0{row.split}\0{row.unit_id}\0{row.record_id}\0"
                f"{row.distance}\0{row.image_path}\0{row.label}".encode()
            )
    return digest.hexdigest()


def load_or_extract_image_embeddings(
    manifests: MultiSourceManifests,
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
        return load_image_embedding_cache(
            embeddings_path,
            image_rows=manifests.image_rows,
            model_id=model_id,
            revision=revision,
            manifest_fingerprint=manifests.fingerprint,
        )

    features, processor_metadata = extract_embeddings(
        manifests.image_rows[["image_path", "label"]],
        model_id=model_id,
        revision=revision,
        cache_dir=model_cache_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    payload = build_image_embedding_cache(
        manifests.image_rows,
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
    print(f"Wrote frozen image embeddings: {project_relative(embeddings_path)}")
    return payload


def build_image_embedding_cache(
    image_rows: pd.DataFrame,
    *,
    features: torch.Tensor,
    model_id: str,
    revision: str,
    manifest_fingerprint: str,
    processor_metadata: dict[str, object],
) -> dict[str, object]:
    required = {"source", "unit_id", "distance", "image_path", "label"}
    missing = required.difference(image_rows.columns)
    if missing:
        raise ValueError(
            "Multi-source image manifest is missing columns: " + ", ".join(sorted(missing))
        )
    if bool(image_rows["image_path"].astype(str).duplicated().any()):
        raise ValueError("Multi-source image manifest contains duplicate image paths.")
    features = features.detach().cpu().float().contiguous()
    if features.ndim != 2 or features.shape[0] != len(image_rows):
        raise ValueError("Image embedding tensor shape does not match the multi-source manifest.")
    _validate_normalized_features(features, context="Image embedding cache")
    if processor_metadata.get("encoder_trainable_parameter_count") != 0:
        raise ValueError("Image embedding provenance must record a fully frozen encoder.")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "context": "Frozen MedSigLIP PAD-UFES + MRA-MIDAS embeddings; development evidence only.",
        "model_id": model_id,
        "model_revision": revision,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "known_pad_pretraining_overlap": True,
        "mra_role": "authorized_multisource_development",
        "preprocessing": PREPROCESSING,
        "embedding_normalization": "l2",
        "mra_aggregation": MRA_AGGREGATION,
        "manifest_fingerprint": manifest_fingerprint,
        "image_count": len(image_rows),
        "feature_dim": int(features.shape[1]),
        "image_paths": image_rows["image_path"].astype(str).tolist(),
        "image_labels": image_rows["label"].astype(str).tolist(),
        "image_sources": image_rows["source"].astype(str).tolist(),
        "unit_ids": image_rows["unit_id"].astype(str).tolist(),
        "distances": image_rows["distance"].astype(str).tolist(),
        "features": features,
        "processor": processor_metadata,
    }


def load_image_embedding_cache(
    cache_path: Path,
    *,
    image_rows: pd.DataFrame,
    model_id: str,
    revision: str,
    manifest_fingerprint: str,
) -> dict[str, object]:
    cache_path = resolve_project_path(Path(cache_path))
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Multi-source embedding cache must be a dictionary.")
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": revision,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "known_pad_pretraining_overlap": True,
        "mra_role": "authorized_multisource_development",
        "preprocessing": PREPROCESSING,
        "embedding_normalization": "l2",
        "mra_aggregation": MRA_AGGREGATION,
        "manifest_fingerprint": manifest_fingerprint,
        "image_count": len(image_rows),
        "image_paths": image_rows["image_path"].astype(str).tolist(),
        "image_labels": image_rows["label"].astype(str).tolist(),
        "image_sources": image_rows["source"].astype(str).tolist(),
        "unit_ids": image_rows["unit_id"].astype(str).tolist(),
        "distances": image_rows["distance"].astype(str).tolist(),
    }
    mismatches = [
        f"{key}={payload.get(key)!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Multi-source embedding cache provenance mismatch: " + ", ".join(mismatches)
        )
    features = payload.get("features")
    if not isinstance(features, torch.Tensor):
        raise ValueError("Multi-source embedding cache is missing tensor features.")
    if features.ndim != 2 or features.shape[0] != len(image_rows):
        raise ValueError("Multi-source embedding cache feature shape does not match the manifest.")
    if payload.get("feature_dim") != int(features.shape[1]):
        raise ValueError("Multi-source embedding cache feature_dim does not match its tensor.")
    processor = payload.get("processor")
    if not isinstance(processor, dict):
        raise ValueError("Multi-source embedding cache is missing processor provenance.")
    if processor.get("encoder_trainable_parameter_count") != 0:
        raise ValueError("Multi-source embedding cache does not record a fully frozen encoder.")
    features = features.cpu().float()
    _validate_normalized_features(features, context="Multi-source embedding cache")
    payload["features"] = features
    return payload


def _validate_normalized_features(features: torch.Tensor, *, context: str) -> None:
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{context} contains non-finite features.")
    norms = features.norm(dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)):
        raise ValueError(f"{context} features are not L2 normalized.")


def aggregate_unit_embeddings(
    image_rows: pd.DataFrame,
    image_cache: dict[str, object],
) -> dict[str, object]:
    expected_paths = image_rows["image_path"].astype(str).tolist()
    if image_cache.get("image_paths") != expected_paths:
        raise ValueError("Image cache order differs from the multi-source manifest.")
    image_features = image_cache.get("features")
    if not isinstance(image_features, torch.Tensor):
        raise ValueError("Image cache is missing tensor features.")
    if image_features.ndim != 2 or image_features.shape[0] != len(image_rows):
        raise ValueError("Image cache feature shape differs from the multi-source manifest.")
    unit_records: list[dict[str, str]] = []
    unit_features: list[torch.Tensor] = []

    for row_index, row in image_rows.loc[image_rows["source"] == "pad_ufes"].iterrows():
        unit_records.append(
            {
                "source": "pad_ufes",
                "unit_id": str(row["unit_id"]),
                "label": str(row["label"]),
            }
        )
        unit_features.append(image_features[row_index])

    mra_rows = image_rows.loc[image_rows["source"] == "mra_midas"]
    for unit_id, group in mra_rows.groupby("unit_id", sort=True):
        labels = set(group["label"].astype(str))
        if len(labels) != 1:
            raise ValueError("An MRA-MIDAS embedding unit has multiple labels.")
        distance_features = []
        for distance in CLINICAL_DISTANCES:
            indices = group.index[group["distance"] == distance].tolist()
            if not indices:
                raise ValueError("An MRA-MIDAS embedding unit is missing a paired distance.")
            mean = image_features[indices].mean(dim=0, keepdim=True)
            distance_features.append(functional.normalize(mean, p=2, dim=-1).squeeze(0))
        combined = functional.normalize(
            torch.stack(distance_features).mean(dim=0, keepdim=True),
            p=2,
            dim=-1,
        ).squeeze(0)
        unit_records.append(
            {
                "source": "mra_midas",
                "unit_id": str(unit_id),
                "label": labels.pop(),
            }
        )
        unit_features.append(combined)

    unit_rows = pd.DataFrame(unit_records).sort_values(["source", "unit_id"]).reset_index(drop=True)
    features = torch.stack(unit_features)
    order = pd.DataFrame(unit_records).reset_index().set_index(["source", "unit_id"])["index"]
    indices = [
        int(order.loc[(row.source, row.unit_id)]) for row in unit_rows.itertuples(index=False)
    ]
    features = features.index_select(0, torch.tensor(indices, dtype=torch.long)).contiguous()
    _validate_normalized_features(features, context="Aggregated unit embeddings")
    return {
        "features": features,
        "feature_dim": int(features.shape[1]),
        "source": unit_rows["source"].tolist(),
        "unit_id": unit_rows["unit_id"].tolist(),
        "label": unit_rows["label"].tolist(),
    }


def fold_unit_rows(
    manifests: MultiSourceManifests,
    *,
    fold_index: int,
) -> pd.DataFrame:
    pad_rows = manifests.pad.fold_rows[fold_index][["split", "image_path", "label"]].copy()
    pad_rows.insert(1, "source", "pad_ufes")
    pad_rows["unit_id"] = pad_rows["image_path"].astype(str).map(_pad_unit_id)
    pad_units = pad_rows[["split", "source", "unit_id", "label"]]
    mra_units = manifests.mra_fold_rows[fold_index][
        ["split", "source", "unit_id", "label"]
    ].drop_duplicates()
    rows = pd.concat([pad_units, mra_units], ignore_index=True)
    if bool(rows[["source", "unit_id"]].duplicated().any()):
        raise ValueError(f"fold_{fold_index} contains duplicate multi-source units.")
    for source in SOURCE_ORDER:
        source_rows = rows.loc[rows["source"] == source]
        coverage = pd.crosstab(source_rows["split"], source_rows["label"]).reindex(
            index=SPLIT_ORDER,
            columns=PAD_UFES_NATIVE_LABELS,
            fill_value=0,
        )
        if bool((coverage == 0).any().any()):
            raise ValueError(f"fold_{fold_index} has incomplete {source} source/class coverage.")
    return rows.sort_values(["split", "source", "label", "unit_id"]).reset_index(drop=True)


def source_class_weights(rows: pd.DataFrame) -> torch.Tensor:
    if rows.empty:
        raise ValueError("Cannot weight an empty multi-source split.")
    counts = Counter(zip(rows["source"].astype(str), rows["label"].astype(str), strict=True))
    expected_cells = {
        (source, label) for source in SOURCE_ORDER for label in PAD_UFES_NATIVE_LABELS
    }
    if set(counts) != expected_cells:
        missing = sorted(expected_cells - set(counts))
        unknown = sorted(set(counts) - expected_cells)
        raise ValueError(
            f"Source-class weighting cells differ: missing={missing}, unknown={unknown}"
        )
    unit_count = len(rows)
    cell_count = len(expected_cells)
    weights = torch.tensor(
        [
            unit_count / (cell_count * counts[(str(row.source), str(row.label))])
            for row in rows.itertuples(index=False)
        ],
        dtype=torch.float32,
    )
    if not math.isclose(float(weights.sum()), float(unit_count), rel_tol=1e-5, abs_tol=1e-4):
        raise RuntimeError("Source-class weights must have mean one.")
    return weights


class MultiSourceEmbeddingDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, unit_cache: dict[str, object]) -> None:
        cache_keys = list(zip(unit_cache["source"], unit_cache["unit_id"], strict=True))
        index_by_key = {key: index for index, key in enumerate(cache_keys)}
        label_by_key = dict(zip(cache_keys, unit_cache["label"], strict=True))
        if len(index_by_key) != len(cache_keys):
            raise ValueError("Aggregated unit cache contains duplicate source/unit keys.")

        indices = []
        for row in rows.itertuples(index=False):
            key = (str(row.source), str(row.unit_id))
            if key not in index_by_key:
                raise ValueError(f"Aggregated unit cache is missing a requested {row.source} unit.")
            if str(label_by_key[key]) != str(row.label):
                raise ValueError(f"Aggregated unit cache label differs for a {row.source} unit.")
            indices.append(index_by_key[key])
        index_tensor = torch.tensor(indices, dtype=torch.long)
        label_to_index = {label: index for index, label in enumerate(PAD_UFES_NATIVE_LABELS)}
        source_to_index = {source: index for index, source in enumerate(SOURCE_ORDER)}
        self.features = unit_cache["features"].index_select(0, index_tensor)
        self.targets = torch.tensor(
            [label_to_index[str(label)] for label in rows["label"]], dtype=torch.long
        )
        self.weights = source_class_weights(rows)
        self.sources = torch.tensor(
            [source_to_index[str(source)] for source in rows["source"]], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float, int]:
        return (
            self.features[index],
            int(self.targets[index]),
            float(self.weights[index]),
            int(self.sources[index]),
        )


def run_probe_epoch(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    head.train(training)
    size = len(PAD_UFES_NATIVE_LABELS)
    combined = [[0 for _ in range(size)] for _ in range(size)]
    by_source = {source: [[0 for _ in range(size)] for _ in range(size)] for source in SOURCE_ORDER}
    weighted_loss_sum = 0.0
    unit_count = 0
    with torch.set_grad_enabled(training):
        for features, targets, weights, source_indices in loader:
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            losses = functional.cross_entropy(logits, targets, reduction="none")
            loss = (losses * weights).mean()
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            weighted_loss_sum += float((losses * weights).sum().item())
            unit_count += int(targets.numel())
            for truth, prediction, source_index in zip(
                targets.cpu(), predictions.cpu(), source_indices, strict=True
            ):
                truth_index = int(truth)
                prediction_index = int(prediction)
                source = SOURCE_ORDER[int(source_index)]
                combined[truth_index][prediction_index] += 1
                by_source[source][truth_index][prediction_index] += 1
    if unit_count == 0:
        raise ValueError("A multi-source probe loader produced no units.")
    combined_metrics = _metrics_from_confusion(combined)
    source_metrics = {
        source: _metrics_from_confusion(confusion) for source, confusion in by_source.items()
    }
    source_macro_f1 = statistics.fmean(
        float(source_metrics[source]["macro_f1"]) for source in SOURCE_ORDER
    )
    return {
        "loss": weighted_loss_sum / unit_count,
        **combined_metrics,
        "by_source": source_metrics,
        "source_mean_macro_f1": source_macro_f1,
        "worst_source_macro_f1": min(
            float(source_metrics[source]["macro_f1"]) for source in SOURCE_ORDER
        ),
    }


def train_probe_fold(
    rows: pd.DataFrame,
    *,
    pad_summary: dict[str, object],
    mra_summary: dict[str, object],
    unit_cache: dict[str, object],
    image_cache: dict[str, object],
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
    split_rows = {split: rows.loc[rows["split"] == split].copy() for split in SPLIT_ORDER}
    datasets = {
        split: MultiSourceEmbeddingDataset(split_rows[split], unit_cache) for split in SPLIT_ORDER
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
    feature_dim = int(unit_cache["feature_dim"])
    head = nn.Linear(feature_dim, len(PAD_UFES_NATIVE_LABELS)).to(device)
    trainable_parameter_count = sum(parameter.numel() for parameter in head.parameters())
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    checkpoint_path = resolve_project_path(Path(checkpoint_path))
    run_dir = resolve_project_path(Path(run_dir))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_source_macro_f1 = -1.0
    best_val_loss = math.inf
    best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_probe_epoch(head, loaders["train"], device, optimizer)
        val_metrics = run_probe_epoch(head, loaders["val"], device)
        history.append(
            {
                "epoch": epoch,
                "train": _compact_source_metrics(train_metrics),
                "val": _compact_source_metrics(val_metrics),
            }
        )
        val_source_macro_f1 = float(val_metrics["source_mean_macro_f1"])
        val_loss = float(val_metrics["loss"])
        improved = val_source_macro_f1 > best_source_macro_f1 or (
            math.isclose(val_source_macro_f1, best_source_macro_f1, abs_tol=1e-12)
            and val_loss < best_val_loss
        )
        if improved:
            best_source_macro_f1 = val_source_macro_f1
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": head.state_dict(),
                    "architecture": ARCHITECTURE,
                    "model_id": model_id,
                    "model_revision": revision,
                    "encoder_frozen": True,
                    "feature_dim": feature_dim,
                    "labels": list(PAD_UFES_NATIVE_LABELS),
                    "sources": list(SOURCE_ORDER),
                    "epoch": epoch,
                    "seed": seed,
                    "selection_metric": SELECTION_METRIC,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics = {
        split: run_probe_epoch(head, loaders[split], device) for split in SPLIT_ORDER
    }
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "source_class_weighting": WEIGHTING,
        "optimizer": "AdamW",
        "schedule": "none",
    }
    report: dict[str, object] = {
        "context": (
            "Experimental frozen MedSigLIP PAD-UFES + MRA-MIDAS development probe; "
            "not medical certainty or independent validation."
        ),
        "architecture": ARCHITECTURE,
        "model_id": model_id,
        "model_revision": revision,
        "encoder_frozen": True,
        "known_pad_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "mra_role": "authorized_multisource_development",
        "embedding_normalization": "l2",
        "preprocessing": PREPROCESSING,
        "mra_aggregation": MRA_AGGREGATION,
        "manifest_fingerprint": image_cache.get("manifest_fingerprint"),
        "feature_dim": feature_dim,
        "trainable_parameter_count": trainable_parameter_count,
        "encoder_parameter_count": image_cache.get("processor", {}).get("encoder_parameter_count"),
        "encoder_trainable_parameter_count": image_cache.get("processor", {}).get(
            "encoder_trainable_parameter_count"
        ),
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": seed,
        "pad_split_summary": pad_summary,
        "mra_split_summary": mra_summary,
        "source_unit_counts": {
            source: int((rows["source"] == source).sum()) for source in SOURCE_ORDER
        },
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "dataset_sizes_by_source": {
            split: {
                source: int((split_rows[split]["source"] == source).sum())
                for source in SOURCE_ORDER
            }
            for split in SPLIT_ORDER
        },
        "hyperparameters": hyperparameters,
        "best_epoch": best_epoch,
        "selection_metric": SELECTION_METRIC,
        "best_val_source_mean_macro_f1": best_source_macro_f1,
        "history": history,
        "selected_train": selected_metrics["train"],
        "selected_val": selected_metrics["val"],
        "test": selected_metrics["test"],
        "caveat": (
            "PAD-UFES-20 is known MedSigLIP pretraining data and MRA-MIDAS is trained on here; "
            "these are development results, not independent external evidence."
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _compact_source_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "loss": float(metrics["loss"]),
        "macro_f1": float(metrics["macro_f1"]),
        "source_mean_macro_f1": float(metrics["source_mean_macro_f1"]),
        "worst_source_macro_f1": float(metrics["worst_source_macro_f1"]),
        "by_source_macro_f1": {
            source: float(metrics["by_source"][source]["macro_f1"]) for source in SOURCE_ORDER
        },
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
    reports = []
    for fold_index in range(num_folds):
        report_path = reports_root / f"fold_{fold_index}" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing multi-source probe report: {report_path}")
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

    consistency_keys = (
        "manifest_fingerprint",
        "feature_dim",
        "trainable_parameter_count",
        "encoder_parameter_count",
        "encoder_trainable_parameter_count",
        "known_pretraining_datasets",
    )
    disagreements = [
        key
        for key in consistency_keys
        if len({json.dumps(report.get(key), sort_keys=True) for report in reports}) != 1
    ]
    if disagreements:
        raise ValueError(
            "Multi-source reports disagree on locked provenance: " + ", ".join(disagreements)
        )

    source_unit_counts = {
        source: {int(report["source_unit_counts"][source]) for report in reports}
        for source in SOURCE_ORDER
    }
    if any(len(counts) != 1 for counts in source_unit_counts.values()):
        raise ValueError("Multi-source reports disagree on source unit counts.")
    expected_source_counts = {source: counts.pop() for source, counts in source_unit_counts.items()}
    pooled_confusions = {
        source: _pooled_source_confusion(reports, source) for source in SOURCE_ORDER
    }
    for source in SOURCE_ORDER:
        support = sum(sum(row) for row in pooled_confusions[source])
        if support != expected_source_counts[source]:
            raise ValueError(f"Outer-test folds do not cover {source} units exactly once.")
    pooled_by_source = {
        source: _metrics_from_confusion(confusion)
        for source, confusion in pooled_confusions.items()
    }
    pooled_combined_confusion = _add_confusions(list(pooled_confusions.values()))
    pooled_combined = _metrics_from_confusion(pooled_combined_confusion)
    pooled_source_mean_macro_f1 = statistics.fmean(
        float(pooled_by_source[source]["macro_f1"]) for source in SOURCE_ORDER
    )
    pooled_worst_source_macro_f1 = min(
        float(pooled_by_source[source]["macro_f1"]) for source in SOURCE_ORDER
    )
    gaps = [
        float(report["selected_train"]["source_mean_macro_f1"])
        - float(report["selected_val"]["source_mean_macro_f1"])
        for report in reports
    ]
    fold_source_mean = [float(report["test"]["source_mean_macro_f1"]) for report in reports]
    fold_by_source = {
        source: _distribution(
            [float(report["test"]["by_source"][source]["macro_f1"]) for report in reports]
        )
        for source in SOURCE_ORDER
    }
    rules = {
        "mean_selected_train_val_source_mean_macro_f1_gap_lte_0_1500": (
            statistics.fmean(gaps) <= 0.15
        ),
        "pooled_pad_macro_f1_gte_0_6393": (
            float(pooled_by_source["pad_ufes"]["macro_f1"]) >= 0.6393
        ),
        "pooled_mra_macro_f1_gte_0_4000": (float(pooled_by_source["mra_midas"]["macro_f1"]) >= 0.4),
        "pooled_source_mean_macro_f1_gte_0_5200": pooled_source_mean_macro_f1 >= 0.52,
        "pooled_worst_source_macro_f1_gte_0_4000": pooled_worst_source_macro_f1 >= 0.4,
        "pooled_pad_scc_f1_gte_0_3000": (
            float(pooled_by_source["pad_ufes"]["per_class"]["squamous_cell_carcinoma"]["f1"]) >= 0.3
        ),
        "pooled_mra_melanoma_f1_gte_0_3500": (
            float(pooled_by_source["mra_midas"]["per_class"]["melanoma"]["f1"]) >= 0.35
        ),
        "pooled_mra_scc_f1_gte_0_2000": (
            float(pooled_by_source["mra_midas"]["per_class"]["squamous_cell_carcinoma"]["f1"])
            >= 0.2
        ),
    }
    summary: dict[str, object] = {
        "context": (
            "Experimental frozen MedSigLIP PAD-UFES + MRA-MIDAS grouped development CV; "
            "not medical certainty or independent validation."
        ),
        "pad_protocol": PAD_PROTOCOL,
        "mra_protocol": MRA_PROTOCOL,
        "num_folds": num_folds,
        "seed": seed,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "architecture": ARCHITECTURE,
        "model_id": model_id,
        "model_revision": revision,
        "encoder_frozen": True,
        "known_pad_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "mra_role": "authorized_multisource_development",
        "embedding_normalization": "l2",
        "preprocessing": PREPROCESSING,
        "mra_aggregation": MRA_AGGREGATION,
        "source_class_weighting": WEIGHTING,
        "selection_metric": SELECTION_METRIC,
        "encoder_trainable_parameter_count": 0,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "source_unit_counts": expected_source_counts,
        "fold_source_mean_macro_f1": _distribution(fold_source_mean),
        "fold_macro_f1_by_source": fold_by_source,
        "selected_train_val_source_mean_macro_f1_gap": _distribution(gaps),
        "pooled_by_source": pooled_by_source,
        "pooled_combined_secondary": pooled_combined,
        "pooled_source_mean_macro_f1": pooled_source_mean_macro_f1,
        "pooled_worst_source_macro_f1": pooled_worst_source_macro_f1,
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
            }
            for fold_index, report in enumerate(reports)
        ],
        "caveat": (
            "PAD-UFES-20 is known MedSigLIP pretraining data and MRA-MIDAS is training data in "
            "this experiment. Passing rules cannot support external robustness, fairness, "
            "patient-self-photo, deployment, diagnosis, or medical-readiness claims."
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
        "known_pad_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "mra_role": "authorized_multisource_development",
        "embedding_normalization": "l2",
        "preprocessing": PREPROCESSING,
        "mra_aggregation": MRA_AGGREGATION,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": seed,
        "selection_metric": SELECTION_METRIC,
        "encoder_trainable_parameter_count": 0,
    }
    mismatches = [
        f"{key}={report.get(key)!r}" for key, value in expected.items() if report.get(key) != value
    ]
    for summary_name, protocol in (
        ("pad_split_summary", PAD_PROTOCOL),
        ("mra_split_summary", MRA_PROTOCOL),
    ):
        split_summary = report.get(summary_name)
        if not isinstance(split_summary, dict):
            mismatches.append(f"{summary_name} is missing")
            continue
        split_expected = {
            "protocol": protocol,
            "num_folds": num_folds,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": (fold_index + 1) % num_folds,
        }
        mismatches.extend(
            f"{summary_name}.{key}={split_summary.get(key)!r}"
            for key, value in split_expected.items()
            if split_summary.get(key) != value
        )
    hyperparameters = report.get("hyperparameters")
    hyperparameter_expected = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "source_class_weighting": WEIGHTING,
        "optimizer": "AdamW",
        "schedule": "none",
    }
    if not isinstance(hyperparameters, dict):
        mismatches.append("hyperparameters are missing")
    else:
        mismatches.extend(
            f"hyperparameters.{key}={hyperparameters.get(key)!r}"
            for key, value in hyperparameter_expected.items()
            if hyperparameters.get(key) != value
        )
    if report.get("encoder_trainable_parameter_count") != 0:
        mismatches.append(
            f"encoder_trainable_parameter_count={report.get('encoder_trainable_parameter_count')!r}"
        )
    feature_dim = report.get("feature_dim")
    trainable_count = report.get("trainable_parameter_count")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        mismatches.append(f"feature_dim={feature_dim!r}")
    elif trainable_count != (feature_dim + 1) * len(PAD_UFES_NATIVE_LABELS):
        mismatches.append(f"trainable_parameter_count={trainable_count!r}")
    fingerprint = report.get("manifest_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        mismatches.append(f"manifest_fingerprint={fingerprint!r}")
    if mismatches:
        raise ValueError(
            f"Multi-source probe report fold_{fold_index} violates the locked protocol: "
            f"{', '.join(mismatches)}"
        )


def _pooled_source_confusion(
    reports: Sequence[dict[str, object]],
    source: str,
) -> list[list[int]]:
    confusions = [report["test"]["by_source"][source]["confusion_matrix"] for report in reports]
    return _add_confusions(confusions)


def _add_confusions(confusions: Sequence[list[list[int]]]) -> list[list[int]]:
    size = len(PAD_UFES_NATIVE_LABELS)
    pooled = [[0 for _ in range(size)] for _ in range(size)]
    for confusion in confusions:
        if len(confusion) != size or any(len(row) != size for row in confusion):
            raise ValueError("A multi-source confusion matrix has an invalid shape.")
        for row_index in range(size):
            for column_index in range(size):
                pooled[row_index][column_index] += int(confusion[row_index][column_index])
    return pooled


def _metrics_from_confusion(confusion: list[list[int]]) -> dict[str, object]:
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


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence.")
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
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
    manifests = load_multi_source_manifests(
        args.pad_splits_dir,
        args.mra_splits_dir,
        num_folds=args.folds,
    )
    print(
        f"Validated {args.folds} PAD/MRA rotating folds over "
        f"{len(manifests.image_rows):,} source images; fingerprint={manifests.fingerprint}"
    )
    image_cache = load_or_extract_image_embeddings(
        manifests,
        embeddings_path=args.embeddings,
        model_id=args.model_id,
        revision=args.revision,
        model_cache_dir=args.model_cache_dir,
        device=device,
        batch_size=args.embedding_batch_size,
        num_workers=args.num_workers,
    )
    if image_cache.get("processor", {}).get("encoder_trainable_parameter_count") != 0:
        raise ValueError("Embedding cache records trainable encoder parameters.")
    unit_cache = aggregate_unit_embeddings(manifests.image_rows, image_cache)
    runs_root = resolve_project_path(args.runs_root)
    checkpoints_dir = resolve_project_path(args.checkpoints_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for fold_index in range(args.folds):
        run_dir = runs_root / f"fold_{fold_index}"
        report_path = run_dir / "report.json"
        if args.resume and report_path.exists():
            print(f"fold_{fold_index}: report exists; skipping")
            continue
        rows = fold_unit_rows(manifests, fold_index=fold_index)
        print(f"fold_{fold_index}: starting frozen multi-source linear probe")
        report = train_probe_fold(
            rows,
            pad_summary=manifests.pad.fold_summaries[fold_index],
            mra_summary=manifests.mra_fold_summaries[fold_index],
            unit_cache=unit_cache,
            image_cache=image_cache,
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
            f"test_source_mean_macro_f1={report['test']['source_mean_macro_f1']:.4f}"
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
        "Multi-source probe complete: "
        f"pooled_source_mean_macro_f1={summary['pooled_source_mean_macro_f1']:.4f} "
        f"pooled_worst_source_macro_f1={summary['pooled_worst_source_macro_f1']:.4f} "
        f"all_rules_pass={summary['decision_rules']['all_pass']}"
    )
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

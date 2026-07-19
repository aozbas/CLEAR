"""Prepare deterministic patient-grouped HIBA development folds.

HIBA has already been scored by CLEAR and is development data in this workflow, not an untouched
holdout. Generated manifests contain public dataset identifiers but remain ignored/private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from ml.evaluation.hiba import (
    DEFAULT_DATA_DIR,
    load_source,
    prepare_primary_cohort,
    resolve_project_path,
    validate_frozen_audit,
)
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_pad_ufes import SPLIT_ORDER, _nested_counts, project_relative
from ml.training.prepare_pad_ufes_cv import DEFAULT_FOLDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "hiba_multisource_cv"
DEFAULT_SEED = 42
PROTOCOL = "hiba_patient_grouped_rotating_development_cv"
OUTPUT_COLUMNS = (
    "split",
    "source",
    "image_path",
    "label",
    "patient_id",
    "lesion_id",
    "isic_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare rotating patient-grouped HIBA six-class development folds."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def assign_patient_folds(
    rows: pd.DataFrame,
    *,
    labels: tuple[str, ...] = PAD_UFES_NATIVE_LABELS,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    if num_folds < 3:
        raise ValueError("num_folds must be at least 3 for rotating train/val/test roles.")
    required = {"patient_id", "label"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"HIBA fold rows are missing columns: {', '.join(sorted(missing))}")
    if rows.empty:
        raise ValueError("HIBA fold rows must not be empty.")
    if bool((rows["patient_id"].astype(str).str.strip() == "").any()):
        raise ValueError("HIBA fold rows contain a blank patient_id.")
    unknown_labels = sorted(set(rows["label"]) - set(labels))
    if unknown_labels:
        raise ValueError(f"HIBA fold rows contain non-native labels: {unknown_labels}")

    patient_labels = {
        str(patient_id): frozenset(group["label"].tolist())
        for patient_id, group in rows.groupby("patient_id", sort=True)
    }
    patients_by_label = {
        label: [patient_id for patient_id, values in patient_labels.items() if label in values]
        for label in labels
    }
    for label, patient_ids in patients_by_label.items():
        if len(patient_ids) < num_folds:
            raise ValueError(
                f"Not enough HIBA patient groups for label={label!r} across {num_folds} folds; "
                f"found {len(patient_ids)}."
            )

    label_frequency = {label: len(patient_ids) for label, patient_ids in patients_by_label.items()}
    stratification_labels = {
        patient_id: min(values, key=lambda label: (label_frequency[label], label))
        for patient_id, values in patient_labels.items()
    }
    rng = random.Random(seed)
    assignments: dict[str, int] = {}
    for label in labels:
        patient_ids = sorted(
            patient_id
            for patient_id, stratification_label in stratification_labels.items()
            if stratification_label == label
        )
        rng.shuffle(patient_ids)
        fold_order = list(range(num_folds))
        rng.shuffle(fold_order)
        for index, patient_id in enumerate(patient_ids):
            assignments[patient_id] = fold_order[index % num_folds]

    if len(assignments) != len(patient_labels):
        raise ValueError("Some HIBA patients were not assigned to an outer fold.")
    _validate_outer_fold_coverage(rows, assignments, labels=labels, num_folds=num_folds)
    return assignments


def prepare_cross_validation(
    data_dir: Path,
    out_dir: Path,
    *,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    metadata_rows, image_paths, source_hashes = load_source(resolve_project_path(data_dir))
    prepared = prepare_primary_cohort(
        metadata_rows,
        image_paths,
        source_hashes=source_hashes,
    )
    validate_frozen_audit(prepared.audit)
    return write_cross_validation_rows(
        prepared.image_rows,
        out_dir,
        audit=prepared.audit,
        num_folds=num_folds,
        seed=seed,
    )


def write_cross_validation_rows(
    image_rows: pd.DataFrame,
    out_dir: Path,
    *,
    audit: dict[str, object],
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    required = {"image_path", "label", "patient_id", "lesion_id", "isic_id"}
    missing = required.difference(image_rows.columns)
    if missing:
        raise ValueError(f"HIBA image rows are missing columns: {', '.join(sorted(missing))}")
    rows = image_rows.copy()
    if bool(rows["image_path"].astype(str).duplicated().any()):
        raise ValueError("HIBA image rows contain duplicate image paths.")
    if bool(rows["isic_id"].astype(str).duplicated().any()):
        raise ValueError("HIBA image rows contain duplicate isic_id values.")
    if bool((rows[["patient_id", "lesion_id", "isic_id"]].astype(str) == "").any().any()):
        raise ValueError("HIBA image rows contain blank identifiers.")
    if set(rows["label"]) != set(PAD_UFES_NATIVE_LABELS):
        raise ValueError("HIBA image rows must contain all six exact native labels.")

    assignments = assign_patient_folds(rows, num_folds=num_folds, seed=seed)
    rows["outer_fold"] = rows["patient_id"].astype(str).map(assignments)
    rows["source"] = "hiba"
    out_dir = resolve_project_path(Path(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_fingerprint = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fold_paths = []
    for fold_index in range(num_folds):
        validation_fold = (fold_index + 1) % num_folds
        fold_rows = rows.copy()
        fold_rows["split"] = fold_rows["outer_fold"].apply(
            lambda outer_fold, test_fold=fold_index, val_fold=validation_fold: _role_for_outer_fold(
                int(outer_fold),
                test_fold=test_fold,
                validation_fold=val_fold,
            )
        )
        _validate_fold(fold_rows)
        output = _sorted_output(fold_rows)
        fold_path = out_dir / f"fold_{fold_index}.csv"
        output.to_csv(fold_path, index=False)
        summary = _build_fold_summary(
            fold_rows,
            fold_index=fold_index,
            validation_fold=validation_fold,
            num_folds=num_folds,
            seed=seed,
            audit_fingerprint=audit_fingerprint,
        )
        fold_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fold_paths.append(fold_path)

    cv_summary = _build_cv_summary(
        rows,
        num_folds=num_folds,
        seed=seed,
        audit_fingerprint=audit_fingerprint,
    )
    (out_dir / "cv.summary.json").write_text(
        json.dumps(cv_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return fold_paths


def _role_for_outer_fold(outer_fold: int, *, test_fold: int, validation_fold: int) -> str:
    if outer_fold == test_fold:
        return "test"
    if outer_fold == validation_fold:
        return "val"
    return "train"


def _validate_outer_fold_coverage(
    rows: pd.DataFrame,
    assignments: dict[str, int],
    *,
    labels: tuple[str, ...],
    num_folds: int,
) -> None:
    patient_labels = rows[["patient_id", "label"]].drop_duplicates().copy()
    patient_labels["outer_fold"] = patient_labels["patient_id"].astype(str).map(assignments)
    coverage = pd.crosstab(patient_labels["outer_fold"], patient_labels["label"]).reindex(
        index=range(num_folds),
        columns=labels,
        fill_value=0,
    )
    missing = [
        f"fold_{fold}/{label}"
        for fold in range(num_folds)
        for label in labels
        if int(coverage.loc[fold, label]) == 0
    ]
    if missing:
        raise ValueError(f"Outer HIBA folds have missing label coverage: {', '.join(missing)}")


def _validate_fold(rows: pd.DataFrame) -> None:
    if bool((rows.groupby("patient_id")["split"].nunique() != 1).any()):
        raise ValueError("A HIBA patient appears in multiple fold roles.")
    if bool((rows.groupby("lesion_id")["split"].nunique() != 1).any()):
        raise ValueError("A HIBA lesion appears in multiple fold roles.")
    if bool((rows.groupby("lesion_id")["patient_id"].nunique() != 1).any()):
        raise ValueError("A HIBA lesion maps to multiple patients.")
    if bool((rows.groupby("lesion_id")["label"].nunique() != 1).any()):
        raise ValueError("A HIBA lesion maps to multiple labels.")
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
        raise ValueError(f"HIBA fold has missing label coverage: {', '.join(missing)}")


def _sorted_output(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows[list(OUTPUT_COLUMNS)].copy()
    output["_split_order"] = output["split"].map(
        {split: index for index, split in enumerate(SPLIT_ORDER)}
    )
    return (
        output.sort_values(["_split_order", "label", "patient_id", "lesion_id", "isic_id"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )


def _build_fold_summary(
    rows: pd.DataFrame,
    *,
    fold_index: int,
    validation_fold: int,
    num_folds: int,
    seed: int,
    audit_fingerprint: str,
) -> dict[str, object]:
    lesion_rows = rows[["lesion_id", "patient_id", "split", "label"]].drop_duplicates()
    patient_rows = rows[["patient_id", "split", "label"]].drop_duplicates()
    image_counts = pd.crosstab(rows["split"], rows["label"]).reindex(
        index=SPLIT_ORDER,
        columns=PAD_UFES_NATIVE_LABELS,
        fill_value=0,
    )
    lesion_counts = pd.crosstab(lesion_rows["split"], lesion_rows["label"]).reindex(
        index=SPLIT_ORDER,
        columns=PAD_UFES_NATIVE_LABELS,
        fill_value=0,
    )
    return {
        "dataset": "hiba",
        "dataset_role": "multisource_development",
        "protocol": PROTOCOL,
        "num_folds": num_folds,
        "fold_index": fold_index,
        "test_outer_fold": fold_index,
        "validation_outer_fold": validation_fold,
        "training_outer_folds": [
            fold for fold in range(num_folds) if fold not in {fold_index, validation_fold}
        ],
        "seed": seed,
        "group_key": "patient_id",
        "audit_fingerprint": audit_fingerprint,
        "patient_overlap_count": int((rows.groupby("patient_id")["split"].nunique() > 1).sum()),
        "lesion_overlap_count": int((rows.groupby("lesion_id")["split"].nunique() > 1).sum()),
        "image_count": len(rows),
        "cv_total_image_count": len(rows),
        "lesion_count": int(rows["lesion_id"].nunique()),
        "cv_total_lesion_count": int(rows["lesion_id"].nunique()),
        "patient_count": int(rows["patient_id"].nunique()),
        "images_by_split": {split: int((rows["split"] == split).sum()) for split in SPLIT_ORDER},
        "lesions_by_split": {
            split: int(rows.loc[rows["split"] == split, "lesion_id"].nunique())
            for split in SPLIT_ORDER
        },
        "patients_by_split": {
            split: int(rows.loc[rows["split"] == split, "patient_id"].nunique())
            for split in SPLIT_ORDER
        },
        "images_by_split_and_label": _nested_counts(image_counts),
        "lesions_by_split_and_label": _nested_counts(lesion_counts),
        "patient_label_row_count": len(patient_rows),
    }


def _build_cv_summary(
    rows: pd.DataFrame,
    *,
    num_folds: int,
    seed: int,
    audit_fingerprint: str,
) -> dict[str, object]:
    patient_outer_fold_counts = rows.groupby("patient_id")["outer_fold"].nunique()
    lesion_outer_fold_counts = rows.groupby("lesion_id")["outer_fold"].nunique()
    return {
        "dataset": "hiba",
        "dataset_role": "multisource_development",
        "protocol": PROTOCOL,
        "num_folds": num_folds,
        "seed": seed,
        "group_key": "patient_id",
        "audit_fingerprint": audit_fingerprint,
        "image_count": len(rows),
        "lesion_count": int(rows["lesion_id"].nunique()),
        "patient_count": int(rows["patient_id"].nunique()),
        "patient_outer_fold_overlap_count": int((patient_outer_fold_counts > 1).sum()),
        "lesion_outer_fold_overlap_count": int((lesion_outer_fold_counts > 1).sum()),
        "patients_assigned_once": bool((patient_outer_fold_counts == 1).all()),
        "lesions_assigned_once": bool((lesion_outer_fold_counts == 1).all()),
        "each_image_lesion_and_patient_is_test_once": True,
        "images_by_outer_fold": {
            str(fold): int((rows["outer_fold"] == fold).sum()) for fold in range(num_folds)
        },
        "lesions_by_outer_fold": {
            str(fold): int(rows.loc[rows["outer_fold"] == fold, "lesion_id"].nunique())
            for fold in range(num_folds)
        },
        "patients_by_outer_fold": {
            str(fold): int(rows.loc[rows["outer_fold"] == fold, "patient_id"].nunique())
            for fold in range(num_folds)
        },
    }


def main() -> None:
    args = parse_args()
    fold_paths = prepare_cross_validation(
        args.data_dir,
        args.out_dir,
        num_folds=args.folds,
        seed=args.seed,
    )
    print(
        f"Wrote {len(fold_paths)} HIBA patient-grouped development folds to "
        f"{project_relative(resolve_project_path(args.out_dir))}"
    )


if __name__ == "__main__":
    main()

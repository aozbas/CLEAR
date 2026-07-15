"""Prepare deterministic patient-grouped PAD-UFES cross-validation folds."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_pad_ufes import (
    DEFAULT_RAW_DIR,
    DEFAULT_SEED,
    SPLIT_ORDER,
    _nested_counts,
    _validate_grouped_splits,
    load_prepared_rows,
    project_relative,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes_native_cv"
DEFAULT_FOLDS = 5
PROTOCOL = "patient_grouped_rotating_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare rotating patient-grouped PAD-UFES-native cross-validation folds."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
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
                f"Not enough patient groups for label={label!r} across {num_folds} folds; "
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
        raise ValueError("Some PAD-UFES patients were not assigned to an outer fold.")
    _validate_outer_fold_coverage(rows, assignments, labels=labels, num_folds=num_folds)
    return assignments


def prepare_cross_validation(
    raw_dir: Path,
    out_dir: Path,
    *,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    prepared, excluded_counts, labels = load_prepared_rows(raw_dir, label_mode="native")
    assignments = assign_patient_folds(
        prepared,
        labels=labels,
        num_folds=num_folds,
        seed=seed,
    )
    prepared["outer_fold"] = prepared["patient_id"].map(assignments)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_paths: list[Path] = []
    for fold_index in range(num_folds):
        fold_rows = prepared.copy()
        validation_fold = (fold_index + 1) % num_folds
        fold_rows["split"] = fold_rows["outer_fold"].apply(
            lambda outer_fold, test_fold=fold_index, val_fold=validation_fold: _role_for_outer_fold(
                int(outer_fold),
                test_fold=test_fold,
                validation_fold=val_fold,
            )
        )
        _validate_grouped_splits(fold_rows, labels=labels)
        output = _sorted_output(fold_rows)
        fold_path = out_dir / f"fold_{fold_index}.csv"
        output.to_csv(fold_path, index=False)
        fold_path.with_suffix(".summary.json").write_text(
            json.dumps(
                _build_fold_summary(
                    fold_rows,
                    fold_index=fold_index,
                    validation_fold=validation_fold,
                    num_folds=num_folds,
                    seed=seed,
                    labels=labels,
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        fold_paths.append(fold_path)

    (out_dir / "cv.summary.json").write_text(
        json.dumps(
            _build_cv_summary(
                prepared,
                num_folds=num_folds,
                seed=seed,
                labels=labels,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "cv.excluded.json").write_text(
        json.dumps(excluded_counts, indent=2, sort_keys=True),
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
    patient_labels["outer_fold"] = patient_labels["patient_id"].map(assignments)
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
        raise ValueError(f"Outer PAD-UFES folds have missing label coverage: {', '.join(missing)}")


def _sorted_output(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows[["split", "image_path", "label"]].copy()
    output["_split_order"] = output["split"].map(
        {split: index for index, split in enumerate(SPLIT_ORDER)}
    )
    return (
        output.sort_values(["_split_order", "label", "image_path"])
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
    labels: tuple[str, ...],
) -> dict[str, object]:
    patient_labels = rows[["patient_id", "split", "label"]].drop_duplicates()
    image_counts = pd.crosstab(rows["split"], rows["label"]).reindex(
        index=SPLIT_ORDER,
        columns=labels,
        fill_value=0,
    )
    patient_counts = pd.crosstab(patient_labels["split"], patient_labels["label"]).reindex(
        index=SPLIT_ORDER,
        columns=labels,
        fill_value=0,
    )
    return {
        "dataset": "pad_ufes",
        "label_mode": "native",
        "split_strategy": "patient",
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
        "patient_overlap_count": int((rows.groupby("patient_id")["split"].nunique() > 1).sum()),
        "patient_lesion_overlap_count": int(
            (rows.groupby(["patient_id", "lesion_id"])["split"].nunique() > 1).sum()
        ),
        "image_count": len(rows),
        "cv_total_image_count": len(rows),
        "patient_count": int(rows["patient_id"].nunique()),
        "patient_lesion_count": int(rows[["patient_id", "lesion_id"]].drop_duplicates().shape[0]),
        "images_by_split": {split: int((rows["split"] == split).sum()) for split in SPLIT_ORDER},
        "patients_by_split": {
            split: int(rows.loc[rows["split"] == split, "patient_id"].nunique())
            for split in SPLIT_ORDER
        },
        "images_by_split_and_label": _nested_counts(image_counts),
        "patients_by_split_and_label": _nested_counts(patient_counts),
    }


def _build_cv_summary(
    rows: pd.DataFrame,
    *,
    num_folds: int,
    seed: int,
    labels: tuple[str, ...],
) -> dict[str, object]:
    patient_labels = rows[["patient_id", "outer_fold", "label"]].drop_duplicates()
    patient_outer_fold_counts = rows.groupby("patient_id")["outer_fold"].nunique()
    patient_outer_fold_overlap_count = int((patient_outer_fold_counts > 1).sum())
    image_counts = pd.crosstab(rows["outer_fold"], rows["label"]).reindex(
        index=range(num_folds),
        columns=labels,
        fill_value=0,
    )
    patient_counts = pd.crosstab(patient_labels["outer_fold"], patient_labels["label"]).reindex(
        index=range(num_folds),
        columns=labels,
        fill_value=0,
    )
    return {
        "dataset": "pad_ufes",
        "label_mode": "native",
        "protocol": PROTOCOL,
        "num_folds": num_folds,
        "seed": seed,
        "group_key": "patient_id",
        "image_count": len(rows),
        "patient_count": int(rows["patient_id"].nunique()),
        "patient_lesion_count": int(rows[["patient_id", "lesion_id"]].drop_duplicates().shape[0]),
        "patient_outer_fold_overlap_count": patient_outer_fold_overlap_count,
        "patients_assigned_once": bool((patient_outer_fold_counts == 1).all()),
        "each_patient_is_test_once": patient_outer_fold_overlap_count == 0,
        "images_by_outer_fold": {
            str(fold): int((rows["outer_fold"] == fold).sum()) for fold in range(num_folds)
        },
        "patients_by_outer_fold": {
            str(fold): int(rows.loc[rows["outer_fold"] == fold, "patient_id"].nunique())
            for fold in range(num_folds)
        },
        "images_by_outer_fold_and_label": _nested_counts(image_counts),
        "patients_by_outer_fold_and_label": _nested_counts(patient_counts),
    }


def main() -> None:
    args = parse_args()
    fold_paths = prepare_cross_validation(
        args.raw_dir,
        args.out_dir,
        num_folds=args.folds,
        seed=args.seed,
    )
    print(
        f"Wrote {len(fold_paths)} PAD-UFES-native grouped CV splits to "
        f"{project_relative(args.out_dir)}"
    )
    for path in fold_paths:
        rows = pd.read_csv(path)
        counts = rows["split"].value_counts().reindex(SPLIT_ORDER).to_dict()
        print(f"{path.name}: {counts}")


if __name__ == "__main__":
    main()

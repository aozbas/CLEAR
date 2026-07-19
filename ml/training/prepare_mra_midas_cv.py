"""Prepare deterministic record-grouped MRA-MIDAS development folds.

MRA-MIDAS is authorized development data in this workflow, not an untouched holdout. Generated
manifests contain private identifiers and must remain under ignored paths.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ml.evaluation.mra_midas import (
    CLINICAL_DISTANCES,
    DEFAULT_DATA_DIR,
    load_source_tables,
    prepare_primary_cohort,
    resolve_project_path,
    validate_authorized_source_hashes,
    validate_frozen_audit,
)
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_pad_ufes import SPLIT_ORDER, _nested_counts, project_relative
from ml.training.prepare_pad_ufes_cv import _role_for_outer_fold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "mra_midas_multisource_cv"
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
PROTOCOL = "record_grouped_rotating_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare record-grouped MRA-MIDAS multi-source development folds."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def assign_record_folds(
    lesion_rows: pd.DataFrame,
    *,
    labels: tuple[str, ...] = PAD_UFES_NATIVE_LABELS,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    if num_folds < 3:
        raise ValueError("num_folds must be at least 3 for rotating train/val/test roles.")
    required = {"record_id", "label"}
    missing = required.difference(lesion_rows.columns)
    if missing:
        raise ValueError(f"MRA-MIDAS lesion rows are missing columns: {', '.join(sorted(missing))}")
    if lesion_rows.empty:
        raise ValueError("MRA-MIDAS lesion rows must not be empty.")
    if bool(lesion_rows["record_id"].astype(str).str.strip().eq("").any()):
        raise ValueError("MRA-MIDAS lesion rows contain a blank record_id.")
    unknown = sorted(set(lesion_rows["label"]) - set(labels))
    if unknown:
        raise ValueError(f"MRA-MIDAS lesion rows contain non-native labels: {unknown}")

    record_labels = {
        str(record_id): frozenset(group["label"].astype(str))
        for record_id, group in lesion_rows.groupby("record_id", sort=True)
    }
    records_by_label = {
        label: [record_id for record_id, values in record_labels.items() if label in values]
        for label in labels
    }
    for label, record_ids in records_by_label.items():
        if len(record_ids) < num_folds:
            raise ValueError(
                f"Not enough MRA-MIDAS record groups for label={label!r} across {num_folds} "
                f"folds; found {len(record_ids)}."
            )

    label_frequency = {label: len(record_ids) for label, record_ids in records_by_label.items()}
    stratification_labels = {
        record_id: min(values, key=lambda label: (label_frequency[label], label))
        for record_id, values in record_labels.items()
    }
    rng = random.Random(seed)
    assignments: dict[str, int] = {}
    for label in labels:
        record_ids = sorted(
            record_id
            for record_id, stratification_label in stratification_labels.items()
            if stratification_label == label
        )
        rng.shuffle(record_ids)
        fold_order = list(range(num_folds))
        rng.shuffle(fold_order)
        for index, record_id in enumerate(record_ids):
            assignments[record_id] = fold_order[index % num_folds]

    if len(assignments) != len(record_labels):
        raise ValueError("Some MRA-MIDAS records were not assigned to an outer fold.")
    _validate_outer_fold_coverage(
        lesion_rows,
        assignments,
        labels=labels,
        num_folds=num_folds,
    )
    return assignments


def prepare_cross_validation(
    data_dir: Path,
    out_dir: Path,
    *,
    num_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    data_dir = resolve_project_path(Path(data_dir))
    source_hashes = validate_authorized_source_hashes(data_dir)
    release_rows, manifest_rows = load_source_tables(data_dir)
    prepared = prepare_primary_cohort(release_rows, manifest_rows)
    validate_frozen_audit(prepared.audit)

    lesion_rows = prepared.lesion_rows.copy()
    assignments = assign_record_folds(
        lesion_rows,
        num_folds=num_folds,
        seed=seed,
    )
    lesion_rows["outer_fold"] = lesion_rows["record_id"].astype(str).map(assignments)
    image_rows = prepared.image_rows.merge(
        lesion_rows[["lesion_index", "profile_id", "record_id", "label", "outer_fold"]],
        on="lesion_index",
        how="inner",
        validate="many_to_one",
    )
    if len(image_rows) != len(prepared.image_rows):
        raise RuntimeError("MRA-MIDAS fold preparation lost primary-cohort images.")

    out_dir = resolve_project_path(Path(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_paths: list[Path] = []
    for fold_index in range(num_folds):
        validation_fold = (fold_index + 1) % num_folds
        fold_rows = image_rows.copy()
        fold_rows["split"] = fold_rows["outer_fold"].apply(
            lambda outer_fold, test_fold=fold_index, val_fold=validation_fold: _role_for_outer_fold(
                int(outer_fold),
                test_fold=test_fold,
                validation_fold=val_fold,
            )
        )
        _validate_grouped_fold(fold_rows)
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
                image_rows,
                source_hashes=source_hashes,
                num_folds=num_folds,
                seed=seed,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(prepared.audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return fold_paths


def _validate_outer_fold_coverage(
    lesion_rows: pd.DataFrame,
    assignments: dict[str, int],
    *,
    labels: tuple[str, ...],
    num_folds: int,
) -> None:
    record_labels = lesion_rows[["record_id", "label"]].drop_duplicates().copy()
    record_labels["outer_fold"] = record_labels["record_id"].astype(str).map(assignments)
    coverage = pd.crosstab(record_labels["outer_fold"], record_labels["label"]).reindex(
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
        raise ValueError("Outer MRA-MIDAS folds have missing label coverage: " + ", ".join(missing))


def _validate_grouped_fold(rows: pd.DataFrame) -> None:
    if set(rows["distance"]) != set(CLINICAL_DISTANCES):
        raise ValueError("MRA-MIDAS development folds must contain both clinical distances.")
    if bool((rows.groupby("record_id")["split"].nunique() != 1).any()):
        raise ValueError("An MRA-MIDAS record appears in multiple fold roles.")
    if bool((rows.groupby("profile_id")["split"].nunique() != 1).any()):
        raise ValueError("An MRA-MIDAS lesion appears in multiple fold roles.")
    distance_coverage = rows.groupby("profile_id")["distance"].agg(lambda values: set(values))
    if any(values != set(CLINICAL_DISTANCES) for values in distance_coverage):
        raise ValueError("An MRA-MIDAS lesion is missing a required paired distance.")
    label_coverage = pd.crosstab(rows["split"], rows["label"]).reindex(
        index=SPLIT_ORDER,
        columns=PAD_UFES_NATIVE_LABELS,
        fill_value=0,
    )
    missing = [
        f"{split}/{label}"
        for split in SPLIT_ORDER
        for label in PAD_UFES_NATIVE_LABELS
        if int(label_coverage.loc[split, label]) == 0
    ]
    if missing:
        raise ValueError("MRA-MIDAS fold has missing label coverage: " + ", ".join(missing))


def _sorted_output(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows[["split", "profile_id", "record_id", "distance", "image_path", "label"]].copy()
    output.insert(1, "source", "mra_midas")
    output = output.rename(columns={"profile_id": "unit_id"})
    output["image_path"] = output["image_path"].map(project_relative)
    output["_split_order"] = output["split"].map(
        {split: index for index, split in enumerate(SPLIT_ORDER)}
    )
    return (
        output.sort_values(["_split_order", "label", "unit_id", "distance", "image_path"])
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
) -> dict[str, object]:
    units = rows[["profile_id", "record_id", "split", "label"]].drop_duplicates()
    unit_counts = pd.crosstab(units["split"], units["label"]).reindex(
        index=SPLIT_ORDER,
        columns=PAD_UFES_NATIVE_LABELS,
        fill_value=0,
    )
    return {
        "dataset": "mra_midas",
        "role": "authorized_multisource_development",
        "protocol": PROTOCOL,
        "num_folds": num_folds,
        "fold_index": fold_index,
        "test_outer_fold": fold_index,
        "validation_outer_fold": validation_fold,
        "training_outer_folds": [
            fold for fold in range(num_folds) if fold not in {fold_index, validation_fold}
        ],
        "seed": seed,
        "group_key": "midas_record_id",
        "feature_unit": "paired_distance_lesion_embedding",
        "distance_aggregation": "mean_within_distance_then_equal_distance_mean_then_l2",
        "record_overlap_count": int((rows.groupby("record_id")["split"].nunique() > 1).sum()),
        "lesion_overlap_count": int((rows.groupby("profile_id")["split"].nunique() > 1).sum()),
        "image_count": len(rows),
        "unit_count": len(units),
        "record_count": int(units["record_id"].nunique()),
        "images_by_split": {split: int((rows["split"] == split).sum()) for split in SPLIT_ORDER},
        "units_by_split": {split: int((units["split"] == split).sum()) for split in SPLIT_ORDER},
        "records_by_split": {
            split: int(units.loc[units["split"] == split, "record_id"].nunique())
            for split in SPLIT_ORDER
        },
        "units_by_split_and_label": _nested_counts(unit_counts),
    }


def _build_cv_summary(
    rows: pd.DataFrame,
    *,
    source_hashes: dict[str, str],
    num_folds: int,
    seed: int,
) -> dict[str, object]:
    units = rows[["profile_id", "record_id", "label", "outer_fold"]].drop_duplicates()
    record_folds = units.groupby("record_id")["outer_fold"].nunique()
    unit_counts = pd.crosstab(units["outer_fold"], units["label"]).reindex(
        index=range(num_folds),
        columns=PAD_UFES_NATIVE_LABELS,
        fill_value=0,
    )
    return {
        "dataset": "mra_midas",
        "role": "authorized_multisource_development",
        "protocol": PROTOCOL,
        "num_folds": num_folds,
        "seed": seed,
        "group_key": "midas_record_id",
        "feature_unit": "paired_distance_lesion_embedding",
        "distance_aggregation": "mean_within_distance_then_equal_distance_mean_then_l2",
        "source_table_sha256": source_hashes,
        "image_count": len(rows),
        "unit_count": len(units),
        "record_count": int(units["record_id"].nunique()),
        "record_outer_fold_overlap_count": int((record_folds > 1).sum()),
        "records_assigned_once": bool((record_folds == 1).all()),
        "each_record_is_test_once": bool((record_folds == 1).all()),
        "units_by_outer_fold": {
            str(fold): int((units["outer_fold"] == fold).sum()) for fold in range(num_folds)
        },
        "records_by_outer_fold": {
            str(fold): int(units.loc[units["outer_fold"] == fold, "record_id"].nunique())
            for fold in range(num_folds)
        },
        "units_by_outer_fold_and_label": _nested_counts(unit_counts),
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
        f"Wrote {len(fold_paths)} MRA-MIDAS development folds to {project_relative(args.out_dir)}"
    )


if __name__ == "__main__":
    main()

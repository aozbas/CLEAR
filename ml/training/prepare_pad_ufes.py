"""Prepare PAD-UFES-20 clinical-phone evaluation splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS, validate_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "pad_ufes"
DEFAULT_OUT_PATH = PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes.csv"
DEFAULT_SEED = 42

PAD_UFES_NATIVE_TO_CANONICAL = {
    "MEL": "melanoma",
    "NEV": "nevus",
    "BCC": "basal_cell_carcinoma",
    "ACK": "actinic_keratosis",
    "SCC": "squamous_cell_carcinoma",
    "SEK": "seborrheic_keratosis",
}
LABEL_MODES = {
    "native": (PAD_UFES_NATIVE_TO_CANONICAL, ("BOD", "BOW"), PAD_UFES_NATIVE_LABELS),
}
METADATA_FILENAMES = ("metadata.csv", "PAD-UFES-20.csv", "pad-ufes-20.csv")
REQUIRED_COLUMNS = {"patient_id", "lesion_id", "img_id", "diagnostic"}
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
SPLIT_ORDER = ("train", "val", "test")
SPLIT_STRATEGIES = ("all-test", "patient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PAD-UFES-20 split.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--label-mode",
        choices=sorted(LABEL_MODES),
        default="native",
        help="Use the PAD-UFES six-class phone-photo taxonomy.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=SPLIT_STRATEGIES,
        default="all-test",
        help=(
            "Use 'all-test' for external/zero-shot evaluation compatibility or "
            "'patient' for deterministic supervised train/val/test splits."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def split_counts(total: int) -> dict[str, int]:
    raw_counts = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: int(raw_counts[split]) for split in SPLIT_ORDER}
    remaining = total - sum(counts.values())
    by_fraction = sorted(
        SPLIT_ORDER,
        key=lambda split: (raw_counts[split] - counts[split], -SPLIT_ORDER.index(split)),
        reverse=True,
    )
    for split in by_fraction[:remaining]:
        counts[split] += 1
    return counts


def assign_patient_splits(
    rows: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    seed: int,
) -> dict[str, str]:
    """Stratify patients by their rarest label and assign each patient once."""
    patient_labels = {
        str(patient_id): frozenset(group["label"].tolist())
        for patient_id, group in rows.groupby("patient_id", sort=True)
    }
    patients_by_label = {
        label: [patient_id for patient_id, values in patient_labels.items() if label in values]
        for label in labels
    }
    for label, patient_ids in patients_by_label.items():
        targets = split_counts(len(patient_ids))
        missing_splits = [split for split, count in targets.items() if count == 0]
        if missing_splits:
            raise ValueError(
                f"Not enough patient groups for label={label!r} to cover every split; "
                f"found {len(patient_ids)}."
            )

    label_frequency = {label: len(patient_ids) for label, patient_ids in patients_by_label.items()}
    stratification_labels = {
        patient_id: min(
            values,
            key=lambda label: (label_frequency[label], label),
        )
        for patient_id, values in patient_labels.items()
    }
    rng = random.Random(seed)
    assignments: dict[str, str] = {}

    for label in labels:
        patient_ids = sorted(
            patient_id
            for patient_id, stratification_label in stratification_labels.items()
            if stratification_label == label
        )
        rng.shuffle(patient_ids)
        counts = split_counts(len(patient_ids))
        start = 0
        for split in SPLIT_ORDER:
            end = start + counts[split]
            for patient_id in patient_ids[start:end]:
                assignments[patient_id] = split
            start = end

    return assignments


def prepare(
    raw_dir: Path,
    out_path: Path,
    *,
    label_mode: str = "native",
    split_strategy: str = "all-test",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    prepared, excluded_counts, allowed_labels = load_prepared_rows(
        raw_dir,
        label_mode=label_mode,
    )

    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"Unknown PAD-UFES split strategy: {split_strategy}")

    if split_strategy == "patient":
        assignments = assign_patient_splits(prepared, labels=allowed_labels, seed=seed)
        prepared["split"] = prepared["patient_id"].map(assignments)
        _validate_grouped_splits(prepared, labels=allowed_labels)
    else:
        prepared["split"] = "test"

    prepared["_split_order"] = prepared["split"].map(
        {split: index for index, split in enumerate(SPLIT_ORDER)}
    )
    prepared = prepared.sort_values(["_split_order", "label", "image_path"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = prepared[["split", "image_path", "label"]].reset_index(drop=True)
    output.to_csv(out_path, index=False)
    out_path.with_suffix(".excluded.json").write_text(
        json.dumps(excluded_counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(
            _build_summary(
                prepared,
                label_mode=label_mode,
                split_strategy=split_strategy,
                seed=seed,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def load_prepared_rows(
    raw_dir: Path,
    *,
    label_mode: str,
) -> tuple[pd.DataFrame, dict[str, int], tuple[str, ...]]:
    try:
        label_map, deferred_labels, allowed_labels = LABEL_MODES[label_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown PAD-UFES label mode: {label_mode}") from exc

    raw_dir = Path(raw_dir)
    rows = _read_metadata(raw_dir)
    excluded_counts = dict.fromkeys(deferred_labels, 0)
    output_rows: list[dict[str, str]] = []

    for row in rows.itertuples(index=False):
        raw_label = str(row.diagnostic).strip().upper()
        if raw_label in excluded_counts:
            excluded_counts[raw_label] += 1
            continue
        if raw_label not in label_map:
            raise ValueError(f"Unknown PAD-UFES diagnostic value: {raw_label}")

        label = label_map[raw_label]
        validate_label(label, labels=allowed_labels)
        image_path = _find_image_path(raw_dir, str(row.img_id).strip())
        output_rows.append(
            {
                "patient_id": _required_group_value(row.patient_id, "patient_id"),
                "lesion_id": _required_group_value(row.lesion_id, "lesion_id"),
                "image_path": project_relative(image_path),
                "label": label,
            }
        )

    prepared = pd.DataFrame(
        output_rows,
        columns=["patient_id", "lesion_id", "image_path", "label"],
    )
    _validate_patient_lesions(prepared)
    return prepared, excluded_counts, allowed_labels


def _required_group_value(value: object, column: str) -> str:
    if pd.isna(value) or not str(value).strip():
        raise ValueError(f"PAD-UFES metadata has a missing {column} value.")
    return str(value).strip()


def _validate_patient_lesions(rows: pd.DataFrame) -> None:
    diagnoses_per_lesion = rows.groupby(["patient_id", "lesion_id"])["label"].nunique()
    if not diagnoses_per_lesion.empty and int(diagnoses_per_lesion.max()) > 1:
        raise ValueError("A PAD-UFES patient-lesion pair has multiple diagnostic labels.")


def _validate_grouped_splits(rows: pd.DataFrame, *, labels: tuple[str, ...]) -> None:
    splits_per_patient = rows.groupby("patient_id")["split"].nunique()
    if not splits_per_patient.empty and int(splits_per_patient.max()) != 1:
        raise ValueError("A PAD-UFES patient was assigned to multiple splits.")

    splits_per_lesion = rows.groupby(["patient_id", "lesion_id"])["split"].nunique()
    if not splits_per_lesion.empty and int(splits_per_lesion.max()) != 1:
        raise ValueError("A PAD-UFES patient-lesion pair was assigned to multiple splits.")

    present = pd.crosstab(rows["split"], rows["label"]).reindex(
        index=SPLIT_ORDER,
        columns=labels,
        fill_value=0,
    )
    missing = [
        f"{split}/{label}"
        for split in SPLIT_ORDER
        for label in labels
        if int(present.loc[split, label]) == 0
    ]
    if missing:
        raise ValueError(f"Grouped PAD-UFES split has missing label coverage: {', '.join(missing)}")


def _build_summary(
    rows: pd.DataFrame,
    *,
    label_mode: str,
    split_strategy: str,
    seed: int,
) -> dict[str, object]:
    patient_labels = rows[["patient_id", "split", "label"]].drop_duplicates()
    image_counts = pd.crosstab(rows["split"], rows["label"]).reindex(
        index=SPLIT_ORDER,
        fill_value=0,
    )
    patient_counts = pd.crosstab(patient_labels["split"], patient_labels["label"]).reindex(
        index=SPLIT_ORDER,
        fill_value=0,
    )
    return {
        "dataset": "pad_ufes",
        "label_mode": label_mode,
        "split_strategy": split_strategy,
        "seed": seed,
        "group_key": "patient_id" if split_strategy == "patient" else None,
        "patient_overlap_count": int((rows.groupby("patient_id")["split"].nunique() > 1).sum()),
        "patient_lesion_overlap_count": int(
            (rows.groupby(["patient_id", "lesion_id"])["split"].nunique() > 1).sum()
        ),
        "image_count": len(rows),
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


def _nested_counts(counts: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        str(split): {str(label): int(value) for label, value in row.items()}
        for split, row in counts.iterrows()
    }


def _read_metadata(raw_dir: Path) -> pd.DataFrame:
    for filename in METADATA_FILENAMES:
        path = raw_dir / filename
        if path.exists():
            rows = pd.read_csv(path)
            missing = REQUIRED_COLUMNS.difference(rows.columns)
            if missing:
                missing_columns = ", ".join(sorted(missing))
                raise ValueError(f"PAD-UFES metadata is missing columns: {missing_columns}")
            return rows
    expected = ", ".join(METADATA_FILENAMES)
    raise FileNotFoundError(f"Missing PAD-UFES metadata in {raw_dir}; expected one of: {expected}")


def _find_image_path(raw_dir: Path, img_id: str) -> Path:
    candidates = [
        raw_dir / "images" / img_id,
        raw_dir / img_id,
        raw_dir / "imgs_part_1" / img_id,
        raw_dir / "imgs_part_2" / img_id,
        raw_dir / "imgs_part_3" / img_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing PAD-UFES image for img_id={img_id}; expected: {expected}")


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    args = parse_args()
    output = prepare(
        args.raw_dir,
        args.out,
        label_mode=args.label_mode,
        split_strategy=args.split_strategy,
        seed=args.seed,
    )
    excluded_counts = json.loads(args.out.with_suffix(".excluded.json").read_text(encoding="utf-8"))

    print(
        f"Wrote {len(output):,} PAD-UFES {args.label_mode}-label rows "
        f"with split_strategy={args.split_strategy} to "
        f"{project_relative(args.out)}"
    )
    if len(output) > 0:
        print(output["label"].value_counts().sort_index().to_string())
    print("Excluded deferred labels:")
    for label, count in excluded_counts.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()

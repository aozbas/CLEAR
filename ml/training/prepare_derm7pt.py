"""Prepare Derm7pt as an evaluation-only external holdout split.

Run from the project root after placing the Derm7pt release under
`ml/data/raw/derm7pt/release_v0/`:
    python -m ml.training.prepare_derm7pt

The generated CSV is intended for local evaluation and stays ignored with other
dataset artifacts unless the project owner explicitly decides to track it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.evaluation.schema import validate_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "derm7pt" / "release_v0"
DEFAULT_OUT_PATH = PROJECT_ROOT / "ml" / "data" / "external_splits" / "derm7pt.csv"

REQUIRED_METADATA_COLUMNS = {"diagnosis", "derm"}
SPLIT_FILES = {
    "train": "train_indexes.csv",
    "val": "valid_indexes.csv",
    "test": "test_indexes.csv",
}

NEVUS_DIAGNOSES = {
    "blue nevus",
    "clark nevus",
    "combined nevus",
    "congenital nevus",
    "dermal nevus",
    "nevus",
    "recurrent nevus",
    "reed or spitz nevus",
}
DIRECT_DIAGNOSIS_MAP = {
    "basal cell carcinoma": "basal_cell_carcinoma",
    "dermatofibroma": "dermatofibroma",
    "seborrheic keratosis": "benign_keratosis",
    "vascular lesion": "vascular_lesion",
    **{diagnosis: "nevus" for diagnosis in NEVUS_DIAGNOSES},
}
UNSUPPORTED_DIAGNOSES = {
    "lentigo",
    "melanosis",
    "miscellaneous",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Derm7pt external holdout split.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def parse_metadata(raw_dir: Path) -> pd.DataFrame:
    metadata_path = Path(raw_dir) / "meta" / "meta.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing Derm7pt metadata file: {metadata_path}")

    rows = pd.read_csv(metadata_path)
    missing = REQUIRED_METADATA_COLUMNS.difference(rows.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Derm7pt metadata is missing required columns: {missing_columns}")
    return rows


def prepare(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = parse_metadata(raw_dir)
    split_by_index = read_split_indexes(raw_dir)

    output_rows = []
    for row_index, row in rows.iterrows():
        label = canonical_label(str(row["diagnosis"]))
        if label is None:
            continue

        split = split_by_index.get(int(row_index))
        if split is None:
            raise ValueError(f"Derm7pt row index {row_index} is missing from split index files.")

        image_path = find_image_path(raw_dir, str(row["derm"]))
        output_rows.append(
            {
                "split": split,
                "image_path": project_relative(image_path),
                "label": label,
            }
        )

    out = pd.DataFrame(output_rows, columns=["split", "image_path", "label"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def canonical_label(diagnosis: str) -> str | None:
    normalized = normalize_diagnosis(diagnosis)
    if normalized.startswith("melanoma"):
        return "melanoma"
    if normalized in UNSUPPORTED_DIAGNOSES:
        return None

    label = DIRECT_DIAGNOSIS_MAP.get(normalized)
    if label is not None:
        validate_label(label)
    return label


def find_image_path(raw_dir: Path, derm_path: str) -> Path:
    derm_path = derm_path.strip()
    if not derm_path:
        raise ValueError("Derm7pt derm image path is empty.")

    image_ref = Path(derm_path)
    candidates = (
        [image_ref]
        if image_ref.is_absolute()
        else [
            Path(raw_dir) / "images" / image_ref,
            Path(raw_dir) / image_ref,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing Derm7pt dermoscopic image; expected one of: {expected}")


def read_split_indexes(raw_dir: Path) -> dict[int, str]:
    split_by_index: dict[int, str] = {}
    for split, filename in SPLIT_FILES.items():
        path = Path(raw_dir) / "meta" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing Derm7pt split file: {path}")

        rows = pd.read_csv(path)
        if "indexes" not in rows.columns:
            raise ValueError(f"Derm7pt split file is missing required column `indexes`: {path}")

        for index in rows["indexes"].dropna():
            row_index = int(index)
            if row_index in split_by_index:
                raise ValueError(f"Derm7pt row index {row_index} appears in multiple splits.")
            split_by_index[row_index] = split

    return split_by_index


def normalize_diagnosis(diagnosis: str) -> str:
    return " ".join(diagnosis.strip().lower().split())


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    args = parse_args()
    out = prepare(args.raw_dir, args.out)
    print(f"Wrote {len(out):,} Derm7pt holdout rows to {project_relative(args.out)}")
    if len(out) > 0:
        print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()

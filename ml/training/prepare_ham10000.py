"""Prepare HAM10000 metadata for supervised training.

Run from the project root:
    python -m ml.training.prepare_ham10000

The output CSV is committed because it defines the reproducible train/val/test
split, while the raw image files remain local and gitignored.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "ham10000"
DEFAULT_OUT_PATH = PROJECT_ROOT / "ml" / "data" / "splits" / "ham10000.csv"
DEFAULT_SEED = 42

HAM10000_TO_CANONICAL = {
    "mel": "melanoma",
    "nv": "nevus",
    "bcc": "basal_cell_carcinoma",
    "akiec": "actinic_keratosis",
    "bkl": "benign_keratosis",
    "df": "dermatofibroma",
    "vasc": "vascular_lesion",
}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

SPLIT_ORDER = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 train/val/test splits.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
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


def find_image_path(raw_dir: Path, image_id: str) -> Path:
    for image_dir in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        path = raw_dir / image_dir / f"{image_id}.jpg"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing image for image_id={image_id}")


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def assign_lesion_splits(lesions: pd.DataFrame, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    assignments: dict[str, str] = {}

    for label, group in lesions.groupby("label", sort=True):
        lesion_ids = sorted(group["lesion_id"].tolist())
        rng.shuffle(lesion_ids)

        counts = split_counts(len(lesion_ids))
        start = 0
        for split in SPLIT_ORDER:
            end = start + counts[split]
            for lesion_id in lesion_ids[start:end]:
                assignments[lesion_id] = split
            start = end

    return assignments


def validate_metadata(meta: pd.DataFrame) -> None:
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"HAM10000_metadata.csv is missing columns: {sorted(missing)}")

    unknown_dx = sorted(set(meta["dx"]) - set(HAM10000_TO_CANONICAL))
    if unknown_dx:
        raise ValueError(f"Unknown HAM10000 dx values: {unknown_dx}")

    dx_per_lesion = meta.groupby("lesion_id")["dx"].nunique()
    bad_lesions = dx_per_lesion[dx_per_lesion > 1]
    if not bad_lesions.empty:
        examples = ", ".join(bad_lesions.index[:5])
        raise ValueError(f"Some lesion_id values have multiple dx labels: {examples}")


def prepare(raw_dir: Path, out_path: Path, seed: int) -> pd.DataFrame:
    metadata_path = raw_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    meta = pd.read_csv(metadata_path)
    validate_metadata(meta)

    meta = meta.copy()
    meta["label"] = meta["dx"].map(HAM10000_TO_CANONICAL)

    lesions = meta[["lesion_id", "label"]].drop_duplicates()
    assignments = assign_lesion_splits(lesions, seed)
    meta["split"] = meta["lesion_id"].map(assignments)

    split_per_lesion = meta.groupby("lesion_id")["split"].nunique()
    if int(split_per_lesion.max()) != 1:
        raise ValueError("A lesion_id was assigned to multiple splits")

    image_paths = []
    for image_id in meta["image_id"]:
        image_paths.append(project_relative(find_image_path(raw_dir, image_id)))
    meta["image_path"] = image_paths

    out = meta[["split", "image_path", "label"]].copy()
    out["_split_order"] = out["split"].map({split: idx for idx, split in enumerate(SPLIT_ORDER)})
    out = out.sort_values(["_split_order", "label", "image_path"]).drop(columns="_split_order")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def print_summary(out: pd.DataFrame, out_path: Path) -> None:
    print(f"Wrote {len(out):,} rows to {project_relative(out_path)}")
    print()
    print("Images by split:")
    print(out["split"].value_counts().reindex(SPLIT_ORDER).to_string())
    print()
    print("Images by split and canonical label:")
    summary = pd.crosstab(out["split"], out["label"]).reindex(SPLIT_ORDER)
    print(summary.to_string())


def main() -> None:
    args = parse_args()
    out = prepare(args.raw_dir, args.out, args.seed)
    print_summary(out, args.out)


if __name__ == "__main__":
    main()

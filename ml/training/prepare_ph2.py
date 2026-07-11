"""Prepare PH2 as an evaluation-only external holdout split.

Run from the project root after placing PH2 files under `ml/data/raw/ph2/`:
    python -m ml.training.prepare_ph2

The generated CSV is intended for local evaluation and stays ignored with other
dataset artifacts unless the project owner explicitly decides to track it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "ph2"
DEFAULT_OUT_PATH = PROJECT_ROOT / "ml" / "data" / "external_splits" / "ph2.csv"

PH2_TO_CANONICAL = {
    "Common Nevus": "nevus",
    "Atypical Nevus": "nevus",
    "Melanoma": "melanoma",
}
PH2_CLINICAL_CODE_TO_DIAGNOSIS = {
    "0": "Common Nevus",
    "1": "Atypical Nevus",
    "2": "Melanoma",
}
METADATA_FILENAMES = ("PH2_dataset.txt", "PH2_dataset.csv", "PH2_dataset.tsv")
REQUIRED_COLUMNS = {"image_id", "diagnosis"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PH2 external holdout split.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def parse_metadata(raw_dir: Path) -> pd.DataFrame:
    metadata_path = _find_metadata_path(raw_dir)
    rows = pd.read_csv(metadata_path, sep=None, engine="python")
    missing = REQUIRED_COLUMNS.difference(rows.columns)
    if missing:
        if metadata_path.suffix.lower() == ".txt":
            return _parse_official_pipe_metadata(metadata_path)
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"PH2 metadata is missing required columns: {missing_columns}")
    return rows


def find_image_path(raw_dir: Path, image_id: str) -> Path:
    matches = sorted(raw_dir.rglob(f"{image_id}.*"))
    image_matches = [
        path for path in matches if path.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}
    ]
    if not image_matches:
        raise FileNotFoundError(f"Missing PH2 image for image_id={image_id}")
    return image_matches[0]


def prepare(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = parse_metadata(raw_dir).copy()
    rows["diagnosis"] = rows["diagnosis"].astype(str).str.strip()
    unknown = sorted(set(rows["diagnosis"]) - set(PH2_TO_CANONICAL))
    if unknown:
        raise ValueError(f"Unknown PH2 diagnosis values: {unknown}")

    output_rows = []
    for row in rows.itertuples(index=False):
        image_id = str(row.image_id).strip()
        image_path = find_image_path(raw_dir, image_id)
        output_rows.append(
            {
                "split": "test",
                "image_path": project_relative(image_path),
                "label": PH2_TO_CANONICAL[str(row.diagnosis).strip()],
            }
        )

    out = pd.DataFrame(output_rows, columns=["split", "image_path", "label"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _find_metadata_path(raw_dir: Path) -> Path:
    for filename in METADATA_FILENAMES:
        path = raw_dir / filename
        if path.exists():
            return path
    expected = ", ".join(METADATA_FILENAMES)
    raise FileNotFoundError(f"Missing PH2 metadata file in {raw_dir}; expected one of: {expected}")


def _parse_official_pipe_metadata(metadata_path: Path) -> pd.DataFrame:
    output_rows = []
    for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip().startswith("||") or "Name" in line:
            continue

        groups = line.split("||")
        if len(groups) < 4:
            raise ValueError(f"Malformed PH2 metadata row: {line}")

        image_id = groups[1].strip()
        clinical_code = groups[3].strip()
        try:
            diagnosis = PH2_CLINICAL_CODE_TO_DIAGNOSIS[clinical_code]
        except KeyError as exc:
            raise ValueError(f"Unknown PH2 clinical diagnosis code: {clinical_code}") from exc

        output_rows.append({"image_id": image_id, "diagnosis": diagnosis})

    if not output_rows:
        raise ValueError(f"No PH2 metadata rows found in {metadata_path}")
    return pd.DataFrame(output_rows, columns=["image_id", "diagnosis"])


def main() -> None:
    args = parse_args()
    out = prepare(args.raw_dir, args.out)
    print(f"Wrote {len(out):,} PH2 holdout rows to {project_relative(args.out)}")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()

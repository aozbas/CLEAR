"""Prepare PAD-UFES-20 clinical-phone evaluation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.evaluation.schema import HAM10000_LABELS, PAD_UFES_NATIVE_LABELS, validate_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "pad_ufes"
DEFAULT_OUT_PATH = PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes.csv"

PAD_UFES_TO_CANONICAL = {
    "MEL": "melanoma",
    "NEV": "nevus",
    "BCC": "basal_cell_carcinoma",
    "ACK": "actinic_keratosis",
}
PAD_UFES_NATIVE_TO_CANONICAL = {
    **PAD_UFES_TO_CANONICAL,
    "SCC": "squamous_cell_carcinoma",
    "SEK": "seborrheic_keratosis",
}
LABEL_MODES = {
    "overlap": (PAD_UFES_TO_CANONICAL, ("SCC", "BOD", "BOW", "SEK"), HAM10000_LABELS),
    "native": (PAD_UFES_NATIVE_TO_CANONICAL, ("BOD", "BOW"), PAD_UFES_NATIVE_LABELS),
}
METADATA_FILENAMES = ("metadata.csv", "PAD-UFES-20.csv", "pad-ufes-20.csv")
REQUIRED_COLUMNS = {"patient_id", "lesion_id", "img_id", "diagnostic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PAD-UFES-20 split.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--label-mode",
        choices=sorted(LABEL_MODES),
        default="overlap",
        help=(
            "Use 'overlap' for the current HAM10000/CLEAR overlap labels or 'native' "
            "for the PAD-UFES six-class phone-photo taxonomy."
        ),
    )
    return parser.parse_args()


def prepare(raw_dir: Path, out_path: Path, *, label_mode: str = "overlap") -> pd.DataFrame:
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
                "split": "test",
                "image_path": project_relative(image_path),
                "label": label,
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(output_rows, columns=["split", "image_path", "label"])
    output.to_csv(out_path, index=False)
    out_path.with_suffix(".excluded.json").write_text(
        json.dumps(excluded_counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


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
    output = prepare(args.raw_dir, args.out, label_mode=args.label_mode)
    excluded_counts = json.loads(args.out.with_suffix(".excluded.json").read_text(encoding="utf-8"))

    print(
        f"Wrote {len(output):,} PAD-UFES {args.label_mode}-label rows to "
        f"{project_relative(args.out)}"
    )
    if len(output) > 0:
        print(output["label"].value_counts().sort_index().to_string())
    print("Excluded deferred labels:")
    for label, count in excluded_counts.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()

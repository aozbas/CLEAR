"""Produce aggregate-only suitability evidence from ignored SCIN metadata."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

TARGET_PATTERNS = {
    "melanoma": re.compile(r"\bmelanoma\b", re.IGNORECASE),
    "squamous_cell_carcinoma": re.compile(r"\bSCC(?:IS)?\b|squamous", re.IGNORECASE),
    "basal_cell_carcinoma": re.compile(r"basal cell carcinoma|\bBCC\b", re.IGNORECASE),
    "actinic_keratosis": re.compile(r"actinic keratos", re.IGNORECASE),
    "nevus": re.compile(r"nevus|naevus|melanocytic nevi|\bmole\b", re.IGNORECASE),
    "seborrheic_keratosis": re.compile(r"SK/ISK|seborrheic", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_literal(value: str, expected_type: type) -> Any:
    if not value:
        return expected_type()
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, expected_type):
        raise ValueError(f"Expected {expected_type.__name__} metadata value")
    return parsed


def audit(data_dir: Path) -> dict[str, object]:
    cases_path = data_dir / "scin_cases.csv"
    labels_path = data_dir / "scin_labels.csv"
    if not cases_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Expected scin_cases.csv and scin_labels.csv")

    case_count = 0
    image_reference_count = 0
    with cases_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            case_count += 1
            image_reference_count += sum(
                bool(row.get(column)) for column in ("image_1_path", "image_2_path", "image_3_path")
            )

    anywhere = dict.fromkeys(TARGET_PATTERNS, 0)
    maximum_weight = dict.fromkeys(TARGET_PATTERNS, 0)
    label_row_count = 0
    weighted_row_count = 0

    with labels_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            label_row_count += 1
            names = parse_literal(row.get("dermatologist_skin_condition_on_label_name", ""), list)
            weights = parse_literal(row.get("weighted_skin_condition_label", ""), dict)
            if weights:
                weighted_row_count += 1

            all_names = [str(name) for name in names] + [str(name) for name in weights]
            for target, pattern in TARGET_PATTERNS.items():
                if any(pattern.search(name) for name in all_names):
                    anywhere[target] += 1

            if weights:
                numeric_weights = {str(name): float(weight) for name, weight in weights.items()}
                highest = max(numeric_weights.values())
                for target, pattern in TARGET_PATTERNS.items():
                    if any(
                        pattern.search(name) and math.isclose(weight, highest)
                        for name, weight in numeric_weights.items()
                    ):
                        maximum_weight[target] += 1

    return {
        "schema_version": 1,
        "aggregate_only": True,
        "cases": case_count,
        "image_references": image_reference_count,
        "label_rows": label_row_count,
        "weighted_differential_rows": weighted_row_count,
        "files": {
            "scin_cases.csv": {"sha256": sha256(cases_path)},
            "scin_labels.csv": {"sha256": sha256(labels_path)},
        },
        "target_case_counts": {
            target: {
                "anywhere_in_differential": anywhere[target],
                "tied_for_maximum_weight": maximum_weight[target],
            }
            for target in TARGET_PATTERNS
        },
        "limitations": [
            "The source labels are image-based dermatologist differentials.",
            "This audit does not inspect or score images.",
            "Counts do not establish six-class performance or medical validity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ml/data/raw/scin_metadata"),
        help="Ignored directory containing the two official SCIN metadata CSV files.",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

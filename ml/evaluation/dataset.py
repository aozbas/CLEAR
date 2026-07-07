"""Dataset loading helpers for evaluation runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.evaluation.schema import HAM10000_LABELS, EvaluationExample, VALID_SPLITS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {"split", "image_path", "label"}


def load_examples(
    split_csv: Path,
    split: str,
    *,
    base_dir: Path | None = None,
    max_samples: int | None = None,
    samples_per_label: int | None = None,
) -> list[EvaluationExample]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if samples_per_label is not None and samples_per_label <= 0:
        raise ValueError("samples_per_label must be positive.")

    split_csv = Path(split_csv)
    rows = pd.read_csv(split_csv)
    missing_columns = REQUIRED_COLUMNS.difference(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Split CSV is missing required columns: {missing}")

    image_base = base_dir if base_dir is not None else PROJECT_ROOT
    examples: list[EvaluationExample] = []
    for row in rows.itertuples(index=False):
        row_split = str(row.split)
        if row_split != split:
            continue

        label = str(row.label)
        image_path = Path(str(row.image_path))
        if not image_path.is_absolute():
            image_path = image_base / image_path
        image_path = image_path.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Missing evaluation image: {image_path}")

        examples.append(EvaluationExample(image_path=image_path, label=label, split=row_split))

    if samples_per_label is not None:
        by_label: dict[str, list[EvaluationExample]] = {
            label: [] for label in HAM10000_LABELS
        }
        for example in examples:
            by_label[example.label].append(example)

        selected: list[EvaluationExample] = []
        for label in HAM10000_LABELS:
            selected.extend(by_label[label][:samples_per_label])
        examples = selected

    if max_samples is not None:
        examples = examples[:max_samples]

    return examples

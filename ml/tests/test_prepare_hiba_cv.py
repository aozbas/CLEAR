from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_hiba_cv import assign_patient_folds, write_cross_validation_rows


def _fixture_rows(*, patients_per_label: int = 5) -> pd.DataFrame:
    rows = []
    image_number = 0
    for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
        for patient_index in range(patients_per_label):
            image_number += 1
            rows.append(
                {
                    "image_path": f"/private/ISIC_{image_number:07d}.jpg",
                    "label": label,
                    "patient_id": f"private-patient-{label_index}-{patient_index}",
                    "lesion_id": f"private-lesion-{label_index}-{patient_index}",
                    "isic_id": f"ISIC_{image_number:07d}",
                }
            )
    image_number += 1
    repeated = dict(rows[2 * patients_per_label])
    repeated["image_path"] = f"/private/ISIC_{image_number:07d}.jpg"
    repeated["isic_id"] = f"ISIC_{image_number:07d}"
    rows.append(repeated)
    return pd.DataFrame(rows)


class PrepareHibaCrossValidationTests(unittest.TestCase):
    def test_rotating_folds_are_deterministic_grouped_and_fully_covered(self) -> None:
        rows = _fixture_rows()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first_paths = write_cross_validation_rows(
                rows,
                root / "first",
                audit={"aggregate": "fixture"},
                num_folds=5,
                seed=42,
            )
            second_paths = write_cross_validation_rows(
                rows,
                root / "second",
                audit={"aggregate": "fixture"},
                num_folds=5,
                seed=42,
            )
            first_folds = [pd.read_csv(path, dtype=str) for path in first_paths]
            second_folds = [pd.read_csv(path, dtype=str) for path in second_paths]
            summary_text = (root / "first" / "cv.summary.json").read_text(encoding="utf-8")
            summary = json.loads(summary_text)

        roles_by_image: dict[str, list[str]] = {}
        for first, second in zip(first_folds, second_folds, strict=True):
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(set(first["split"]), {"train", "val", "test"})
            self.assertEqual(int(first.groupby("patient_id")["split"].nunique().max()), 1)
            self.assertEqual(int(first.groupby("lesion_id")["split"].nunique().max()), 1)
            coverage = first.groupby(["source", "label"])["split"].apply(set)
            self.assertTrue(all(roles == {"train", "val", "test"} for roles in coverage))
            for image_path, split in first[["image_path", "split"]].itertuples(index=False):
                roles_by_image.setdefault(image_path, []).append(split)

        for roles in roles_by_image.values():
            self.assertEqual(roles.count("test"), 1)
            self.assertEqual(roles.count("val"), 1)
            self.assertEqual(roles.count("train"), 3)
        self.assertEqual(summary["protocol"], "hiba_patient_grouped_rotating_development_cv")
        self.assertEqual(summary["image_count"], 31)
        self.assertEqual(summary["lesion_count"], 30)
        self.assertEqual(summary["patient_outer_fold_overlap_count"], 0)
        self.assertEqual(summary["lesion_outer_fold_overlap_count"], 0)
        self.assertTrue(summary["each_image_lesion_and_patient_is_test_once"])
        self.assertNotIn("private-patient", summary_text)
        self.assertNotIn("ISIC_", summary_text)

    def test_assignment_rejects_insufficient_patient_groups(self) -> None:
        rows = _fixture_rows(patients_per_label=5)
        melanoma = PAD_UFES_NATIVE_LABELS[2]
        rows = rows[
            (rows["label"] != melanoma)
            | rows["patient_id"].isin(
                rows.loc[rows["label"] == melanoma, "patient_id"].unique()[:4]
            )
        ]

        with self.assertRaisesRegex(ValueError, "Not enough HIBA patient groups"):
            assign_patient_folds(rows, num_folds=5, seed=42)

    def test_rejects_conflicting_lesion_labels(self) -> None:
        rows = _fixture_rows()
        rows.loc[rows.index[-1], "label"] = PAD_UFES_NATIVE_LABELS[0]

        with TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "lesion maps to multiple labels"):
                write_cross_validation_rows(
                    rows,
                    Path(tmp_dir),
                    audit={"aggregate": "fixture"},
                )


if __name__ == "__main__":
    unittest.main()

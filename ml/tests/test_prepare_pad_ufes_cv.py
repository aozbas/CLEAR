import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ml.training.prepare_pad_ufes_cv import prepare_cross_validation


class PreparePadUfesCrossValidationTests(unittest.TestCase):
    def test_rotating_folds_are_deterministic_grouped_and_fully_covered(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "pad_ufes"
            images_dir = raw_dir / "images"
            images_dir.mkdir(parents=True)
            metadata_rows = []
            for diagnostic in ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK"):
                for index in range(5):
                    image_id = f"{diagnostic.lower()}-{index}.png"
                    (images_dir / image_id).write_bytes(b"fake image")
                    metadata_rows.append(
                        {
                            "patient_id": f"{diagnostic.lower()}-patient-{index}",
                            "lesion_id": "reused-lesion-id",
                            "img_id": image_id,
                            "diagnostic": diagnostic,
                        }
                    )
            metadata = pd.DataFrame(metadata_rows)
            metadata.to_csv(raw_dir / "metadata.csv", index=False)

            first_dir = root / "first"
            second_dir = root / "second"
            first_paths = prepare_cross_validation(raw_dir, first_dir, num_folds=5, seed=42)
            second_paths = prepare_cross_validation(raw_dir, second_dir, num_folds=5, seed=42)

            first_folds = [pd.read_csv(path) for path in first_paths]
            second_folds = [pd.read_csv(path) for path in second_paths]
            summary_text = (first_dir / "cv.summary.json").read_text(encoding="utf-8")
            summary = json.loads(summary_text)

        for first, second in zip(first_folds, second_folds, strict=True):
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(set(first["split"]), {"train", "val", "test"})
            self.assertEqual(
                first.groupby("label")["split"].apply(set).to_dict(),
                {label: {"train", "val", "test"} for label in first["label"].unique()},
            )

        patient_by_image = metadata.set_index("img_id")["patient_id"].to_dict()
        roles_by_patient: dict[str, list[str]] = {}
        for fold in first_folds:
            fold = fold.copy()
            fold["patient_id"] = fold["image_path"].apply(
                lambda path: patient_by_image[Path(path).name]
            )
            self.assertEqual(int(fold.groupby("patient_id")["split"].nunique().max()), 1)
            for patient_id, split in (
                fold[["patient_id", "split"]].drop_duplicates().itertuples(index=False)
            ):
                roles_by_patient.setdefault(patient_id, []).append(split)

        for roles in roles_by_patient.values():
            self.assertEqual(roles.count("test"), 1)
            self.assertEqual(roles.count("val"), 1)
            self.assertEqual(roles.count("train"), 3)
        self.assertEqual(summary["protocol"], "patient_grouped_rotating_cv")
        self.assertEqual(summary["patient_outer_fold_overlap_count"], 0)
        self.assertTrue(summary["patients_assigned_once"])
        self.assertTrue(summary["each_patient_is_test_once"])
        self.assertNotIn("ack-patient-0", summary_text)

    def test_rejects_too_few_patient_groups_for_folds(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "pad_ufes"
            images_dir = raw_dir / "images"
            images_dir.mkdir(parents=True)
            metadata_rows = []
            for diagnostic in ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK"):
                count = 4 if diagnostic == "MEL" else 5
                for index in range(count):
                    image_id = f"{diagnostic.lower()}-{index}.png"
                    (images_dir / image_id).write_bytes(b"fake image")
                    metadata_rows.append(
                        {
                            "patient_id": f"{diagnostic.lower()}-patient-{index}",
                            "lesion_id": f"lesion-{index}",
                            "img_id": image_id,
                            "diagnostic": diagnostic,
                        }
                    )
            pd.DataFrame(metadata_rows).to_csv(raw_dir / "metadata.csv", index=False)

            with self.assertRaisesRegex(ValueError, "Not enough patient groups"):
                prepare_cross_validation(raw_dir, root / "cv", num_folds=5, seed=42)


if __name__ == "__main__":
    unittest.main()

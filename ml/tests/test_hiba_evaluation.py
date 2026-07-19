from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image

from ml.evaluation.hiba import (
    DIAGNOSIS_TO_PAD_UFES,
    EXPECTED_PREPROCESSING,
    aggregate_lesion_probabilities,
    calibration_metrics,
    load_source,
    patient_cluster_bootstrap_intervals,
    prepare_primary_cohort,
    select_artifact_review_rows,
    validate_checkpoint_metadata,
    validate_frozen_audit,
    write_artifact_contact_sheet,
)
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS


def write_hiba_fixture(root: Path) -> Path:
    data_dir = root / "hiba"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    rows = []
    image_number = 0
    diagnosis_by_label = {label: diagnosis for diagnosis, label in DIAGNOSIS_TO_PAD_UFES.items()}
    for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
        for sample_index in range(2):
            image_number += 1
            isic_id = f"ISIC_{image_number:07d}"
            lesion_id = f"fixture-lesion-{label_index}-{sample_index}"
            rows.append(
                _metadata_row(
                    isic_id,
                    diagnosis_by_label[label],
                    lesion_id=lesion_id,
                    patient_id=f"fixture-patient-{label_index}-{sample_index}",
                )
            )
            _write_image(images_dir / f"{isic_id}.jpg", color=(label_index * 30, 40, 80))

    image_number += 1
    repeated_id = f"ISIC_{image_number:07d}"
    rows.append(
        _metadata_row(
            repeated_id,
            "Melanoma, NOS",
            lesion_id="fixture-lesion-2-0",
            patient_id="fixture-patient-2-0",
        )
    )
    _write_image(images_dir / f"{repeated_id}.jpg", color=(220, 40, 80))

    image_number += 1
    dermoscopy_id = f"ISIC_{image_number:07d}"
    rows.append(
        _metadata_row(
            dermoscopy_id,
            "Melanoma, NOS",
            lesion_id="fixture-dermoscopy-lesion",
            patient_id="fixture-dermoscopy-patient",
            image_type="dermoscopic",
        )
    )
    _write_image(images_dir / f"{dermoscopy_id}.jpg", color=(10, 10, 10))

    image_number += 1
    excluded_id = f"ISIC_{image_number:07d}"
    rows.append(
        _metadata_row(
            excluded_id,
            "Dermatofibroma",
            lesion_id="fixture-excluded-lesion",
            patient_id="fixture-excluded-patient",
        )
    )
    _write_image(images_dir / f"{excluded_id}.jpg", color=(150, 150, 150))
    pd.DataFrame(rows).to_csv(data_dir / "metadata.csv", index=False)
    return data_dir


def _metadata_row(
    isic_id: str,
    diagnosis: str,
    *,
    lesion_id: str,
    patient_id: str,
    image_type: str = "clinical: overview",
) -> dict[str, str]:
    return {
        "isic_id": isic_id,
        "attribution": "Hospital Italiano de Buenos Aires",
        "copyright_license": "CC-BY",
        "diagnosis_3": diagnosis,
        "diagnosis_confirm_type": "histopathology",
        "fitzpatrick_skin_type": "II",
        "image_type": image_type,
        "lesion_id": lesion_id,
        "patient_id": patient_id,
    }


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 24), color=color).save(path)


def load_fixture(data_dir: Path):
    metadata, image_paths, source_hashes = load_source(
        data_dir,
        expected_metadata_sha256=None,
    )
    return prepare_primary_cohort(
        metadata,
        image_paths,
        source_hashes=source_hashes,
    )


class HibaSourceTests(unittest.TestCase):
    def test_loads_complete_source_and_keeps_only_exact_clinical_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            prepared = load_fixture(write_hiba_fixture(Path(tmp_dir)))

        self.assertEqual(len(prepared.metadata_rows), 15)
        self.assertEqual(len(prepared.clinical_rows), 14)
        self.assertEqual(len(prepared.image_rows), 13)
        self.assertEqual(len(prepared.lesion_rows), 12)
        self.assertEqual(set(prepared.lesion_rows["label"]), set(PAD_UFES_NATIVE_LABELS))
        self.assertEqual(prepared.audit["repeated_lesion_count"], 1)
        self.assertEqual(
            prepared.audit["excluded_clinical_diagnosis_support"],
            {"Dermatofibroma": 1},
        )
        self.assertNotIn("fixture-patient", json.dumps(prepared.audit))
        self.assertNotIn("ISIC_", json.dumps(prepared.audit))

    def test_rejects_extra_image_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_hiba_fixture(Path(tmp_dir))
            _write_image(data_dir / "images" / "ISIC_9999999.jpg", color=(0, 0, 0))

            with self.assertRaisesRegex(ValueError, "missing=0 extra=1"):
                load_source(data_dir, expected_metadata_sha256=None)

    def test_rejects_conflicting_patient_ids_within_lesion(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_hiba_fixture(Path(tmp_dir))
            metadata_path = data_dir / "metadata.csv"
            rows = pd.read_csv(metadata_path, dtype=str)
            repeated = rows["lesion_id"] == "fixture-lesion-2-0"
            rows.loc[repeated, "patient_id"] = ["first-patient", "second-patient"]
            rows.to_csv(metadata_path, index=False)
            metadata, image_paths, source_hashes = load_source(
                data_dir,
                expected_metadata_sha256=None,
            )

            with self.assertRaisesRegex(ValueError, "multiple patient IDs"):
                prepare_primary_cohort(
                    metadata,
                    image_paths,
                    source_hashes=source_hashes,
                )

    def test_frozen_audit_rejects_count_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_row_count"):
            validate_frozen_audit({"source_row_count": 1})


class HibaProtocolTests(unittest.TestCase):
    def test_averages_repeated_images_within_lesion(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            prepared = load_fixture(write_hiba_fixture(Path(tmp_dir)))
        probabilities = np.full(
            (len(prepared.image_rows), len(PAD_UFES_NATIVE_LABELS)),
            1.0 / len(PAD_UFES_NATIVE_LABELS),
        )
        repeated = prepared.lesion_rows[prepared.lesion_rows["image_count"] == 2].iloc[0]
        indices = prepared.image_rows.index[
            prepared.image_rows["lesion_index"] == repeated["lesion_index"]
        ].tolist()
        probabilities[indices[0]] = [0.8, 0.2, 0, 0, 0, 0]
        probabilities[indices[1]] = [0.2, 0.8, 0, 0, 0, 0]

        aggregated = aggregate_lesion_probabilities(
            probabilities,
            prepared.image_rows,
            prepared.lesion_rows,
        )

        np.testing.assert_allclose(
            aggregated[int(repeated["lesion_index"])],
            [0.5, 0.5, 0, 0, 0, 0],
        )

    def test_perfect_probabilities_have_perfect_calibration(self) -> None:
        truth = list(PAD_UFES_NATIVE_LABELS)
        probabilities = np.eye(len(PAD_UFES_NATIVE_LABELS))

        calibration = calibration_metrics(truth, probabilities, bins=10)

        self.assertEqual(calibration["negative_log_likelihood"], 0.0)
        self.assertEqual(calibration["multiclass_brier_score"], 0.0)
        self.assertEqual(calibration["top_label_ece"], 0.0)
        self.assertFalse(calibration["calibration_fitted"])
        self.assertEqual(sum(item["count"] for item in calibration["reliability_bins"]), 6)

    def test_patient_cluster_bootstrap_is_deterministic_and_private(self) -> None:
        truth = list(PAD_UFES_NATIVE_LABELS)
        probabilities = np.eye(len(PAD_UFES_NATIVE_LABELS))
        patients = [f"private-patient-{index}" for index in range(len(truth))]

        first = patient_cluster_bootstrap_intervals(
            truth,
            probabilities,
            patients,
            samples=100,
            seed=42,
        )
        second = patient_cluster_bootstrap_intervals(
            truth,
            probabilities,
            patients,
            samples=100,
            seed=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["sampling_unit"], "released_patient_id")
        self.assertEqual(first["patient_count"], 6)
        self.assertNotIn("private-patient", json.dumps(first))
        self.assertEqual(first["intervals"]["accuracy"], {"lower": 1.0, "upper": 1.0})

    def test_artifact_sample_is_deterministic_and_prediction_blind(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prepared = load_fixture(write_hiba_fixture(root))

            first = select_artifact_review_rows(
                prepared.image_rows,
                samples_per_class=2,
                seed=42,
            )
            second = select_artifact_review_rows(
                prepared.image_rows,
                samples_per_class=2,
                seed=42,
            )
            sheet_path = root / "artifact-sheet.jpg"
            write_artifact_contact_sheet(first, sheet_path, samples_per_class=2)

            self.assertEqual(first["isic_id"].tolist(), second["isic_id"].tolist())
            self.assertEqual(len(first), 12)
            self.assertNotIn("prediction", "|".join(first.columns))
            self.assertTrue(sheet_path.is_file())

    def test_checkpoint_metadata_must_match_frozen_native_protocol(self) -> None:
        checkpoint = {
            "architecture": "convnext_tiny",
            "input_mode": "image_only",
            "dataset": "pad_ufes",
            "label_set": "pad_ufes_native",
            "labels": list(PAD_UFES_NATIVE_LABELS),
            "preprocessing": EXPECTED_PREPROCESSING,
            "seed": 42,
            "model_state_dict": {},
        }

        validate_checkpoint_metadata(checkpoint, architecture="convnext_tiny", seed=42)

        checkpoint["labels"] = list(reversed(PAD_UFES_NATIVE_LABELS))
        with self.assertRaisesRegex(ValueError, "labels"):
            validate_checkpoint_metadata(checkpoint, architecture="convnext_tiny", seed=42)


if __name__ == "__main__":
    unittest.main()

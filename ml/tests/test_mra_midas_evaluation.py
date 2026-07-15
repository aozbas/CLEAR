import base64
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ml.evaluation.mra_midas import (
    EXPECTED_PREPROCESSING,
    PATHOLOGY_TO_PAD_UFES,
    aggregate_lesion_probabilities,
    load_source_tables,
    multiclass_metrics,
    prepare_primary_cohort,
    record_cluster_bootstrap_intervals,
    validate_checkpoint_metadata,
    validate_frozen_audit,
)
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS

PATHOLOGIES = (
    "malignant- ak",
    "malignant- bcc",
    "malignant- melanoma",
    "benign-melanocytic nevus",
    "malignant- scc",
    "benign-seborrheic keratosis",
)


def write_mra_midas_fixture(root: Path) -> Path:
    data_dir = root / "mra_midas"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    release_rows = []
    manifest_rows = []

    def add_image(
        *,
        case_index: int,
        pathology: str,
        distance: str,
        sequence: int,
        scientific_name: str | None = None,
        actual_name: str | None = None,
        control: str = "no",
        gender: str = "male",
    ) -> None:
        row_index = len(release_rows)
        distance_token = distance.replace("/", "_").replace(" ", "_")
        scientific = scientific_name or (
            f"fixture-case{case_index}-{distance_token}-{sequence}.jpg"
        )
        actual = actual_name or scientific
        relative = f"nested/{actual}" if row_index == 3 else actual
        image_path = images_dir / Path(relative)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"fixture-image-{row_index}-{actual}".encode()
        image_path.write_bytes(content)
        manifest_rows.append(
            {
                "file_id": f"file-{row_index}",
                "file_name": relative,
                "size": len(content),
                "added_at": "2026-01-01T00:00:00Z",
                "md5_hash": base64.b64encode(hashlib.md5(content).digest()).decode(),
            }
        )
        release_rows.append(
            {
                "_unnamed_var": row_index,
                "midas_record_id": f"record-{case_index}",
                "midas_file_name": scientific,
                "midas_iscontrol": control,
                "midas_distance": distance,
                "midas_location": f"location-{case_index}",
                "midas_path": pathology,
                "midas_pathreport": f"private-pathology-report-{case_index}",
                "midas_gender": gender,
                "midas_age": "60",
                "midas_fitzpatrick": "ii fair skin, blue eyes",
                "midas_melanoma": "no",
                "midas_ethnicity": "no",
                "midas_race": "white",
                "clinical_impression_1": f"impression-{case_index}",
                "clinical_impression_2": "",
                "clinical_impression_3": "",
                "length__mm_": str(case_index + 1),
                "width__mm_": str(case_index + 2),
            }
        )

    for case_index, pathology in enumerate(PATHOLOGIES):
        if case_index == 0:
            add_image(
                case_index=case_index,
                pathology=pathology,
                distance="6in",
                sequence=1,
                scientific_name="fixture-case0-6in-1.jpeg",
                actual_name="fixture-case0-6in-1.jpg",
            )
            add_image(
                case_index=case_index,
                pathology=pathology,
                distance="6in",
                sequence=2,
            )
            add_image(
                case_index=case_index,
                pathology=pathology,
                distance="1ft",
                sequence=1,
                scientific_name="fixture-case0-1ft-1.jpg",
                actual_name="fixture-case0-1ft-1_extra.jpg",
                gender="female",
            )
        else:
            add_image(
                case_index=case_index,
                pathology=pathology,
                distance="6in",
                sequence=1,
            )
            add_image(
                case_index=case_index,
                pathology=pathology,
                distance="1ft",
                sequence=1,
            )
        if pathology == "malignant- melanoma":
            add_image(
                case_index=case_index,
                pathology=pathology,
                distance="n/a - virtual",
                sequence=2,
            )

    add_image(
        case_index=6,
        pathology="malignant- sccis",
        distance="6in",
        sequence=1,
    )
    add_image(
        case_index=6,
        pathology="malignant- sccis",
        distance="1ft",
        sequence=1,
    )
    add_image(
        case_index=7,
        pathology="",
        distance="6in",
        sequence=1,
        control="yes",
    )
    add_image(
        case_index=7,
        pathology="",
        distance="1ft",
        sequence=1,
        control="yes",
    )
    add_image(
        case_index=8,
        pathology="benign-other",
        distance="n/a - virtual",
        sequence=1,
        scientific_name="metadata-unmatched-1.jpg",
        actual_name="archive-unmatched-1.jpg",
    )

    pd.DataFrame(release_rows).to_csv(data_dir / "release_midas.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(data_dir / "data.csv", index=False)
    return data_dir


class MraMidasMetadataTests(unittest.TestCase):
    def test_validates_manifest_and_builds_aggregate_only_primary_cohort(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_mra_midas_fixture(Path(tmp_dir))
            release_rows, manifest_rows = load_source_tables(data_dir)
            prepared = prepare_primary_cohort(release_rows, manifest_rows)

        self.assertEqual(len(prepared.lesion_rows), 6)
        self.assertEqual(set(prepared.lesion_rows["label"]), set(PAD_UFES_NATIVE_LABELS))
        self.assertEqual(
            prepared.audit["filename_resolution"],
            {
                "exact": 16,
                "unique_stem": 1,
                "mutual_unique_normalized_prefix": 1,
                "quarantined": 1,
            },
        )
        self.assertEqual(prepared.audit["primary_repeated_view_profile_count"], 1)
        self.assertEqual(
            prepared.audit["gender_data_quality"]["profiles_with_conflicting_values"], 1
        )
        self.assertEqual(prepared.audit["virtual_exact_class_support"]["image_count"], 1)
        self.assertEqual(
            prepared.audit["scc_in_situ_mapping"],
            "excluded_not_folded_into_squamous_cell_carcinoma",
        )
        serialized = json.dumps(prepared.audit)
        self.assertNotIn("record-", serialized)
        self.assertNotIn("private-pathology-report", serialized)
        self.assertNotIn("fixture-case", serialized)

    def test_rejects_unsafe_manifest_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_mra_midas_fixture(Path(tmp_dir))
            manifest_path = data_dir / "data.csv"
            rows = pd.read_csv(manifest_path)
            rows.loc[0, "file_name"] = "../escape.jpg"
            rows.to_csv(manifest_path, index=False)

            with self.assertRaisesRegex(ValueError, "unsafe file path"):
                load_source_tables(data_dir, validate_files=False)

    def test_rejects_manifest_digest_mismatch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_mra_midas_fixture(Path(tmp_dir))
            first_image = next((data_dir / "images").rglob("*.jpg"))
            first_image.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "size does not match|MD5 does not match"):
                load_source_tables(data_dir)

    def test_frozen_audit_rejects_fixture_cohort_drift(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_mra_midas_fixture(Path(tmp_dir))
            release_rows, manifest_rows = load_source_tables(data_dir)
            prepared = prepare_primary_cohort(release_rows, manifest_rows)

            with self.assertRaisesRegex(ValueError, "frozen cohort audit drifted"):
                validate_frozen_audit(prepared.audit)


class MraMidasProtocolTests(unittest.TestCase):
    def test_pathology_mapping_is_exact_and_excludes_scc_in_situ(self) -> None:
        self.assertEqual(set(PATHOLOGY_TO_PAD_UFES.values()), set(PAD_UFES_NATIVE_LABELS))
        self.assertNotIn("malignant- sccis", PATHOLOGY_TO_PAD_UFES)

    def test_equal_view_aggregation_does_not_overweight_repeated_images(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_mra_midas_fixture(Path(tmp_dir))
            release_rows, manifest_rows = load_source_tables(data_dir)
            prepared = prepare_primary_cohort(release_rows, manifest_rows)

        probabilities = np.full(
            (len(prepared.image_rows), len(PAD_UFES_NATIVE_LABELS)),
            1.0 / len(PAD_UFES_NATIVE_LABELS),
        )
        first_lesion = prepared.lesion_rows.iloc[0]["lesion_index"]
        six_indices = prepared.image_rows.index[
            (prepared.image_rows["lesion_index"] == first_lesion)
            & (prepared.image_rows["distance"] == "6in")
        ].tolist()
        one_indices = prepared.image_rows.index[
            (prepared.image_rows["lesion_index"] == first_lesion)
            & (prepared.image_rows["distance"] == "1ft")
        ].tolist()
        probabilities[six_indices[0]] = [0.8, 0.2, 0, 0, 0, 0]
        probabilities[six_indices[1]] = [0.6, 0.4, 0, 0, 0, 0]
        probabilities[one_indices[0]] = [0.2, 0.8, 0, 0, 0, 0]

        aggregated = aggregate_lesion_probabilities(
            probabilities,
            prepared.image_rows,
            prepared.lesion_rows,
        )

        np.testing.assert_allclose(aggregated["6in"][first_lesion], [0.7, 0.3, 0, 0, 0, 0])
        np.testing.assert_allclose(aggregated["1ft"][first_lesion], [0.2, 0.8, 0, 0, 0, 0])
        np.testing.assert_allclose(
            aggregated["paired_equal_view_mean"][first_lesion],
            [0.45, 0.55, 0, 0, 0, 0],
        )

    def test_multiclass_metrics_use_the_native_six_class_order(self) -> None:
        truth = list(PAD_UFES_NATIVE_LABELS)
        probabilities = np.eye(len(PAD_UFES_NATIVE_LABELS))

        metrics = multiclass_metrics(truth, probabilities)

        self.assertEqual(metrics["labels"], list(PAD_UFES_NATIVE_LABELS))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_record_cluster_bootstrap_is_deterministic_and_private(self) -> None:
        truth = list(PAD_UFES_NATIVE_LABELS)
        probabilities = np.eye(len(PAD_UFES_NATIVE_LABELS))
        records = [f"private-record-{index}" for index in range(len(truth))]

        first = record_cluster_bootstrap_intervals(
            truth,
            probabilities,
            records,
            samples=100,
            seed=42,
        )
        second = record_cluster_bootstrap_intervals(
            truth,
            probabilities,
            records,
            samples=100,
            seed=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["sampling_unit"], "midas_record_id")
        self.assertEqual(first["record_count"], 6)
        self.assertNotIn("private-record", json.dumps(first))

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

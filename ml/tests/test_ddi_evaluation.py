import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ml.evaluation.ddi import (
    BINARY_THRESHOLD,
    EXPECTED_PREPROCESSING,
    MALIGNANT_PAD_UFES_LABELS,
    binary_metrics,
    bootstrap_confidence_intervals,
    build_dataset_audit,
    collapse_native_probabilities,
    evaluate_binary_scores,
    load_ddi_metadata,
    validate_checkpoint_metadata,
)
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS


def write_ddi_fixture(root: Path) -> Path:
    data_dir = root / "ddi"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    rows = []
    row_index = 0
    for skin_tone in (12, 34, 56):
        for malignant in (False, True):
            row_index += 1
            filename = f"{row_index:06d}.png"
            (images_dir / filename).touch()
            rows.append(
                {
                    "_unnamed_var": row_index - 1,
                    "DDI_ID": f"DDI{row_index:06d}",
                    "DDI_file": filename,
                    "skin_tone": skin_tone,
                    "malignant": malignant,
                    "disease": "fixture-disease",
                }
            )
    pd.DataFrame(rows).to_csv(data_dir / "ddi_metadata.csv", index=False)
    return data_dir


class DdiMetadataTests(unittest.TestCase):
    def test_loads_complete_unique_authorized_layout(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rows = load_ddi_metadata(write_ddi_fixture(Path(tmp_dir)))
            audit = build_dataset_audit(rows)

        self.assertEqual(len(rows), 6)
        self.assertEqual(audit["image_count"], 6)
        self.assertEqual(audit["binary_support"], {"non_malignant": 3, "malignant": 3})
        self.assertEqual(audit["skin_tone_support"]["FST_I_II"]["malignant"], 1)
        self.assertEqual(audit["patient_grouping"], "unavailable_in_supplied_metadata")
        self.assertNotIn("DDI_ID", json.dumps(audit))

    def test_rejects_duplicate_image_rows(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_ddi_fixture(Path(tmp_dir))
            metadata_path = data_dir / "ddi_metadata.csv"
            rows = pd.read_csv(metadata_path)
            rows.loc[1, "DDI_file"] = rows.loc[0, "DDI_file"]
            rows.to_csv(metadata_path, index=False)

            with self.assertRaisesRegex(ValueError, "duplicate DDI_file"):
                load_ddi_metadata(data_dir)

    def test_rejects_unindexed_extra_png(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = write_ddi_fixture(Path(tmp_dir))
            (data_dir / "images" / "extra.png").touch()

            with self.assertRaisesRegex(ValueError, "do not match metadata"):
                load_ddi_metadata(data_dir)


class DdiProtocolTests(unittest.TestCase):
    def test_collapses_native_probabilities_into_predeclared_malignancy_score(self) -> None:
        probabilities = np.array(
            [
                [0.10, 0.20, 0.30, 0.15, 0.10, 0.15],
                [0.40, 0.05, 0.05, 0.20, 0.10, 0.20],
            ],
            dtype=np.float64,
        )

        scores = collapse_native_probabilities(
            probabilities,
            labels=PAD_UFES_NATIVE_LABELS,
        )

        self.assertEqual(
            MALIGNANT_PAD_UFES_LABELS,
            (
                "basal_cell_carcinoma",
                "melanoma",
                "squamous_cell_carcinoma",
            ),
        )
        np.testing.assert_allclose(scores, np.array([0.60, 0.20]))
        self.assertEqual(BINARY_THRESHOLD, 0.5)

    def test_binary_metrics_include_confusion_and_auc(self) -> None:
        metrics = binary_metrics(
            np.array([False, False, True, True]),
            np.array([0.1, 0.8, 0.4, 0.9]),
        )

        self.assertEqual(
            metrics["confusion"],
            {
                "true_non_malignant_pred_non_malignant": 1,
                "true_non_malignant_pred_malignant": 1,
                "true_malignant_pred_non_malignant": 1,
                "true_malignant_pred_malignant": 1,
            },
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["roc_auc"], 0.75)

    def test_constant_non_malignant_reference_has_half_balanced_accuracy(self) -> None:
        metrics = binary_metrics(
            np.array([False, False, True, True]),
            np.zeros(4),
        )

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 1.0 / 3.0)
        self.assertEqual(metrics["sensitivity"], 0.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["malignant_f1"], 0.0)
        self.assertEqual(metrics["roc_auc"], 0.5)

    def test_bootstrap_intervals_are_deterministic_and_stratified(self) -> None:
        truth = np.array([False] * 10 + [True] * 10)
        scores = np.array(
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.49]
            + [0.51, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        )

        first = bootstrap_confidence_intervals(truth, scores, samples=100, seed=42)
        second = bootstrap_confidence_intervals(truth, scores, samples=100, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first["method"], "image_level_stratified_percentile_bootstrap")
        self.assertEqual(first["samples"], 100)
        self.assertEqual(first["intervals"]["sensitivity"], {"lower": 1.0, "upper": 1.0})

    def test_fairness_report_includes_support_and_uncertainty_for_every_group(self) -> None:
        truth = np.array([False, True, False, True, False, True])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        skin_tones = np.array([12, 12, 34, 34, 56, 56])

        report = evaluate_binary_scores(
            truth,
            scores,
            skin_tones,
            bootstrap_samples=100,
            seed=42,
        )

        self.assertEqual(report["overall"]["metrics"]["support"]["total"], 6)
        self.assertEqual(
            set(report["by_skin_tone"]),
            {"FST_I_II", "FST_III_IV", "FST_V_VI"},
        )
        for group in report["by_skin_tone"].values():
            self.assertEqual(group["metrics"]["support"]["total"], 2)
            self.assertIn("confidence_intervals", group)

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

        validate_checkpoint_metadata(
            checkpoint,
            architecture="convnext_tiny",
            seed=42,
        )

        checkpoint["labels"] = list(reversed(PAD_UFES_NATIVE_LABELS))
        with self.assertRaisesRegex(ValueError, "labels"):
            validate_checkpoint_metadata(
                checkpoint,
                architecture="convnext_tiny",
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()

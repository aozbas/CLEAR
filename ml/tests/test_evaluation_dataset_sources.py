import unittest

from ml.evaluation.dataset_sources import (
    DATASET_SOURCES,
    contamination_notes,
    get_dataset_source,
)


class EvaluationDatasetSourceTests(unittest.TestCase):
    def test_registry_includes_internal_ham10000_ph2_and_derm7pt_holdouts(self) -> None:
        self.assertIn("ham10000_internal", DATASET_SOURCES)
        self.assertIn("ph2_holdout", DATASET_SOURCES)
        self.assertIn("derm7pt_holdout", DATASET_SOURCES)
        self.assertIn("pad_ufes_clinical", DATASET_SOURCES)

        ph2 = get_dataset_source("ph2_holdout")
        derm7pt = get_dataset_source("derm7pt_holdout")

        self.assertEqual(ph2.name, "PH2 external holdout")
        self.assertEqual(ph2.split_type, "external_holdout")
        self.assertEqual(ph2.labels, ["melanoma", "nevus"])
        self.assertTrue(ph2.partial_label_set)
        self.assertEqual(derm7pt.name, "Derm7pt external holdout")
        self.assertEqual(derm7pt.split_type, "external_holdout")
        self.assertEqual(
            derm7pt.labels,
            [
                "melanoma",
                "nevus",
                "basal_cell_carcinoma",
                "benign_keratosis",
                "dermatofibroma",
                "vascular_lesion",
            ],
        )
        self.assertTrue(derm7pt.partial_label_set)

    def test_registry_includes_phone_clinical_sources(self) -> None:
        pad_ufes = get_dataset_source("pad_ufes_clinical")
        scin = get_dataset_source("scin_user_submitted")
        ddi = get_dataset_source("ddi_clinical")

        self.assertEqual(pad_ufes.split_type, "clinical_phone_holdout")
        self.assertEqual(
            pad_ufes.labels,
            ["melanoma", "nevus", "basal_cell_carcinoma", "actinic_keratosis"],
        )
        self.assertTrue(pad_ufes.partial_label_set)
        self.assertEqual(scin.split_type, "clinical_user_submitted_optional")
        self.assertTrue(scin.partial_label_set)
        self.assertEqual(ddi.split_type, "clinical_fairness_optional")
        self.assertTrue(ddi.partial_label_set)

    def test_known_overlap_warns_for_ham10000_trained_candidate_on_ham10000(self) -> None:
        notes = contamination_notes(
            dataset_source=get_dataset_source("ham10000_internal"),
            model_datasets=["HAM10000"],
        )

        self.assertTrue(any("possible train/test overlap" in note for note in notes))

    def test_external_holdout_notes_no_known_overlap(self) -> None:
        notes = contamination_notes(
            dataset_source=get_dataset_source("ph2_holdout"),
            model_datasets=["HAM10000"],
        )

        self.assertEqual(notes, [])

    def test_derm7pt_holdout_notes_no_known_ham10000_overlap(self) -> None:
        notes = contamination_notes(
            dataset_source=get_dataset_source("derm7pt_holdout"),
            model_datasets=["HAM10000"],
        )

        self.assertEqual(notes, [])

    def test_pad_ufes_notes_known_medsiglip_overlap(self) -> None:
        notes = contamination_notes(
            dataset_source=get_dataset_source("pad_ufes_clinical"),
            model_datasets=["PAD-UFES-20"],
        )

        self.assertTrue(any("possible train/test overlap" in note for note in notes))

    def test_unknown_dataset_source_fails_clearly(self) -> None:
        with self.assertRaises(ValueError):
            get_dataset_source("unknown")


if __name__ == "__main__":
    unittest.main()

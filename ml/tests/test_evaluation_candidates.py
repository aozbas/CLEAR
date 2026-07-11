import unittest

from ml.evaluation.candidates import CANDIDATES, get_candidate


class EvaluationCandidateTests(unittest.TestCase):
    def test_registry_includes_first_wave_candidates(self) -> None:
        for name in (
            "baseline",
            "Miguel764/efficientnetv2s-skin-cancer-classifier",
            "syaha/skin_cancer_detection_model",
            "google/medsiglip-448",
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        ):
            with self.subTest(name=name):
                self.assertIn(name, CANDIDATES)

    def test_each_candidate_has_required_metadata(self) -> None:
        for candidate in CANDIDATES.values():
            with self.subTest(name=candidate.name):
                self.assertTrue(candidate.name)
                self.assertTrue(candidate.source_url)
                self.assertTrue(candidate.adapter_type)
                self.assertTrue(candidate.label_strategy)
                self.assertIsInstance(candidate.notes, list)
                self.assertIsInstance(candidate.training_datasets, list)
                self.assertIsInstance(candidate.clean_dataset_sources, list)

    def test_get_candidate_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            get_candidate("unknown")

    def test_keras_candidates_include_artifact_and_ordered_label_map(self) -> None:
        miguel = get_candidate("Miguel764/efficientnetv2s-skin-cancer-classifier")
        syaha = get_candidate("syaha/skin_cancer_detection_model")

        self.assertEqual(miguel.artifact_filename, "efficientnetv2s.h5")
        self.assertEqual(syaha.artifact_filename, "skin_cancer_model.h5")
        self.assertEqual(
            list(miguel.label_map.items()),
            [
                ("akiec", "actinic_keratosis"),
                ("bcc", "basal_cell_carcinoma"),
                ("bkl", "benign_keratosis"),
                ("df", "dermatofibroma"),
                ("mel", "melanoma"),
                ("nv", "nevus"),
                ("vasc", "vascular_lesion"),
            ],
        )
        self.assertEqual(
            list(syaha.label_map.items()),
            [
                ("akiec", "actinic_keratosis"),
                ("bcc", "basal_cell_carcinoma"),
                ("bkl", "benign_keratosis"),
                ("df", "dermatofibroma"),
                ("nv", "nevus"),
                ("vasc", "vascular_lesion"),
                ("mel", "melanoma"),
            ],
        )

    def test_ham10000_trained_candidates_record_training_dataset(self) -> None:
        for name in (
            "baseline",
            "Miguel764/efficientnetv2s-skin-cancer-classifier",
            "syaha/skin_cancer_detection_model",
            "gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer",
        ):
            with self.subTest(name=name):
                self.assertIn("HAM10000", get_candidate(name).training_datasets)

    def test_baseline_marks_internal_split_as_clean(self) -> None:
        self.assertEqual(get_candidate("baseline").clean_dataset_sources, ["ham10000_internal"])
        self.assertEqual(
            get_candidate("Miguel764/efficientnetv2s-skin-cancer-classifier").clean_dataset_sources,
            [],
        )

    def test_foundation_candidates_use_explicit_zero_shot_adapters(self) -> None:
        self.assertEqual(
            get_candidate("google/medsiglip-448").adapter_type,
            "transformers_zero_shot",
        )
        self.assertEqual(
            get_candidate("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224").adapter_type,
            "open_clip_zero_shot",
        )
        self.assertEqual(
            get_candidate("redlessone/DermLIP_ViT-B-16").adapter_type,
            "open_clip_zero_shot",
        )

    def test_medsiglip_records_pad_ufes_training_overlap(self) -> None:
        self.assertIn("PAD-UFES-20", get_candidate("google/medsiglip-448").training_datasets)


if __name__ == "__main__":
    unittest.main()

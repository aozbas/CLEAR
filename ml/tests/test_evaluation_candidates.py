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

    def test_get_candidate_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            get_candidate("unknown")


if __name__ == "__main__":
    unittest.main()

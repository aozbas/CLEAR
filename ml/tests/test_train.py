import unittest

from ml.training.train import canonical_to_binary


class CanonicalToBinaryTests(unittest.TestCase):
    def test_maps_suspicious_labels(self) -> None:
        self.assertEqual(canonical_to_binary("melanoma"), "suspicious")
        self.assertEqual(canonical_to_binary("basal_cell_carcinoma"), "suspicious")
        self.assertEqual(canonical_to_binary("actinic_keratosis"), "suspicious")

    def test_maps_non_suspicious_labels(self) -> None:
        self.assertEqual(canonical_to_binary("nevus"), "non_suspicious")
        self.assertEqual(canonical_to_binary("benign_keratosis"), "non_suspicious")
        self.assertEqual(canonical_to_binary("dermatofibroma"), "non_suspicious")
        self.assertEqual(canonical_to_binary("vascular_lesion"), "non_suspicious")

    def test_rejects_unknown_labels(self) -> None:
        with self.assertRaises(ValueError):
            canonical_to_binary("squamous_cell_carcinoma")


if __name__ == "__main__":
    unittest.main()

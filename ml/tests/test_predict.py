import unittest

from ml.inference.predict import HAM10000_LABELS, DEFAULT_LABELS, InvalidImageError, get_checkpoint_labels, load_image


class CheckpointLabelTests(unittest.TestCase):
    def test_default_labels_are_ham10000_phase_2_labels(self) -> None:
        self.assertEqual(DEFAULT_LABELS, HAM10000_LABELS)
        self.assertEqual(len(DEFAULT_LABELS), 7)
        self.assertIn("melanoma", DEFAULT_LABELS)
        self.assertIn("vascular_lesion", DEFAULT_LABELS)

    def test_checkpoint_labels_default_for_plain_state_dict(self) -> None:
        self.assertEqual(get_checkpoint_labels({"fc.weight": object()}), DEFAULT_LABELS)

    def test_checkpoint_labels_use_saved_labels(self) -> None:
        labels = ["melanoma", "nevus", "basal_cell_carcinoma"]
        self.assertEqual(get_checkpoint_labels({"labels": labels}), labels)

    def test_checkpoint_labels_reject_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            get_checkpoint_labels({"labels": []})

    def test_load_image_rejects_unreadable_bytes(self) -> None:
        with self.assertRaises(InvalidImageError):
            load_image(b"not an image")


if __name__ == "__main__":
    unittest.main()

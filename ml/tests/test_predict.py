import unittest

from ml.inference.predict import DEFAULT_LABELS, InvalidImageError, get_checkpoint_labels, load_image


class CheckpointLabelTests(unittest.TestCase):
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

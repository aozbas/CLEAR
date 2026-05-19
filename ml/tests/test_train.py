from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ml.training.train import (
    BINARY_LABELS,
    Ham10000Dataset,
    HAM10000_LABELS,
    canonical_to_binary,
    canonical_to_training_label,
    labels_for_mode,
)


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


class LabelModeTests(unittest.TestCase):
    def test_ham10000_mode_keeps_canonical_labels(self) -> None:
        self.assertEqual(
            canonical_to_training_label("melanoma", "ham10000"),
            "melanoma",
        )
        self.assertEqual(
            canonical_to_training_label("vascular_lesion", "ham10000"),
            "vascular_lesion",
        )

    def test_binary_mode_uses_phase_1_grouping(self) -> None:
        self.assertEqual(
            canonical_to_training_label("melanoma", "binary"),
            "suspicious",
        )
        self.assertEqual(
            canonical_to_training_label("nevus", "binary"),
            "non_suspicious",
        )

    def test_labels_for_mode(self) -> None:
        self.assertEqual(labels_for_mode("binary"), BINARY_LABELS)
        self.assertEqual(labels_for_mode("ham10000"), HAM10000_LABELS)

    def test_rejects_unknown_label_mode(self) -> None:
        with self.assertRaises(ValueError):
            labels_for_mode("phase3")


class Ham10000DatasetTests(unittest.TestCase):
    def test_capped_train_sample_keeps_each_ham10000_class(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            split_csv = Path(tmp_dir) / "split.csv"
            rows = ["split,label,image_path"]
            for idx, label in enumerate(HAM10000_LABELS):
                rows.append(f"train,{label},{tmp_dir}/image_{idx}.jpg")
            for idx in range(40):
                rows.append(f"train,nevus,{tmp_dir}/extra_nevus_{idx}.jpg")
            split_csv.write_text("\n".join(rows), encoding="utf-8")

            label_to_idx = {label: idx for idx, label in enumerate(HAM10000_LABELS)}
            dataset = Ham10000Dataset(
                split_csv,
                "train",
                "ham10000",
                HAM10000_LABELS,
                label_to_idx,
                max_samples=len(HAM10000_LABELS),
                seed=42,
            )

        self.assertEqual(sorted(dataset.labels()), list(range(len(HAM10000_LABELS))))


if __name__ == "__main__":
    unittest.main()

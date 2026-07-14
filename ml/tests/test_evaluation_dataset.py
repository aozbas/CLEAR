import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from ml.evaluation.dataset import load_examples
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS


class EvaluationDatasetTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(120, 30, 200)).save(path)

    def test_load_examples_filters_split_and_preserves_order(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "images" / "first.jpg"
            second = root / "images" / "second.jpg"
            val = root / "images" / "val.jpg"
            for image_path in (first, second, val):
                self._write_image(image_path)

            split_csv = root / "split.csv"
            split_csv.write_text(
                "\n".join(
                    [
                        "split,image_path,label",
                        "test,images/first.jpg,melanoma",
                        "val,images/val.jpg,nevus",
                        "test,images/second.jpg,vascular_lesion",
                    ]
                ),
                encoding="utf-8",
            )

            examples = load_examples(split_csv, "test", base_dir=root)

        self.assertEqual([example.label for example in examples], ["melanoma", "vascular_lesion"])
        self.assertEqual([example.image_path for example in examples], [first, second])
        self.assertTrue(all(example.split == "test" for example in examples))

    def test_unknown_split_raises_value_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            split_csv = Path(tmp_dir) / "split.csv"
            split_csv.write_text("split,image_path,label\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_examples(split_csv, "holdout")

    def test_unknown_label_raises_value_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            self._write_image(image_path)
            split_csv = root / "split.csv"
            split_csv.write_text(
                "split,image_path,label\ntest,image.jpg,unknown\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_examples(split_csv, "test", base_dir=root)

    def test_missing_image_raises_file_not_found_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_csv = root / "split.csv"
            split_csv.write_text(
                "split,image_path,label\ntest,missing.jpg,nevus\n",
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                load_examples(split_csv, "test", base_dir=root)

    def test_max_samples_must_be_positive(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            split_csv = Path(tmp_dir) / "split.csv"
            split_csv.write_text("split,image_path,label\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_examples(split_csv, "test", max_samples=0)

    def test_samples_per_label_uses_canonical_label_order(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            rows = ["split,image_path,label"]
            for filename, label in (
                ("nevus.jpg", "nevus"),
                ("melanoma.jpg", "melanoma"),
                ("vascular.jpg", "vascular_lesion"),
                ("second-nevus.jpg", "nevus"),
            ):
                self._write_image(root / filename)
                rows.append(f"test,{filename},{label}")
            split_csv = root / "split.csv"
            split_csv.write_text("\n".join(rows), encoding="utf-8")

            examples = load_examples(
                split_csv,
                "test",
                base_dir=root,
                samples_per_label=1,
            )

        self.assertEqual(
            [example.label for example in examples],
            ["melanoma", "nevus", "vascular_lesion"],
        )

    def test_samples_per_label_accepts_pad_ufes_native_label_order(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            rows = ["split,image_path,label"]
            for filename, label in (
                ("scc.jpg", "squamous_cell_carcinoma"),
                ("melanoma.jpg", "melanoma"),
                ("sek.jpg", "seborrheic_keratosis"),
            ):
                self._write_image(root / filename)
                rows.append(f"test,{filename},{label}")
            split_csv = root / "split.csv"
            split_csv.write_text("\n".join(rows), encoding="utf-8")

            examples = load_examples(
                split_csv,
                "test",
                base_dir=root,
                samples_per_label=1,
                labels=PAD_UFES_NATIVE_LABELS,
            )

        self.assertEqual(
            [example.label for example in examples],
            ["melanoma", "squamous_cell_carcinoma", "seborrheic_keratosis"],
        )


if __name__ == "__main__":
    unittest.main()

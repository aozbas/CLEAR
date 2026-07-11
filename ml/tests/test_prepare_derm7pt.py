import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ml.training.prepare_derm7pt import prepare


class PrepareDerm7ptTests(unittest.TestCase):
    def _write_release(
        self,
        raw_dir: Path,
        *,
        metadata_rows: list[tuple[str, str]],
        train_indexes: list[int],
        valid_indexes: list[int],
        test_indexes: list[int],
        write_images: bool = True,
    ) -> None:
        (raw_dir / "meta").mkdir(parents=True, exist_ok=True)
        (raw_dir / "images" / "derm").mkdir(parents=True, exist_ok=True)

        lines = ["diagnosis,derm,clinic"]
        for index, (diagnosis, derm_path) in enumerate(metadata_rows):
            lines.append(f"{diagnosis},{derm_path},clinic/image-{index}.jpg")
            if write_images:
                (raw_dir / "images" / derm_path).parent.mkdir(parents=True, exist_ok=True)
                (raw_dir / "images" / derm_path).write_bytes(b"fake image")

        (raw_dir / "meta" / "meta.csv").write_text("\n".join(lines), encoding="utf-8")
        self._write_indexes(raw_dir / "meta" / "train_indexes.csv", train_indexes)
        self._write_indexes(raw_dir / "meta" / "valid_indexes.csv", valid_indexes)
        self._write_indexes(raw_dir / "meta" / "test_indexes.csv", test_indexes)

    def _write_indexes(self, path: Path, indexes: list[int]) -> None:
        path.write_text(
            "indexes\n" + "\n".join(str(index) for index in indexes),
            encoding="utf-8",
        )

    def test_prepare_maps_supported_derm7pt_labels_to_official_splits(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "derm7pt" / "release_v0"
            self._write_release(
                raw_dir,
                metadata_rows=[
                    ("melanoma (in situ)", "derm/image-0.jpg"),
                    ("clark nevus", "derm/image-1.jpg"),
                    ("basal cell carcinoma", "derm/image-2.jpg"),
                    ("seborrheic keratosis", "derm/image-3.jpg"),
                    ("dermatofibroma", "derm/image-4.jpg"),
                    ("vascular lesion", "derm/image-5.jpg"),
                    ("lentigo", "derm/image-6.jpg"),
                ],
                train_indexes=[0],
                valid_indexes=[1],
                test_indexes=[2, 3, 4, 5, 6],
            )
            out_path = root / "external_splits" / "derm7pt.csv"

            rows = prepare(raw_dir, out_path)
            written = pd.read_csv(out_path)

        self.assertEqual(list(rows.columns), ["split", "image_path", "label"])
        self.assertEqual(list(written.columns), ["split", "image_path", "label"])
        self.assertEqual(rows["split"].tolist(), ["train", "val", "test", "test", "test", "test"])
        self.assertEqual(
            rows["label"].tolist(),
            [
                "melanoma",
                "nevus",
                "basal_cell_carcinoma",
                "benign_keratosis",
                "dermatofibroma",
                "vascular_lesion",
            ],
        )
        self.assertNotIn("lentigo", written["label"].tolist())
        self.assertTrue(all("images/derm/image-" in path for path in written["image_path"]))

    def test_prepare_missing_supported_image_raises_file_not_found_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "derm7pt" / "release_v0"
            self._write_release(
                raw_dir,
                metadata_rows=[("melanoma", "derm/missing.jpg")],
                train_indexes=[],
                valid_indexes=[],
                test_indexes=[0],
                write_images=False,
            )

            with self.assertRaises(FileNotFoundError):
                prepare(raw_dir, root / "derm7pt.csv")

    def test_prepare_missing_split_file_raises_file_not_found_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "derm7pt" / "release_v0"
            self._write_release(
                raw_dir,
                metadata_rows=[("melanoma", "derm/image.jpg")],
                train_indexes=[],
                valid_indexes=[],
                test_indexes=[0],
            )
            (raw_dir / "meta" / "test_indexes.csv").unlink()

            with self.assertRaises(FileNotFoundError):
                prepare(raw_dir, root / "derm7pt.csv")


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from PIL import Image

from ml.training.prepare_ph2 import prepare


class PreparePh2Tests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(20, 80, 140)).save(path)

    def test_prepare_maps_ph2_labels_to_evaluation_split(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "ph2"
            for image_id in ("IMD001", "IMD002", "IMD003"):
                self._write_image(
                    raw_dir
                    / "PH2 Dataset images"
                    / image_id
                    / f"{image_id}_Dermoscopic_Image"
                    / f"{image_id}.bmp"
                )
            (raw_dir / "PH2_dataset.txt").write_text(
                "\n".join(
                    [
                        "image_id,diagnosis",
                        "IMD001,Common Nevus",
                        "IMD002,Atypical Nevus",
                        "IMD003,Melanoma",
                    ]
                ),
                encoding="utf-8",
            )
            out_path = root / "external_splits" / "ph2.csv"

            rows = prepare(raw_dir, out_path)
            self.assertTrue(out_path.exists())
            written = pd.read_csv(out_path)

        self.assertEqual(list(rows.columns), ["split", "image_path", "label"])
        self.assertEqual(rows["split"].tolist(), ["test", "test", "test"])
        self.assertEqual(rows["label"].tolist(), ["nevus", "nevus", "melanoma"])
        self.assertEqual(written["label"].tolist(), ["nevus", "nevus", "melanoma"])
        self.assertTrue(all("PH2 Dataset images" in path for path in written["image_path"]))

    def test_prepare_reads_official_pipe_table_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "ph2"
            for image_id in ("IMD001", "IMD002", "IMD003"):
                self._write_image(
                    raw_dir
                    / "PH2 Dataset images"
                    / image_id
                    / f"{image_id}_Dermoscopic_Image"
                    / f"{image_id}.bmp"
                )
            (raw_dir / "PH2_dataset.txt").write_text(
                "\n".join(
                    [
                        (
                            "||   Name || Histological Diagnosis || Clinical Diagnosis || "
                            "Asymmetry | Pigment Network | Dots/Globules | Streaks | "
                            "Regression Areas | Blue-Whitish Veil ||           Colors ||"
                        ),
                        (
                            "|| IMD001 ||                        ||                  0 ||"
                            "         0 |               T |             A |       A |"
                            "                A |                 A ||                4 ||"
                        ),
                        (
                            "|| IMD002 ||                        ||                  1 ||"
                            "         0 |               T |             A |       A |"
                            "                A |                 A ||                3 ||"
                        ),
                        (
                            "|| IMD003 ||              Melanoma ||                  2 ||"
                            "         2 |               AT |             P |       P |"
                            "                P |                 P ||             3  4 ||"
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            out_path = root / "external_splits" / "ph2.csv"

            rows = prepare(raw_dir, out_path)

        self.assertEqual(rows["split"].tolist(), ["test", "test", "test"])
        self.assertEqual(rows["label"].tolist(), ["nevus", "nevus", "melanoma"])

    def test_prepare_rejects_unknown_ph2_diagnosis(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "ph2"
            raw_dir.mkdir(parents=True)
            (raw_dir / "PH2_dataset.txt").write_text(
                "image_id,diagnosis\nIMD001,Other\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                prepare(raw_dir, Path(tmp_dir) / "ph2.csv")

    def test_prepare_missing_image_raises_file_not_found_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "ph2"
            raw_dir.mkdir(parents=True)
            (raw_dir / "PH2_dataset.txt").write_text(
                "image_id,diagnosis\nIMD001,Melanoma\n",
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                prepare(raw_dir, Path(tmp_dir) / "ph2.csv")


if __name__ == "__main__":
    unittest.main()

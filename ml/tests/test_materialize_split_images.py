import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from PIL import Image, ImageDraw

from ml.preprocessing import get_transforms
from ml.training.materialize_split_images import materialize_split


class MaterializeSplitImagesTests(unittest.TestCase):
    def test_materialized_png_matches_direct_validation_tensor_and_is_reused(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.png"
            image = Image.new("RGB", (360, 480), color=(20, 40, 60))
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 80, 300, 400), fill=(180, 120, 90))
            draw.ellipse((100, 140, 250, 320), fill=(70, 150, 200))
            image.save(source)

            split_csv = root / "fold_0.csv"
            pd.DataFrame(
                [
                    {
                        "split": "test",
                        "image_path": source.as_posix(),
                        "label": "melanoma",
                    }
                ]
            ).to_csv(split_csv, index=False)
            source_summary = {
                "dataset": "pad_ufes",
                "label_mode": "native",
                "split_strategy": "patient",
                "protocol": "patient_grouped_rotating_cv",
                "group_key": "patient_id",
                "patient_overlap_count": 0,
                "patient_lesion_overlap_count": 0,
                "image_count": 1,
                "images_by_split": {"train": 0, "val": 0, "test": 1},
            }
            split_csv.with_suffix(".summary.json").write_text(
                json.dumps(source_summary),
                encoding="utf-8",
            )
            image_dir = root / "processed"
            first_csv = root / "first.csv"
            second_csv = root / "second.csv"

            first, first_stats = materialize_split(
                split_csv,
                first_csv,
                image_dir,
                workers=1,
            )
            second, second_stats = materialize_split(
                split_csv,
                second_csv,
                image_dir,
                workers=1,
            )
            processed_path = Path(first.iloc[0]["image_path"])
            output_summary = json.loads(
                first_csv.with_suffix(".summary.json").read_text(encoding="utf-8")
            )

            direct_tensor = get_transforms("val")(Image.open(source).convert("RGB"))
            processed_tensor = get_transforms("val")(Image.open(processed_path).convert("RGB"))

        self.assertTrue(torch.equal(direct_tensor, processed_tensor))
        self.assertEqual(first_stats["created_image_count"], 1)
        self.assertEqual(second_stats["reused_image_count"], 1)
        self.assertEqual(first.iloc[0]["image_path"], second.iloc[0]["image_path"])
        self.assertTrue(output_summary["materialization"]["lossless"])
        self.assertEqual(
            output_summary["materialization"]["cache_key"],
            "source_path_and_content_sha256",
        )
        self.assertEqual(output_summary["protocol"], "patient_grouped_rotating_cv")

    def test_source_content_change_invalidates_cached_target(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.png"
            Image.new("RGB", (32, 32), color=(20, 40, 60)).save(source)
            split_csv = root / "fold_0.csv"
            pd.DataFrame(
                [{"split": "test", "image_path": source.as_posix(), "label": "melanoma"}]
            ).to_csv(split_csv, index=False)
            split_csv.with_suffix(".summary.json").write_text(
                json.dumps({"image_count": 1}),
                encoding="utf-8",
            )
            image_dir = root / "processed"

            first, _ = materialize_split(split_csv, root / "first.csv", image_dir, workers=1)
            Image.new("RGB", (32, 32), color=(200, 100, 50)).save(source)
            second, stats = materialize_split(
                split_csv,
                root / "second.csv",
                image_dir,
                workers=1,
            )

        self.assertNotEqual(first.iloc[0]["image_path"], second.iloc[0]["image_path"])
        self.assertEqual(stats["created_image_count"], 1)


if __name__ == "__main__":
    unittest.main()

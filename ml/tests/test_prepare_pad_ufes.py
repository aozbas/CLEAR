import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ml.training.prepare_pad_ufes import prepare


class PreparePadUfesTests(unittest.TestCase):
    def test_prepare_maps_overlap_labels_and_records_deferred_counts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "pad_ufes"
            images_dir = raw_dir / "images"
            images_dir.mkdir(parents=True)
            rows = [
                ("mel.png", "MEL"),
                ("nev.png", "NEV"),
                ("bcc.png", "BCC"),
                ("ack.png", "ACK"),
                ("scc.png", "SCC"),
                ("bod.png", "BOD"),
                ("bow.png", "BOW"),
                ("sek.png", "SEK"),
            ]
            for filename, _ in rows:
                (images_dir / filename).write_bytes(b"fake image")
            pd.DataFrame(
                [
                    {
                        "patient_id": f"p{index}",
                        "lesion_id": f"l{index}",
                        "img_id": filename,
                        "diagnostic": diagnostic,
                    }
                    for index, (filename, diagnostic) in enumerate(rows, start=1)
                ]
            ).to_csv(raw_dir / "metadata.csv", index=False)
            out_path = root / "external_splits" / "pad_ufes.csv"

            written = prepare(raw_dir, out_path)
            split_rows = pd.read_csv(out_path)
            excluded_counts = json.loads(
                out_path.with_suffix(".excluded.json").read_text(encoding="utf-8")
            )

        self.assertEqual(list(split_rows.columns), ["split", "image_path", "label"])
        self.assertEqual(len(written), 4)
        self.assertEqual(
            sorted(split_rows["label"].tolist()),
            [
                "actinic_keratosis",
                "basal_cell_carcinoma",
                "melanoma",
                "nevus",
            ],
        )
        self.assertEqual(set(split_rows["split"]), {"test"})
        self.assertEqual(excluded_counts, {"BOD": 1, "BOW": 1, "SCC": 1, "SEK": 1})

    def test_prepare_native_mode_keeps_pad_ufes_six_class_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "pad_ufes"
            images_dir = raw_dir / "images"
            images_dir.mkdir(parents=True)
            rows = [
                ("mel.png", "MEL"),
                ("nev.png", "NEV"),
                ("bcc.png", "BCC"),
                ("ack.png", "ACK"),
                ("scc.png", "SCC"),
                ("sek.png", "SEK"),
                ("bod.png", "BOD"),
                ("bow.png", "BOW"),
            ]
            for filename, _ in rows:
                (images_dir / filename).write_bytes(b"fake image")
            pd.DataFrame(
                [
                    {
                        "patient_id": f"p{index}",
                        "lesion_id": f"l{index}",
                        "img_id": filename,
                        "diagnostic": diagnostic,
                    }
                    for index, (filename, diagnostic) in enumerate(rows, start=1)
                ]
            ).to_csv(raw_dir / "metadata.csv", index=False)
            out_path = root / "external_splits" / "pad_ufes_native.csv"

            written = prepare(raw_dir, out_path, label_mode="native")
            split_rows = pd.read_csv(out_path)
            excluded_counts = json.loads(
                out_path.with_suffix(".excluded.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(written), 6)
        self.assertEqual(
            sorted(split_rows["label"].tolist()),
            [
                "actinic_keratosis",
                "basal_cell_carcinoma",
                "melanoma",
                "nevus",
                "seborrheic_keratosis",
                "squamous_cell_carcinoma",
            ],
        )
        self.assertEqual(excluded_counts, {"BOD": 1, "BOW": 1})

    def test_prepare_missing_image_raises_file_not_found_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "pad_ufes"
            raw_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "patient_id": "p1",
                        "lesion_id": "l1",
                        "img_id": "missing.png",
                        "diagnostic": "MEL",
                    }
                ]
            ).to_csv(raw_dir / "metadata.csv", index=False)

            with self.assertRaises(FileNotFoundError):
                prepare(raw_dir, root / "pad_ufes.csv")


if __name__ == "__main__":
    unittest.main()

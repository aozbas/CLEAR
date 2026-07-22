from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.evaluation.prepare_open_images_ood import (
    cohort_fingerprint,
    collect_candidates,
    stable_candidate_order,
)

BBOX_COLUMNS = (
    "ImageID",
    "Source",
    "LabelName",
    "Confidence",
    "XMin",
    "XMax",
    "YMin",
    "YMax",
    "IsOccluded",
    "IsTruncated",
    "IsGroupOf",
    "IsDepiction",
    "IsInside",
)


def _bbox(image_id: str, label: str, **overrides: str) -> dict[str, str]:
    row = {
        "ImageID": image_id,
        "Source": "xclick",
        "LabelName": label,
        "Confidence": "1",
        "XMin": "0.0",
        "XMax": "0.8",
        "YMin": "0.0",
        "YMax": "0.8",
        "IsOccluded": "0",
        "IsTruncated": "0",
        "IsGroupOf": "0",
        "IsDepiction": "0",
        "IsInside": "0",
    }
    row.update(overrides)
    return row


class PrepareOpenImagesOodTests(unittest.TestCase):
    def test_candidate_collection_requires_large_literal_nonhuman_objects(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "boxes.csv"
            rows = [
                _bbox("keep", "target"),
                _bbox("human", "target"),
                _bbox("human", "person"),
                _bbox("small", "target", XMax="0.1", YMax="0.1"),
                _bbox("depiction", "target", IsDepiction="1"),
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=BBOX_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            candidates = collect_candidates(
                path,
                target_ids=["target"],
                excluded_ids=["person"],
            )

        self.assertEqual(candidates, {"target": {"keep"}})

    def test_candidate_order_is_reproducible_and_context_specific(self) -> None:
        image_ids = {"a", "b", "c", "d"}
        first = stable_candidate_order(image_ids, partition="calibration", class_name="Chair")
        second = stable_candidate_order(image_ids, partition="calibration", class_name="Chair")
        evaluation = stable_candidate_order(image_ids, partition="evaluation", class_name="Chair")

        self.assertEqual(first, second)
        self.assertNotEqual(first, evaluation)

    def test_cohort_fingerprint_ignores_paths_and_attribution_text(self) -> None:
        row = {
            "partition": "calibration",
            "semantic_group": "vehicles",
            "class_name": "Car",
            "image_id": "abc",
            "image_path": "/private/one.jpg",
            "license": "https://creativecommons.org/licenses/by/2.0/",
            "author": "First Author",
            "title": "First title",
            "sha256": "a" * 64,
        }
        changed = dict(
            row,
            image_path="D:\\different\\one.jpg",
            author="Changed Author",
            title="Changed title",
        )

        self.assertEqual(cohort_fingerprint([row]), cohort_fingerprint([changed]))


if __name__ == "__main__":
    unittest.main()

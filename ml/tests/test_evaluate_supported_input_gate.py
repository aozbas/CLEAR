from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ml.evaluation.evaluate_supported_input_gate import (
    InputRecord,
    choose_strictest_threshold,
    select_gate,
)


def _records() -> list[InputRecord]:
    records = []
    for source in ("pad_ufes", "hiba"):
        records.extend(
            InputRecord(
                path=Path(f"/{source}-{index}.jpg"),
                partition="calibration",
                kind="positive",
                group=source,
                identity=f"{source}:{index}",
            )
            for index in range(20)
        )
    records.extend(
        InputRecord(
            path=Path(f"/negative-{index}.jpg"),
            partition="calibration",
            kind="negative",
            group="objects" if index < 10 else "animals",
            identity=f"negative:{index}",
        )
        for index in range(20)
    )
    return records


class EvaluateSupportedInputGateTests(unittest.TestCase):
    def test_threshold_includes_quality_rejections_in_retention_target(self) -> None:
        score = np.arange(20, dtype=np.float64)
        quality = np.ones(20, dtype=bool)
        quality[0] = False
        mask = np.ones(20, dtype=bool)

        threshold = choose_strictest_threshold(
            score,
            quality,
            [mask],
            minimum_retention=0.95,
        )

        self.assertEqual(threshold, 1.0)

    def test_selection_uses_calibration_only_and_prefers_lowest_false_accept(self) -> None:
        records = _records()
        quality = np.ones(len(records), dtype=bool)
        positive = np.asarray([record.kind == "positive" for record in records])
        scores = {
            "logsumexp": np.where(positive, 10.0, 0.0),
            "max_logit": np.where(positive, 9.0, 9.0),
            "maximum_softmax_probability": np.where(positive, 0.9, 0.9),
        }

        method, threshold, candidates = select_gate(records, quality, scores)

        self.assertEqual(method, "logsumexp")
        self.assertEqual(threshold, 10.0)
        self.assertEqual(candidates["logsumexp"]["negative_false_accept_rate"], 0.0)

    def test_selection_uses_frozen_method_preference_for_false_accept_ties(self) -> None:
        records = _records()
        quality = np.ones(len(records), dtype=bool)
        positive = np.asarray([record.kind == "positive" for record in records])
        scores = {
            "logsumexp": np.where(positive, 10.0, 0.0),
            "max_logit": np.where(positive, 9.0, -1.0),
            "maximum_softmax_probability": np.where(positive, 0.9, 0.1),
        }

        method, _, _ = select_gate(records, quality, scores)

        self.assertEqual(method, "logsumexp")


if __name__ == "__main__":
    unittest.main()

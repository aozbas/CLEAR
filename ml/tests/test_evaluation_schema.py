import unittest

from ml.evaluation.schema import HAM10000_LABELS, ModelPrediction


class EvaluationSchemaTests(unittest.TestCase):
    def test_canonical_labels_match_ham10000_order(self) -> None:
        self.assertEqual(
            HAM10000_LABELS,
            (
                "melanoma",
                "nevus",
                "basal_cell_carcinoma",
                "actinic_keratosis",
                "benign_keratosis",
                "dermatofibroma",
                "vascular_lesion",
            ),
        )

    def test_prediction_accepts_known_label_and_confidence(self) -> None:
        prediction = ModelPrediction(label="nevus", confidence=0.85)

        self.assertEqual(prediction.label, "nevus")
        self.assertEqual(prediction.confidence, 0.85)

    def test_prediction_rejects_unknown_label(self) -> None:
        with self.assertRaises(ValueError):
            ModelPrediction(label="squamous_cell_carcinoma", confidence=0.5)

    def test_prediction_rejects_confidence_outside_probability_range(self) -> None:
        for confidence in (-0.01, 1.01):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    ModelPrediction(label="nevus", confidence=confidence)

    def test_prediction_probability_map_uses_canonical_labels(self) -> None:
        prediction = ModelPrediction(
            label="nevus",
            confidence=0.7,
            probabilities={"nevus": 0.7, "melanoma": 0.3},
        )

        self.assertEqual(prediction.probabilities["nevus"], 0.7)

    def test_prediction_rejects_unknown_probability_label(self) -> None:
        with self.assertRaises(ValueError):
            ModelPrediction(
                label="nevus",
                confidence=0.7,
                probabilities={"unknown": 0.3},
            )

    def test_prediction_rejects_invalid_probability_value(self) -> None:
        with self.assertRaises(ValueError):
            ModelPrediction(
                label="nevus",
                confidence=0.7,
                probabilities={"nevus": 1.2},
            )

    def test_prediction_rejects_negative_latency(self) -> None:
        with self.assertRaises(ValueError):
            ModelPrediction(label="nevus", confidence=0.7, latency_ms=-1.0)


if __name__ == "__main__":
    unittest.main()

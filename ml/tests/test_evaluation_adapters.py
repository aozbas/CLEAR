import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ml.evaluation.adapters.base import LesionModelAdapter
from ml.evaluation.adapters.baseline import BaselineAdapter
from ml.evaluation.schema import HAM10000_LABELS, ModelMetadata, ModelPrediction


class FakeAdapter:
    metadata = ModelMetadata(
        name="fake",
        source="test",
        adapter="fake",
        revision=None,
        license=None,
        labels=list(HAM10000_LABELS),
        notes=[],
    )

    def predict_image(self, image_path: Path) -> ModelPrediction:
        return ModelPrediction(label="nevus", confidence=0.9)


class EvaluationAdapterTests(unittest.TestCase):
    def test_fake_adapter_satisfies_protocol(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.jpg"
            image_path.write_bytes(b"not-used")
            adapter: LesionModelAdapter = FakeAdapter()

            prediction = adapter.predict_image(image_path)

        self.assertEqual(prediction.label, "nevus")
        self.assertEqual(prediction.confidence, 0.9)

    def test_baseline_adapter_uses_bytes_based_inference(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.jpg"
            image_path.write_bytes(b"image-bytes")
            adapter = BaselineAdapter()

            with patch("ml.evaluation.adapters.baseline.predict") as predict:
                predict.return_value = {"label": "nevus", "confidence": 0.88}

                prediction = adapter.predict_image(image_path)

        predict.assert_called_once_with(b"image-bytes", model_path=None)
        self.assertEqual(prediction.label, "nevus")
        self.assertEqual(prediction.confidence, 0.88)
        self.assertIsInstance(prediction.latency_ms, float)
        self.assertGreaterEqual(prediction.latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()

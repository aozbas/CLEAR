import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from ml.evaluation.adapters.base import LesionModelAdapter
from ml.evaluation.adapters.baseline import BaselineAdapter
from ml.evaluation.adapters.huggingface_image_classifier import (
    HuggingFaceImageClassifierAdapter,
)
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
    def _write_image(self, path: Path) -> None:
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)

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

    def test_huggingface_adapter_passes_model_revision_and_cache(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "cache"
            processor = object()
            model = FakeModel()

            with (
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier.AutoImageProcessor"
                    ".from_pretrained",
                    return_value=processor,
                ) as load_processor,
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier."
                    "AutoModelForImageClassification.from_pretrained",
                    return_value=model,
                ) as load_model,
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier.pipeline",
                    return_value=FakeHuggingFaceClassifier(),
                ) as pipeline,
            ):
                HuggingFaceImageClassifierAdapter(
                    model_id="example/model",
                    revision="abc123",
                    label_map={"mel": "melanoma", "nv": "nevus"},
                    cache_dir=cache_dir,
                )

        load_processor.assert_called_once_with(
            "example/model",
            revision="abc123",
            cache_dir=cache_dir,
        )
        load_model.assert_called_once_with(
            "example/model",
            revision="abc123",
            cache_dir=cache_dir,
        )
        pipeline.assert_called_once_with(
            "image-classification",
            model=model,
            image_processor=processor,
        )

    def test_huggingface_adapter_maps_labels_and_records_latency(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.jpg"
            self._write_image(image_path)

            with (
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier.AutoImageProcessor"
                    ".from_pretrained",
                    return_value=object(),
                ),
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier."
                    "AutoModelForImageClassification.from_pretrained",
                    return_value=FakeModel(),
                ),
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier.pipeline",
                    return_value=FakeHuggingFaceClassifier(),
                ),
            ):
                adapter = HuggingFaceImageClassifierAdapter(
                    model_id="example/model",
                    revision="abc123",
                    label_map={"mel": "melanoma", "nv": "nevus"},
                    cache_dir=Path(tmp_dir) / "cache",
                )
                prediction = adapter.predict_image(image_path)

        self.assertEqual(prediction.label, "melanoma")
        self.assertEqual(prediction.confidence, 0.9)
        self.assertEqual(prediction.probabilities, {"melanoma": 0.9, "nevus": 0.1})
        self.assertIsInstance(prediction.latency_ms, float)
        self.assertGreaterEqual(prediction.latency_ms, 0.0)
        self.assertEqual(adapter.metadata.revision, "resolved456")

    def test_huggingface_adapter_rejects_unknown_candidate_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.jpg"
            self._write_image(image_path)

            with (
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier.AutoImageProcessor"
                    ".from_pretrained",
                    return_value=object(),
                ),
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier."
                    "AutoModelForImageClassification.from_pretrained",
                    return_value=FakeModel(),
                ),
                patch(
                    "ml.evaluation.adapters.huggingface_image_classifier.pipeline",
                    return_value=FakeHuggingFaceClassifier(outputs=[{"label": "other", "score": 1.0}]),
                ),
            ):
                adapter = HuggingFaceImageClassifierAdapter(
                    model_id="example/model",
                    revision="abc123",
                    label_map={"mel": "melanoma"},
                    cache_dir=Path(tmp_dir) / "cache",
                )

                with self.assertRaises(ValueError):
                    adapter.predict_image(image_path)


class FakeConfig:
    _commit_hash = "resolved456"


class FakeModel:
    config = FakeConfig()


class FakeHuggingFaceClassifier:
    model = FakeModel()

    def __init__(self, outputs: list[dict[str, float | str]] | None = None) -> None:
        self.outputs = outputs or [
            {"label": "mel", "score": 0.9},
            {"label": "nv", "score": 0.1},
        ]

    def __call__(self, image: Image.Image, *, top_k: int | None) -> list[dict[str, float | str]]:
        return self.outputs


if __name__ == "__main__":
    unittest.main()

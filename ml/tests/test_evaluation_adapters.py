import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

from ml.evaluation.adapters import keras_h5
from ml.evaluation.adapters.base import LesionModelAdapter
from ml.evaluation.adapters.baseline import BaselineAdapter
from ml.evaluation.adapters.huggingface_image_classifier import (
    HuggingFaceImageClassifierAdapter,
)
from ml.evaluation.adapters.keras_h5 import KerasH5Adapter
from ml.evaluation.adapters.zero_shot import OpenClipZeroShotAdapter, TransformersZeroShotAdapter
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
                    return_value=FakeHuggingFaceClassifier(
                        outputs=[{"label": "other", "score": 1.0}]
                    ),
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

    def test_keras_h5_adapter_downloads_loads_and_maps_probabilities(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir = root / "cache"
            expected_cache_dir = cache_dir
            model_path = root / "model.h5"
            image_path = root / "image.jpg"
            self._write_image(image_path)

            def fake_download(
                *,
                repo_id: str,
                filename: str,
                revision: str,
                cache_dir: Path,
            ) -> str:
                self.assertEqual(repo_id, "example/keras")
                self.assertEqual(filename, "model.h5")
                self.assertEqual(revision, "abc123")
                self.assertEqual(cache_dir, expected_cache_dir)
                model_path.write_bytes(b"fake-model")
                return str(model_path)

            adapter = KerasH5Adapter(
                model_id="example/keras",
                revision="abc123",
                artifact_filename="model.h5",
                label_map={
                    "akiec": "actinic_keratosis",
                    "bcc": "basal_cell_carcinoma",
                    "nv": "nevus",
                },
                cache_dir=cache_dir,
                load_model=lambda path: FakeKerasModel(path),
                download_file=fake_download,
                license_name="mit",
            )

            prediction = adapter.predict_image(image_path)

        self.assertEqual(prediction.label, "basal_cell_carcinoma")
        self.assertEqual(prediction.confidence, 0.7)
        self.assertEqual(
            prediction.probabilities,
            {
                "actinic_keratosis": 0.1,
                "basal_cell_carcinoma": 0.7,
                "nevus": 0.2,
            },
        )
        self.assertIsInstance(prediction.latency_ms, float)
        self.assertGreaterEqual(prediction.latency_ms, 0.0)
        self.assertEqual(adapter.metadata.adapter, "keras_h5")
        self.assertEqual(adapter.metadata.revision, "abc123")

    def test_keras_h5_adapter_rejects_prediction_count_mismatches(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_path = root / "model.h5"
            image_path = root / "image.jpg"
            self._write_image(image_path)

            adapter = KerasH5Adapter(
                model_id="example/keras",
                revision="abc123",
                artifact_filename="model.h5",
                label_map={
                    "akiec": "actinic_keratosis",
                    "bcc": "basal_cell_carcinoma",
                    "nv": "nevus",
                },
                cache_dir=root / "cache",
                load_model=lambda path: FakeKerasModel(path, outputs=[[0.4, 0.6]]),
                download_file=lambda **_: str(model_path),
            )

            with self.assertRaises(ValueError):
                adapter.predict_image(image_path)

    def test_keras_h5_default_loader_uses_legacy_depthwise_compatibility(self) -> None:
        models_module = FakeKerasModelsModule()
        layers_module = FakeKerasLayersModule()

        def fake_import_module(name: str):
            if name == "tensorflow.keras.models":
                return models_module
            if name == "tensorflow.keras.layers":
                return layers_module
            raise ModuleNotFoundError(name)

        with patch("ml.evaluation.adapters.keras_h5.importlib.import_module", fake_import_module):
            loaded = keras_h5._load_model(Path("model.h5"))

        self.assertEqual(loaded, "loaded-model")
        self.assertEqual(models_module.path, Path("model.h5"))
        self.assertFalse(models_module.compile)
        self.assertIn("DepthwiseConv2D", models_module.custom_objects)

        compat_layer = models_module.custom_objects["DepthwiseConv2D"].from_config(
            {"name": "depthwise", "groups": 1}
        )
        self.assertIsInstance(compat_layer, FakeDepthwiseConv2D)
        self.assertEqual(type(compat_layer).received_config, {"name": "depthwise"})

    def test_open_clip_zero_shot_adapter_scores_canonical_prompts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            self._write_image(image_path)

            adapter = OpenClipZeroShotAdapter(
                model_id="example/openclip",
                revision="abc123",
                cache_dir=root / "cache",
                license_name="mit",
                open_clip_module=FakeOpenClipModule(),
                torch_module=torch,
                device="cpu",
            )
            prediction = adapter.predict_image(image_path)

        self.assertEqual(prediction.label, "melanoma")
        self.assertGreater(prediction.confidence, 0.5)
        self.assertEqual(set(prediction.probabilities or {}), set(HAM10000_LABELS))
        self.assertEqual(adapter.metadata.adapter, "open_clip_zero_shot")
        self.assertEqual(adapter.metadata.revision, "abc123")

    def test_transformers_zero_shot_adapter_scores_canonical_prompts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            self._write_image(image_path)

            adapter = TransformersZeroShotAdapter(
                model_id="example/medsiglip",
                revision="abc123",
                cache_dir=root / "cache",
                license_name="other",
                processor_loader=lambda *_, **__: FakeZeroShotProcessor(),
                model_loader=lambda *_, **__: FakeTransformersZeroShotModel(),
                torch_module=torch,
                device="cpu",
            )
            prediction = adapter.predict_image(image_path)

        self.assertEqual(prediction.label, "melanoma")
        self.assertGreater(prediction.confidence, 0.5)
        self.assertEqual(set(prediction.probabilities or {}), set(HAM10000_LABELS))
        self.assertEqual(adapter.metadata.adapter, "transformers_zero_shot")


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


class FakeKerasModel:
    input_shape = (None, 4, 6, 3)

    def __init__(self, path: str, outputs: list[list[float]] | None = None) -> None:
        self.path = path
        self.outputs = outputs or [[0.1, 0.7, 0.2]]

    def predict(self, batch, *, verbose: int = 0):
        self.batch_shape = batch.shape
        self.verbose = verbose
        return self.outputs


class FakeKerasModelsModule:
    def load_model(self, path: Path, *, compile: bool, custom_objects: dict[str, object]):
        self.path = path
        self.compile = compile
        self.custom_objects = custom_objects
        return "loaded-model"


class FakeKerasLayersModule:
    DepthwiseConv2D = None


class FakeDepthwiseConv2D:
    received_config: dict[str, object] | None = None

    @classmethod
    def from_config(cls, config: dict[str, object]):
        cls.received_config = config
        return cls()


FakeKerasLayersModule.DepthwiseConv2D = FakeDepthwiseConv2D


class FakeOpenClipModule:
    def create_model_and_transforms(self, model_name: str, *, cache_dir: Path):
        self.model_name = model_name
        self.cache_dir = cache_dir
        return FakeOpenClipModel(), None, lambda image: torch.ones(3, 2, 2)

    def get_tokenizer(self, model_name: str):
        self.tokenizer_model_name = model_name
        return lambda prompts: torch.zeros(len(prompts), 2)


class FakeOpenClipModel:
    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def encode_image(self, image):
        return torch.tensor([[1.0, 0.0]])

    def encode_text(self, text):
        features = torch.zeros(text.shape[0], 2)
        features[0, 0] = 1.0
        if text.shape[0] > 1:
            features[1:, 1] = 1.0
        return features


class FakeBatch(dict):
    def to(self, device: str):
        self.device = device
        return self


class FakeZeroShotProcessor:
    def __call__(self, *, text, images, padding: str, return_tensors: str):
        self.text = text
        self.images = images
        self.padding = padding
        self.return_tensors = return_tensors
        return FakeBatch(input_ids=torch.zeros(len(text), 2))


class FakeTransformersZeroShotModel:
    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, **kwargs):
        logits = torch.zeros(1, len(HAM10000_LABELS))
        logits[0, 0] = 4.0
        return type("FakeOutputs", (), {"logits_per_image": logits})()


if __name__ == "__main__":
    unittest.main()

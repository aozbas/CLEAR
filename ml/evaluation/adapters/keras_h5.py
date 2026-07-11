"""Keras `.h5` adapter for evaluation-only model runs."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from ml.evaluation.schema import HAM10000_LABELS, ModelMetadata, ModelPrediction, validate_label

DownloadFile = Callable[..., str]
LoadModel = Callable[[Path], Any]


class KerasH5Adapter:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str = "main",
        artifact_filename: str,
        label_map: Mapping[str, str],
        cache_dir: Path | None = None,
        license_name: str | None = None,
        load_model: LoadModel | None = None,
        download_file: DownloadFile | None = None,
    ) -> None:
        if not label_map:
            raise ValueError("label_map must include at least one candidate label.")
        if not artifact_filename:
            raise ValueError("artifact_filename is required for Keras .h5 candidates.")

        for canonical_label in label_map.values():
            validate_label(canonical_label)

        self.model_id = model_id
        self.revision = revision
        self.artifact_filename = artifact_filename
        self.label_map = dict(label_map)
        self.cache_dir = cache_dir

        downloader = download_file or hf_hub_download
        self.model_path = Path(
            downloader(
                repo_id=model_id,
                filename=artifact_filename,
                revision=revision,
                cache_dir=cache_dir,
            )
        )
        loader = load_model or _load_model
        self._model = loader(self.model_path)
        self._target_size = _target_size_from_model(self._model)
        self.metadata = ModelMetadata(
            name=model_id,
            source=f"https://huggingface.co/{model_id}",
            adapter="keras_h5",
            revision=revision,
            license=license_name,
            labels=self._canonical_labels(),
            notes=["Keras .h5 model used for experimental classification evaluation only."],
        )

    def predict_image(self, image_path: Path) -> ModelPrediction:
        batch = self._preprocess_image(image_path)
        started = time.perf_counter()
        raw_outputs = self._model.predict(batch, verbose=0)
        latency_ms = (time.perf_counter() - started) * 1000

        probabilities = self._map_outputs(raw_outputs)
        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        return ModelPrediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
        )

    def _canonical_labels(self) -> list[str]:
        labels = set(self.label_map.values())
        return [label for label in HAM10000_LABELS if label in labels]

    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        height, width = self._target_size
        with Image.open(image_path) as image:
            resized = image.convert("RGB").resize((width, height))
            array = np.asarray(resized, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def _map_outputs(self, raw_outputs: Any) -> dict[str, float]:
        values = np.asarray(raw_outputs, dtype=np.float64)
        if values.ndim == 2:
            if values.shape[0] != 1:
                raise ValueError(
                    f"Model {self.model_id} returned {values.shape[0]} batches for one image."
                )
            values = values[0]
        if values.ndim != 1:
            raise ValueError(f"Model {self.model_id} returned unsupported output shape.")

        class_names = list(self.label_map)
        if len(values) != len(class_names):
            raise ValueError(
                f"Model {self.model_id} returned {len(values)} scores for "
                f"{len(class_names)} labels."
            )

        probabilities = _probability_values(values)
        return {
            self.label_map[class_name]: float(probability)
            for class_name, probability in zip(class_names, probabilities, strict=True)
        }


def _load_model(model_path: Path) -> Any:
    try:
        models = importlib.import_module("tensorflow.keras.models")
        layers = importlib.import_module("tensorflow.keras.layers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required for Keras .h5 evaluation. "
            "Install the optional evaluation runtime first."
        ) from exc

    custom_objects = {
        "DepthwiseConv2D": _legacy_depthwise_conv2d(layers.DepthwiseConv2D),
    }
    return models.load_model(model_path, compile=False, custom_objects=custom_objects)


def _legacy_depthwise_conv2d(depthwise_conv2d: type[Any]) -> type[Any]:
    class LegacyDepthwiseConv2D(depthwise_conv2d):
        @classmethod
        def from_config(cls, config: dict[str, Any]) -> Any:
            config = dict(config)
            config.pop("groups", None)
            return super().from_config(config)

    return LegacyDepthwiseConv2D


def _target_size_from_model(model: Any) -> tuple[int, int]:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0] if input_shape else None
    if input_shape is None or len(input_shape) < 3:
        return (224, 224)

    height = input_shape[1]
    width = input_shape[2]
    if height is None or width is None:
        return (224, 224)
    return (int(height), int(width))


def _probability_values(values: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError("Model returned non-finite classification scores.")

    if np.any(values < 0.0) or np.any(values > 1.0):
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    total = float(np.sum(values))
    if math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        return values
    if total <= 0.0:
        raise ValueError("Model returned no positive classification scores.")
    return values / total

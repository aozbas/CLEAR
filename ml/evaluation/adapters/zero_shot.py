"""Zero-shot image-text adapters for evaluation-only foundation models."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from PIL import Image

from ml.evaluation.schema import HAM10000_LABELS, ModelMetadata, ModelPrediction

DownloadSnapshot = Callable[..., str]

LABEL_PROMPTS = {
    "melanoma": "a clinical skin lesion image of melanoma",
    "nevus": "a clinical skin lesion image of a nevus",
    "basal_cell_carcinoma": "a clinical skin lesion image of basal cell carcinoma",
    "actinic_keratosis": "a clinical skin lesion image of actinic keratosis",
    "benign_keratosis": "a clinical skin lesion image of benign keratosis",
    "dermatofibroma": "a clinical skin lesion image of dermatofibroma",
    "vascular_lesion": "a clinical skin lesion image of a vascular lesion",
    "squamous_cell_carcinoma": "a clinical skin lesion image of squamous cell carcinoma",
    "seborrheic_keratosis": "a clinical skin lesion image of seborrheic keratosis",
}


class OpenClipZeroShotAdapter:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str = "main",
        cache_dir: Path | None = None,
        license_name: str | None = None,
        open_clip_module: Any | None = None,
        torch_module: Any | None = None,
        device: str | None = None,
        labels: tuple[str, ...] = HAM10000_LABELS,
        download_snapshot: DownloadSnapshot | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self._labels = labels
        self._torch = torch_module or _import_optional("torch", "torch")
        self._open_clip = open_clip_module or _import_optional("open_clip", "open_clip_torch")
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        downloader = download_snapshot or snapshot_download
        snapshot_path = Path(
            downloader(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
            )
        )
        local_model_id = f"local-dir:{snapshot_path}"

        model, _, preprocess = self._open_clip.create_model_and_transforms(
            local_model_id,
            cache_dir=cache_dir,
        )
        self._model = model.to(self.device)
        self._model.eval()
        self._preprocess = preprocess
        self._tokenizer = self._open_clip.get_tokenizer(local_model_id)
        self._prompts = _canonical_prompts(self._labels)
        self.metadata = ModelMetadata(
            name=model_id,
            source=f"https://huggingface.co/{model_id}",
            adapter="open_clip_zero_shot",
            revision=revision,
            license=license_name,
            labels=list(self._labels),
            notes=["OpenCLIP zero-shot adapter used for experimental classification."],
        )

    def predict_image(self, image_path: Path) -> ModelPrediction:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_tensor = self._preprocess(rgb_image).unsqueeze(0).to(self.device)
            text_tensor = self._move_to_device(self._tokenizer(self._prompts))
            started = time.perf_counter()
            with self._torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(text_tensor)
                image_features = _normalize(self._torch, image_features)
                text_features = _normalize(self._torch, text_features)
                scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            latency_ms = (time.perf_counter() - started) * 1000

        return _prediction_from_scores(scores[0], labels=self._labels, latency_ms=latency_ms)

    def _move_to_device(self, value: Any) -> Any:
        return value.to(self.device) if hasattr(value, "to") else value


class TransformersZeroShotAdapter:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str = "main",
        cache_dir: Path | None = None,
        license_name: str | None = None,
        processor_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
        torch_module: Any | None = None,
        device: str | None = None,
        labels: tuple[str, ...] = HAM10000_LABELS,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self._labels = labels
        self._torch = torch_module or _import_optional("torch", "torch")
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        processor_loader = processor_loader or _transformers_processor_loader()
        model_loader = model_loader or _transformers_model_loader()
        self._processor = processor_loader(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        self._model = model_loader(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        ).to(self.device)
        self._model.eval()
        self._prompts = _canonical_prompts(self._labels)
        self.metadata = ModelMetadata(
            name=model_id,
            source=f"https://huggingface.co/{model_id}",
            adapter="transformers_zero_shot",
            revision=revision,
            license=license_name,
            labels=list(self._labels),
            notes=["Transformers zero-shot adapter used for experimental classification."],
        )

    def predict_image(self, image_path: Path) -> ModelPrediction:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            inputs = self._processor(
                text=self._prompts,
                images=rgb_image,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = self._move_to_device(inputs)
            started = time.perf_counter()
            with self._torch.no_grad():
                outputs = self._model(**inputs)
                logits = getattr(outputs, "logits_per_image", None)
                if logits is None:
                    raise ValueError(f"Model {self.model_id} did not return logits_per_image.")
                scores = self._torch.softmax(logits, dim=-1)
            latency_ms = (time.perf_counter() - started) * 1000

        return _prediction_from_scores(scores[0], labels=self._labels, latency_ms=latency_ms)

    def _move_to_device(self, value: Any) -> Any:
        return value.to(self.device) if hasattr(value, "to") else value


def _canonical_prompts(labels: tuple[str, ...]) -> list[str]:
    return [LABEL_PROMPTS[label] for label in labels]


def _prediction_from_scores(
    scores: Any,
    *,
    labels: tuple[str, ...],
    latency_ms: float,
) -> ModelPrediction:
    probabilities = {
        label: float(scores[index].detach().cpu()) for index, label in enumerate(labels)
    }
    label, confidence = max(probabilities.items(), key=lambda item: item[1])
    return ModelPrediction(
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        latency_ms=latency_ms,
        labels=labels,
    )


def _normalize(torch_module: Any, features: Any) -> Any:
    epsilon = torch_module.finfo(features.dtype).eps
    return features / features.norm(dim=-1, keepdim=True).clamp(min=epsilon)


def _import_optional(module_name: str, package_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Optional dependency {package_name} is required for this evaluation adapter."
        ) from exc


def _transformers_processor_loader() -> Callable[..., Any]:
    transformers = _import_optional("transformers", "transformers")
    return transformers.AutoProcessor.from_pretrained


def _transformers_model_loader() -> Callable[..., Any]:
    transformers = _import_optional("transformers", "transformers")
    model_loader = getattr(transformers, "AutoModelForZeroShotImageClassification", None)
    if model_loader is not None:
        return model_loader.from_pretrained
    return transformers.AutoModel.from_pretrained

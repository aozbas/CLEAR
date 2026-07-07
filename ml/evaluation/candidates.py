"""Static candidate registry for evaluation-only model comparison."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateModel:
    name: str
    source_url: str
    adapter_type: str
    label_strategy: str
    notes: list[str] = field(default_factory=list)
    label_map: dict[str, str] = field(default_factory=dict)
    revision: str = "main"
    license: str | None = None


CANDIDATES: dict[str, CandidateModel] = {
    "baseline": CandidateModel(
        name="baseline",
        source_url="local ml/inference/predict.py",
        adapter_type="baseline",
        label_strategy="native CLEAR HAM10000 labels",
        notes=["Current local checkpoint; no Hugging Face download required."],
    ),
    "Miguel764/efficientnetv2s-skin-cancer-classifier": CandidateModel(
        name="Miguel764/efficientnetv2s-skin-cancer-classifier",
        source_url="https://huggingface.co/Miguel764/efficientnetv2s-skin-cancer-classifier",
        adapter_type="huggingface_image_classifier",
        label_strategy="inspect config labels before evaluation",
        notes=["First-wave task-specific image-classification candidate."],
    ),
    "syaha/skin_cancer_detection_model": CandidateModel(
        name="syaha/skin_cancer_detection_model",
        source_url="https://huggingface.co/syaha/skin_cancer_detection_model",
        adapter_type="huggingface_image_classifier",
        label_strategy="inspect config labels before evaluation",
        notes=["First-wave task-specific image-classification candidate."],
    ),
    "google/medsiglip-448": CandidateModel(
        name="google/medsiglip-448",
        source_url="https://huggingface.co/google/medsiglip-448",
        adapter_type="zero_shot",
        label_strategy="canonical-label prompts for experimental classification",
        notes=["Foundation candidate; evaluate only if runtime dependencies stay modest."],
    ),
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": CandidateModel(
        name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        source_url=(
            "https://huggingface.co/microsoft/"
            "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        ),
        adapter_type="zero_shot",
        label_strategy="canonical-label prompts if dependency footprint is acceptable",
        notes=["Biomedical CLIP candidate; second choice after MedSigLIP."],
    ),
    "google/derm-foundation": CandidateModel(
        name="google/derm-foundation",
        source_url="https://huggingface.co/google/derm-foundation",
        adapter_type="inspection_only",
        label_strategy="specialized research runtime; inspect before any adapter work",
        notes=["Second-wave research candidate."],
    ),
    "redlessone/DermLIP_ViT-B-16": CandidateModel(
        name="redlessone/DermLIP_ViT-B-16",
        source_url="https://huggingface.co/redlessone/DermLIP_ViT-B-16",
        adapter_type="inspection_only",
        label_strategy="specialized research runtime; inspect before any adapter work",
        notes=["Second-wave research candidate."],
    ),
}


def get_candidate(name: str) -> CandidateModel:
    try:
        return CANDIDATES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc

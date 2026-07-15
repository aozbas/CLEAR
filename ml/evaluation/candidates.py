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
    training_datasets: list[str] = field(default_factory=list)
    clean_dataset_sources: list[str] = field(default_factory=list)
    revision: str = "main"
    license: str | None = None
    model_size_bytes: int | None = None
    artifact_filename: str | None = None


CANDIDATES: dict[str, CandidateModel] = {
    "baseline": CandidateModel(
        name="baseline",
        source_url="local ml/inference/predict.py",
        adapter_type="baseline",
        label_strategy="native CLEAR HAM10000 labels",
        notes=["Current local checkpoint; no Hugging Face download required."],
        training_datasets=["HAM10000"],
        clean_dataset_sources=["ham10000_internal"],
    ),
    "Miguel764/efficientnetv2s-skin-cancer-classifier": CandidateModel(
        name="Miguel764/efficientnetv2s-skin-cancer-classifier",
        source_url="https://huggingface.co/Miguel764/efficientnetv2s-skin-cancer-classifier",
        adapter_type="keras_h5",
        label_strategy="7-class HAM10000 labels documented in README; no config id2label",
        notes=[
            "First-wave task-specific candidate.",
            "Evaluation requires optional TensorFlow/Keras runtime.",
        ],
        label_map={
            "akiec": "actinic_keratosis",
            "bcc": "basal_cell_carcinoma",
            "bkl": "benign_keratosis",
            "df": "dermatofibroma",
            "mel": "melanoma",
            "nv": "nevus",
            "vasc": "vascular_lesion",
        },
        training_datasets=["HAM10000"],
        revision="97a25e6b71c4b426c259b747a6c49d235c2dade7",
        license="mit",
        model_size_bytes=247255104,
        artifact_filename="efficientnetv2s.h5",
    ),
    "syaha/skin_cancer_detection_model": CandidateModel(
        name="syaha/skin_cancer_detection_model",
        source_url="https://huggingface.co/syaha/skin_cancer_detection_model",
        adapter_type="keras_h5",
        label_strategy="7-class HAM10000 class_names in README/app.py",
        notes=[
            "First-wave task-specific candidate.",
            "Evaluation requires optional TensorFlow/Keras runtime.",
        ],
        label_map={
            "akiec": "actinic_keratosis",
            "bcc": "basal_cell_carcinoma",
            "bkl": "benign_keratosis",
            "df": "dermatofibroma",
            "nv": "nevus",
            "vasc": "vascular_lesion",
            "mel": "melanoma",
        },
        training_datasets=["HAM10000"],
        revision="c1c88efd59fa52c0adbbb5f6ebb3610744933938",
        license="mit",
        model_size_bytes=134084584,
        artifact_filename="skin_cancer_model.h5",
    ),
    "gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer": CandidateModel(
        name="gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer",
        source_url=(
            "https://huggingface.co/gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer"
        ),
        adapter_type="huggingface_image_classifier",
        label_strategy="Transformers id2label maps cleanly to CLEAR HAM10000 labels",
        notes=["Search-discovered compatible Transformers candidate for local evaluation."],
        label_map={
            "Actinic-keratoses": "actinic_keratosis",
            "Basal-cell-carcinoma": "basal_cell_carcinoma",
            "Benign-keratosis-like-lesions": "benign_keratosis",
            "Dermatofibroma": "dermatofibroma",
            "Melanocytic-nevi": "nevus",
            "Melanoma": "melanoma",
            "Vascular-lesions": "vascular_lesion",
        },
        training_datasets=["HAM10000"],
        revision="3b408dc64c66e7a39c86b87d2283146821a8be28",
        license="apache-2.0",
        model_size_bytes=110404975,
    ),
    "google/medsiglip-448": CandidateModel(
        name="google/medsiglip-448",
        source_url="https://huggingface.co/google/medsiglip-448",
        adapter_type="transformers_zero_shot",
        label_strategy="zero-shot clinical-photo prompts over CLEAR canonical labels",
        notes=[
            "Foundation candidate; evaluate only if runtime dependencies stay modest.",
            "Gated model with Health AI Developer Foundations terms.",
            "PAD-UFES is domain evidence, not a clean holdout, because it is listed "
            "as training data.",
        ],
        training_datasets=["PAD-UFES-20", "SCIN"],
        revision="9cea28a1a1195f665105faa6e8544c112fd960a4",
        license="other",
        model_size_bytes=3513309984,
    ),
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": CandidateModel(
        name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        source_url=(
            "https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        ),
        adapter_type="open_clip_zero_shot",
        label_strategy="zero-shot clinical-photo prompts over CLEAR canonical labels",
        notes=[
            "Biomedical CLIP candidate; broad medical-domain baseline.",
            "Requires optional OpenCLIP/timm evaluation runtime.",
        ],
        training_datasets=["PMC-15M"],
        revision="9f341de24bfb00180f1b847274256e9b65a3a32e",
        license="mit",
        model_size_bytes=783705670,
    ),
    "google/derm-foundation": CandidateModel(
        name="google/derm-foundation",
        source_url="https://huggingface.co/google/derm-foundation",
        adapter_type="embedding_linear_probe",
        label_strategy="embedding extraction plus local linear probe",
        notes=[
            "Legacy dermatology embedding model; not a direct zero-shot classifier.",
            "Requires a separate probe workflow before evaluation.",
        ],
    ),
    "redlessone/DermLIP_ViT-B-16": CandidateModel(
        name="redlessone/DermLIP_ViT-B-16",
        source_url="https://huggingface.co/redlessone/DermLIP_ViT-B-16",
        adapter_type="open_clip_zero_shot",
        label_strategy="zero-shot clinical-photo prompts over CLEAR canonical labels",
        notes=[
            "Dermatology CLIP-style model trained from Derm1M.",
            "Model card quick start uses OpenCLIP rather than Transformers for direct scoring.",
        ],
        training_datasets=["Derm1M"],
        revision="main",
        license="cc-by-4.0",
    ),
}


def get_candidate(name: str) -> CandidateModel:
    try:
        return CANDIDATES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc

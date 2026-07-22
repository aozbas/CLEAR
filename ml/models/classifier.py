import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("resnet18", "convnext_tiny")


def build_model(num_classes: int = 2, *, architecture: str = "resnet18") -> nn.Module:
    # Phase 1 default: 2 = binary (suspicious / non_suspicious).
    # Legacy Phase 2 checkpoints use seven-class ResNet18. The current demo uses
    # six-class ConvNeXt-Tiny; inference never downloads pretrained weights.
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported inference architecture: {architecture!r}")
    return model

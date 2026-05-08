import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2) -> nn.Module:
    # Phase 1 default: 2 = binary (suspicious / non_suspicious).
    # Phase 2: pass num_classes=7 for full HAM10000 canonical labels.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

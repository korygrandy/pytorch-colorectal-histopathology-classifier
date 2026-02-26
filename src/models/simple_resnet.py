"""A minimal ResNet-based classifier.

This is intentionally simpler than `ColorectalClassifier`: it swaps the final
fully-connected layer of a torchvision ResNet to match `num_classes`.

Use this when you want the most straightforward transfer-learning baseline.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


_SUPPORTED = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}


class SimpleResNetClassifier(nn.Module):
    """A simple ResNet backbone with a single linear classification head."""

    def __init__(
        self,
        num_classes: int,
        architecture: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()

        if architecture not in _SUPPORTED:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Choose one of: {', '.join(sorted(_SUPPORTED))}"
            )

        self.num_classes = num_classes
        self.architecture = architecture

        self.backbone = _SUPPORTED[architecture](pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_simple_resnet(
    num_classes: int,
    architecture: str = "resnet18",
    pretrained: bool = True,
    device: Optional[str] = None,
) -> SimpleResNetClassifier:
    """Factory helper mirroring `create_model` style used elsewhere in the repo."""

    model = SimpleResNetClassifier(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=pretrained,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return model.to(device)

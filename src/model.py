from __future__ import annotations

import torch.nn as nn
from torchvision import models


# ResNet-based model for deepfake detection
class ResNetDeepfake(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet50(weights=weights)

        # freeze all layer parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze the last block (layer4) parameters for fine-tuning!
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


def build_model(num_classes: int = 2, pretrained: bool = True):
    return ResNetDeepfake(num_classes=num_classes, pretrained=pretrained)
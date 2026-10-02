"""
Model definitions only — no training loop, no data loading, no UI.

Two models, both loadable from a state_dict checkpoint produced by
train.py:

- `EmotionCNN`   — the old from-scratch architecture (kept as the
                   baseline for comparison in the README), retrained
                   under the new pipeline (augmentation, class weights,
                   LR schedule) rather than reused as-is.
- `EmotionResNet` — the transfer-learning model actually used by the app.
                    Wraps a torchvision ResNet18 (ImageNet-pretrained),
                    with its final layer replaced for 7 emotion classes.

Input contract for both: a single-channel (grayscale) 48x48 face crop,
already normalized. `EmotionResNet` expects the 3-channel/224x224 resize
to have been done in the *transform* pipeline (see `data.py`), not here —
keeping the resize/channel-duplication as a data concern, not a model
concern, so both models can share one dataset class.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

NUM_CLASSES = 7
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


class EmotionCNN(nn.Module):
    """From-scratch baseline CNN (4x Conv-BN-ReLU-MaxPool-Dropout).

    Same architecture as the old project's model — kept unchanged so the
    README comparison against transfer learning is apples-to-apples
    (same architecture, only the *training pipeline* improved).
    Expects a (N, 1, 48, 48) input.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class EmotionResNet(nn.Module):
    """ImageNet-pretrained ResNet18, fine-tuned for 7 emotion classes.

    Expects a (N, 3, 224, 224) input (grayscale duplicated to 3 channels
    and resized — done in the data transform, not here). Only the final
    fully-connected layer is replaced/randomly initialized; the rest of
    the backbone starts from ImageNet weights.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Factory used by train.py/evaluate.py/app.py so they don't each
    hardcode the if/else between the two architectures."""
    if name == "cnn":
        return EmotionCNN(num_classes=num_classes)
    if name == "resnet":
        return EmotionResNet(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name!r} (expected 'cnn' or 'resnet')")

"""
Dataset + transforms only — no training loop, no downloading. Kept
separate and free of any network/kagglehub dependency so it's testable
against a tiny synthetic folder, without pulling the real ~35k-image
FER-2013 dataset.

Expected directory layout (what FER-2013 already comes as, e.g. the
Kaggle "msambare/fer2013" mirror):

    root/
      angry/*.jpg
      disgust/*.jpg
      fear/*.jpg
      happy/*.jpg
      neutral/*.jpg
      sad/*.jpg
      surprise/*.jpg
"""
from __future__ import annotations

import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model import EMOTION_LABELS

# FER-2013 folder names are lowercase; EMOTION_LABELS (used for display)
# are capitalized — keep the mapping explicit in one place.
CLASS_DIRS = [label.lower() for label in EMOTION_LABELS]

# ImageNet normalization stats — only used for the ResNet transform, since
# its backbone was pretrained on ImageNet-normalized inputs.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EmotionFolderDataset(Dataset):
    """One folder-per-class dataset of grayscale face crops."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for class_idx, class_name in enumerate(CLASS_DIRS):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {root_dir!r} (expected subfolders "
                f"named {CLASS_DIRS!r})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")  # grayscale
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> Counter:
        return Counter(label for _, label in self.samples)


def get_transform(model_name: str, train: bool, augment: bool = True) -> transforms.Compose:
    """Returns the right transform pipeline for a given model.

    `cnn`    -> 1x48x48, matches EmotionCNN's input contract.
    `resnet` -> 3x224x224, ImageNet-normalized, matches EmotionResNet's
                pretrained backbone.

    `augment=False` disables the random train-time augmentation (used for
    pipeline ablations); the deterministic resize is still applied so the
    model always gets the input size it expects.
    """
    use_augment = train and augment
    augment_ops = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(
            48 if model_name == "cnn" else 224,
            scale=(0.85, 1.0),
        ),
    ] if use_augment else []

    if model_name == "cnn":
        resize = [] if use_augment else [transforms.Resize((48, 48))]
        return transforms.Compose(
            resize + augment_ops + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    if model_name == "resnet":
        # RandomResizedCrop already outputs 224x224 when augmenting.
        resize = [] if use_augment else [transforms.Resize((224, 224))]
        return transforms.Compose(
            resize + augment_ops + [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unknown model name: {model_name!r} (expected 'cnn' or 'resnet')")


def compute_class_weights(dataset: EmotionFolderDataset) -> torch.Tensor:
    """Inverse-frequency class weights for `nn.CrossEntropyLoss(weight=...)`,
    normalized so the mean weight is 1.0 (keeps the loss scale comparable
    to the unweighted case).
    """
    counts = dataset.class_counts()
    n_classes = len(CLASS_DIRS)
    freqs = torch.tensor(
        [counts.get(i, 0) for i in range(n_classes)], dtype=torch.float32
    )
    freqs = freqs.clamp(min=1)  # avoid div-by-zero for an empty class
    weights = 1.0 / freqs
    weights = weights * (n_classes / weights.sum())
    return weights

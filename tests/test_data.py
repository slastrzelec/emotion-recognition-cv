import os

import numpy as np
import pytest
import torch
from PIL import Image

from data import EmotionFolderDataset, compute_class_weights, get_transform


@pytest.fixture
def fake_dataset_root(tmp_path):
    classes_counts = {
        "angry": 5, "disgust": 1, "fear": 3, "happy": 8,
        "neutral": 6, "sad": 4, "surprise": 2,
    }
    for cls, n in classes_counts.items():
        d = tmp_path / cls
        d.mkdir()
        for i in range(n):
            arr = (np.random.rand(48, 48) * 255).astype("uint8")
            Image.fromarray(arr, mode="L").save(d / f"{i}.jpg")
    return str(tmp_path), classes_counts


def test_dataset_loads_all_images(fake_dataset_root):
    root, counts = fake_dataset_root
    ds = EmotionFolderDataset(root, transform=get_transform("cnn", train=False))
    assert len(ds) == sum(counts.values())


def test_dataset_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmotionFolderDataset(str(tmp_path), transform=None)


def test_cnn_transform_shape(fake_dataset_root):
    root, _ = fake_dataset_root
    ds = EmotionFolderDataset(root, transform=get_transform("cnn", train=False))
    img, label = ds[0]
    assert img.shape == (1, 48, 48)
    assert isinstance(label, int)


def test_resnet_transform_shape(fake_dataset_root):
    root, _ = fake_dataset_root
    ds = EmotionFolderDataset(root, transform=get_transform("resnet", train=False))
    img, label = ds[0]
    assert img.shape == (3, 224, 224)


def test_class_weights_favor_rare_class(fake_dataset_root):
    root, counts = fake_dataset_root
    ds = EmotionFolderDataset(root, transform=get_transform("cnn", train=False))
    weights = compute_class_weights(ds)
    assert weights.shape == (7,)
    # "disgust" (index 1) has only 1 sample -> should get the highest weight
    disgust_idx = 1
    assert weights[disgust_idx] == weights.max()
    # weights should average out to roughly 1.0 (keeps loss scale comparable)
    assert abs(float(weights.mean()) - 1.0) < 1e-4


def test_get_transform_rejects_unknown_model():
    with pytest.raises(ValueError):
        get_transform("not-a-real-model", train=False)


@pytest.mark.parametrize("model_name,expected", [("cnn", (1, 48, 48)), ("resnet", (3, 224, 224))])
@pytest.mark.parametrize("augment", [True, False])
def test_train_transform_shape_with_and_without_augment(fake_dataset_root, model_name, expected, augment):
    root, _ = fake_dataset_root
    ds = EmotionFolderDataset(root, transform=get_transform(model_name, train=True, augment=augment))
    img, _ = ds[0]
    assert img.shape == expected

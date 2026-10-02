import torch
from torch.utils.data import DataLoader

from evaluate import evaluate
from model import EmotionCNN


def test_evaluate_returns_sane_shapes(tmp_path):
    import numpy as np
    from PIL import Image
    from data import EmotionFolderDataset, get_transform

    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    for cls in classes:
        d = tmp_path / cls
        d.mkdir()
        for i in range(2):
            arr = (np.random.rand(48, 48) * 255).astype("uint8")
            Image.fromarray(arr, mode="L").save(d / f"{i}.jpg")

    ds = EmotionFolderDataset(str(tmp_path), transform=get_transform("cnn", train=False))
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = EmotionCNN()

    result = evaluate(model, loader, torch.device("cpu"))

    assert 0.0 <= result.overall_accuracy <= 1.0
    assert len(result.confusion) == 7
    assert all(len(row) == 7 for row in result.confusion)
    assert set(result.per_class.keys()) == set(
        ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    )
    for metrics in result.per_class.values():
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert metrics["support"] >= 0

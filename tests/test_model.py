import torch

from model import EmotionCNN, EmotionResNet, build_model, EMOTION_LABELS, NUM_CLASSES


def test_emotion_labels():
    assert len(EMOTION_LABELS) == NUM_CLASSES == 7


def test_cnn_output_shape():
    model = EmotionCNN()
    x = torch.randn(2, 1, 48, 48)
    out = model(x)
    assert out.shape == (2, 7)


def test_resnet_output_shape():
    model = EmotionResNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


def test_build_model_factory():
    assert isinstance(build_model("cnn", pretrained=False), EmotionCNN)
    assert isinstance(build_model("resnet", pretrained=False), EmotionResNet)


def test_build_model_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        build_model("not-a-real-model")

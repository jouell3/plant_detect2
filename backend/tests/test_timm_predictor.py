from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from torchvision import transforms

from herbs_detection.timm_predictor import TimmPredictor


@pytest.fixture
def dummy_predictor():
    """TimmPredictor with mocked internals - no file I/O or GPU needed."""
    classes = ["Basilic", "Menthe", "Thym", "Lavande"]
    p = TimmPredictor.__new__(TimmPredictor)
    p._classes   = classes
    p._img_size  = 224
    p._ready     = MagicMock()
    p._ready.wait = lambda: None

    # Mock model that returns appropriate batch size tensors
    def mock_model(x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, len(classes))

    p._model     = MagicMock(side_effect=mock_model)
    p._preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return p


def test_predict_top3_returns_three_tuples(dummy_predictor, tmp_path):
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64)).save(img_path)
    result = dummy_predictor.predict_top3(str(img_path))
    assert len(result) == 3
    assert all(isinstance(name, str) and isinstance(conf, float)
               for name, conf in result)


def test_predict_set_returns_one_per_image(dummy_predictor, tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (64, 64)).save(p)
        paths.append(str(p))
    result = dummy_predictor.predict_set(paths)
    assert len(result) == 3
    assert all(isinstance(name, str) and isinstance(conf, float)
               for name, conf in result)

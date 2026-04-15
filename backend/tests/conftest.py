"""
All TimmPredictor instances replaced with stubs - no weights or GPU needed.
"""
import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

TOP3_STUB = [("Basilic", 0.90), ("Menthe", 0.07), ("Thym", 0.03)]
TOP1_STUB = ("Basilic", 0.90)


@pytest.fixture(autouse=True)
def mock_all_predictors(monkeypatch):
    import app.api.main as api_mod
    from herbs_detection.model_registry import ENABLED_KEYS

    monkeypatch.setattr(api_mod, "_load_all", lambda: None)

    fake_predictors = {}
    for key in ENABLED_KEYS:
        stub = MagicMock()
        stub.predict_top3.side_effect = lambda _path: list(TOP3_STUB)
        stub.predict_set.side_effect  = lambda paths: [TOP1_STUB] * len(paths)
        fake_predictors[key] = stub

    monkeypatch.setattr(api_mod, "_predictors", fake_predictors)


def make_image_bytes(suffix: str = ".jpg") -> bytes:
    img = Image.new("RGB", (10, 10), color=(100, 150, 200))
    buf = io.BytesIO()
    fmt = "JPEG" if suffix in (".jpg", ".jpeg") else "PNG"
    img.save(buf, format=fmt)
    return buf.getvalue()

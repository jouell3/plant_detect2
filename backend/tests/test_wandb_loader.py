import time
from pathlib import Path

import pytest

from herbs_detection.wandb_loader import is_cache_valid


def test_cache_miss_missing_dir(tmp_path):
    assert not is_cache_valid(tmp_path / "nonexistent", ["model.pth"])


def test_cache_miss_missing_file(tmp_path):
    (tmp_path / "model.pth").write_text("x")
    assert not is_cache_valid(tmp_path, ["model.pth", "encoder.pkl"])


def test_cache_hit(tmp_path):
    for name in ["model.pth", "encoder.pkl"]:
        (tmp_path / name).write_text("x")
    assert is_cache_valid(tmp_path, ["model.pth", "encoder.pkl"], max_age=3600)


def test_cache_expired(tmp_path, monkeypatch):
    p = tmp_path / "model.pth"
    p.write_text("x")
    monkeypatch.setattr(time, "time", lambda: p.stat().st_mtime + 99999)
    assert not is_cache_valid(tmp_path, ["model.pth"], max_age=3600)

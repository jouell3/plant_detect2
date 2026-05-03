import time

import pytest

from herbs_detection.wandb_loader import artifact_local_path, is_cache_valid, label_encoder_local_path


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


def test_artifact_local_path_returns_existing_model_dir(tmp_path):
    artifact_dir = tmp_path / "resnet50_best"
    artifact_dir.mkdir()
    (artifact_dir / "resnet50_best.pth").write_text("weights")

    resolved = artifact_local_path("resnet50_best", tmp_path)

    assert resolved == artifact_dir


def test_artifact_local_path_copies_flat_model_file(tmp_path):
    (tmp_path / "resnet50_best.pth").write_text("weights")

    resolved = artifact_local_path("resnet50_best", tmp_path)

    assert resolved == tmp_path / "resnet50_best"
    assert (resolved / "resnet50_best.pth").read_text() == "weights"


def test_label_encoder_local_path_returns_existing_dir(tmp_path):
    artifact_dir = tmp_path / "label_encoder"
    artifact_dir.mkdir()
    (artifact_dir / "label_encoder.pkl").write_text("encoder")

    resolved = label_encoder_local_path(tmp_path)

    assert resolved == artifact_dir


def test_artifact_local_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifact_local_path("resnet50_best", tmp_path)

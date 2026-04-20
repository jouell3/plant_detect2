import threading
from unittest.mock import MagicMock, patch

import pytest

from herbs_detection.monitoring import WandbMonitor


def test_start_sets_active_on_success():
    m = WandbMonitor()
    mock_run = MagicMock()
    with patch("wandb.init", return_value=mock_run), \
         patch("wandb.Table", return_value=MagicMock()):
        m.start(project="test")
    assert m._active is True


def test_start_sets_inactive_on_failure():
    m = WandbMonitor()
    with patch("wandb.init", side_effect=Exception("no key")):
        m.start(project="test")
    assert m._active is False


def test_log_prediction_noop_when_inactive():
    m = WandbMonitor()
    m._active = False
    m.log_prediction("resnet50", "Basilic", 0.9, 42.0, "predict")


def test_log_artifact_download_noop_when_inactive():
    m = WandbMonitor()
    m._active = False
    m.log_artifact_download("resnet50_best", 1.5, cache_hit=False)


def test_log_prediction_increments_step():
    m = WandbMonitor()
    m._active = True
    m._run = MagicMock()
    m._table = MagicMock()
    m.log_prediction("resnet50", "Basilic", 0.9, 42.0, "predict")
    m.log_prediction("resnet50", "Menthe", 0.8, 38.0, "predict")
    assert m._step == 2


def test_log_prediction_calls_wandb_log():
    m = WandbMonitor()
    mock_run = MagicMock()
    m._active = True
    m._run = mock_run
    m._table = MagicMock()
    m.log_prediction("resnet50", "Basilic", 0.9, 42.0, "predict")
    mock_run.log.assert_called_once()
    call_kwargs = mock_run.log.call_args[0][0]
    assert "resnet50/latency_ms" in call_kwargs
    assert "resnet50/confidence" in call_kwargs


def test_step_counter_thread_safe():
    m = WandbMonitor()
    m._active = True
    m._run = MagicMock()
    m._table = MagicMock()
    threads = [
        threading.Thread(target=m.log_prediction,
                         args=("resnet50", "Basilic", 0.9, 10.0, "predict"))
        for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m._step == 20

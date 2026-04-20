import threading
import time

import pytest

from herbs_detection.metrics_store import MetricsStore, LOW_CONFIDENCE_THRESHOLD


def _req(ts="2026-04-20T11:42:01Z", conf=0.91, latency=187.0):
    return {
        "resnet50": ("Basilic", conf, latency),
        "convnext_tiny": ("Menthe", conf, latency),
    }


def test_record_request_increments_total():
    store = MetricsStore()
    store.record_request("2026-04-20T11:42:01Z", _req())
    snap = store.snapshot()
    assert snap["kpis"]["total_requests"] == 1


def test_snapshot_returns_last_20_newest_first():
    store = MetricsStore()
    for i in range(25):
        store.record_request(
            f"2026-04-20T11:42:{i:02d}Z",
            {"resnet50": ("Basilic", 0.91, 187.0)},
        )
    snap = store.snapshot()
    assert len(snap["recent_requests"]) == 20
    assert snap["recent_requests"][0]["timestamp"] == "11:42:24"
    assert snap["recent_requests"][-1]["timestamp"] == "11:42:05"


def test_all_predictions_flat_shape():
    store = MetricsStore()
    store.record_request(
        "2026-04-20T11:42:01Z",
        {
            "resnet50": ("Basilic", 0.91, 187.0),
            "convnext_tiny": ("Basilic", 0.88, 210.0),
        },
    )
    flat = store.all_predictions()
    assert len(flat) == 2
    assert all(
        k in flat[0]
        for k in ["timestamp", "model", "class", "confidence", "latency_ms"]
    )


def test_low_confidence_threshold():
    store = MetricsStore()
    store.record_request(
        "2026-04-20T11:42:01Z",
        {
            "resnet50": ("Basilic", LOW_CONFIDENCE_THRESHOLD - 0.01, 187.0),
            "convnext_tiny": ("Basilic", 0.88, 210.0),
        },
    )
    assert store.snapshot()["kpis"]["low_confidence_count"] == 1


def test_thread_safety_total_count():
    store = MetricsStore()

    def add():
        store.record_request(
            "2026-04-20T11:42:01Z",
            {"resnet50": ("Basilic", 0.91, 187.0)},
        )

    threads = [threading.Thread(target=add) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.snapshot()["kpis"]["total_requests"] == 20


def test_empty_store_returns_zeroed_kpis():
    store = MetricsStore()
    snap = store.snapshot()
    assert snap["kpis"]["total_requests"] == 0
    assert snap["kpis"]["avg_latency_ms"] == 0.0
    assert snap["kpis"]["avg_confidence"] == 0.0
    assert snap["kpis"]["low_confidence_count"] == 0
    assert snap["recent_requests"] == []
    assert snap["class_distribution"] == {}
    assert snap["model_stats"] == {}

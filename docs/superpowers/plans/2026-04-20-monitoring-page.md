# Live Monitoring Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-memory request tracking to the backend and a Streamlit page that polls it every 10 seconds to display live model health KPIs, a styled predictions table, CSV export, and 4 bar charts.

**Architecture:** A new `MetricsStore` singleton accumulates all prediction requests in-memory (thread-safe). Two new FastAPI endpoints expose it: `GET /metrics` returns a snapshot (KPIs + last 20 rows), `GET /metrics/predictions` returns the full flat list for CSV export. The Streamlit page polls `/metrics` every 10 seconds and renders the dashboard.

**Tech Stack:** Python 3.11, FastAPI, threading.Lock, collections.Counter, Streamlit, pandas, requests

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `backend/app/src/herbs_detection/metrics_store.py` | **Create** | `MetricsStore` class + module singleton |
| `backend/tests/test_metrics_store.py` | **Create** | 6 unit tests for MetricsStore |
| `backend/tests/conftest.py` | **Modify** | Mock `metrics_store` singleton in API tests |
| `backend/app/api/main.py` | **Modify** | Import MetricsStore, add 2 endpoints, wire `record_request()` in 3 handlers |
| `frontend/pages/3_Monitoring.py` | **Create** | Streamlit monitoring page |

---

## Task 1: MetricsStore — tests then implementation

**Files:**
- Create: `backend/app/src/herbs_detection/metrics_store.py`
- Create: `backend/tests/test_metrics_store.py`

### Step 1: Write the failing tests

- [ ] Create `backend/tests/test_metrics_store.py` with this full content:

```python
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
```

### Step 2: Run tests — expect failure (module not found)

- [ ] Run:
```bash
cd backend && pytest tests/test_metrics_store.py -v
```
Expected: `ModuleNotFoundError: No module named 'herbs_detection.metrics_store'`

### Step 3: Implement MetricsStore

- [ ] Create `backend/app/src/herbs_detection/metrics_store.py`:

```python
import threading
import time
from collections import Counter

LOW_CONFIDENCE_THRESHOLD = 0.4


class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: list[dict] = []
        self._total_requests: int = 0
        self._per_model: dict[str, dict] = {}
        self._low_confidence_count: int = 0
        self._class_counter: Counter = Counter()
        self._started_at: float = time.time()

    def record_request(
        self,
        timestamp: str,
        predictions: dict[str, tuple[str, float, float]],
    ) -> None:
        """Record one image prediction event. predictions: model_key → (class, confidence, latency_ms)."""
        display_ts = timestamp.split("T")[-1].rstrip("Z") if "T" in timestamp else timestamp
        with self._lock:
            self._total_requests += 1
            row: dict = {"timestamp": display_ts}
            for model_key, (top1_class, confidence, latency_ms) in predictions.items():
                row[model_key] = {
                    "class": top1_class,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                }
                if model_key not in self._per_model:
                    self._per_model[model_key] = {
                        "count": 0,
                        "sum_latency": 0.0,
                        "sum_confidence": 0.0,
                    }
                self._per_model[model_key]["count"] += 1
                self._per_model[model_key]["sum_latency"] += latency_ms
                self._per_model[model_key]["sum_confidence"] += confidence
                if confidence < LOW_CONFIDENCE_THRESHOLD:
                    self._low_confidence_count += 1
                self._class_counter[top1_class] += 1
            self._requests.append(row)

    def snapshot(self) -> dict:
        """Return KPIs + last 20 requests (newest first) + class distribution + per-model stats."""
        with self._lock:
            total = self._total_requests
            all_counts = sum(s["count"] for s in self._per_model.values())
            avg_latency = (
                sum(s["sum_latency"] for s in self._per_model.values()) / all_counts
                if all_counts
                else 0.0
            )
            avg_confidence = (
                sum(s["sum_confidence"] for s in self._per_model.values()) / all_counts
                if all_counts
                else 0.0
            )
            model_stats = {
                key: {
                    "avg_latency_ms": round(s["sum_latency"] / s["count"], 1),
                    "avg_confidence": round(s["sum_confidence"] / s["count"], 3),
                }
                for key, s in self._per_model.items()
            }
            recent = list(reversed(self._requests[-20:]))
            return {
                "kpis": {
                    "total_requests": total,
                    "avg_latency_ms": round(avg_latency, 1),
                    "avg_confidence": round(avg_confidence, 3),
                    "low_confidence_count": self._low_confidence_count,
                    "uptime_seconds": int(time.time() - self._started_at),
                },
                "recent_requests": recent,
                "class_distribution": dict(self._class_counter.most_common()),
                "model_stats": model_stats,
            }

    def all_predictions(self) -> list[dict]:
        """Flat list — one entry per model per request, for CSV export."""
        with self._lock:
            flat = []
            for row in self._requests:
                ts = row["timestamp"]
                for key, val in row.items():
                    if key == "timestamp":
                        continue
                    flat.append({
                        "timestamp": ts,
                        "model": key,
                        "class": val["class"],
                        "confidence": val["confidence"],
                        "latency_ms": val["latency_ms"],
                    })
            return flat


metrics_store = MetricsStore()
```

### Step 4: Run tests — expect all green

- [ ] Run:
```bash
cd backend && pytest tests/test_metrics_store.py -v
```
Expected: 6 tests pass, 0 failures.

### Step 5: Commit

- [ ] Run:
```bash
git add backend/app/src/herbs_detection/metrics_store.py backend/tests/test_metrics_store.py
git commit -m "feat: add MetricsStore for in-memory prediction tracking"
```

---

## Task 2: Backend endpoints and wiring

**Files:**
- Modify: `backend/tests/conftest.py`
- Modify: `backend/app/api/main.py`

### Step 1: Update conftest.py to mock metrics_store

- [ ] Open `backend/tests/conftest.py`. After the line `monkeypatch.setattr(monitoring_mod, "monitor", MagicMock())`, add two lines to patch `metrics_store` in both the module and the API's local reference. Replace the entire `mock_all_predictors` fixture with:

```python
"""
All TimmPredictor instances replaced with stubs - no weights or GPU needed.
Monitor singleton patched to a MagicMock so tests never hit wandb.
MetricsStore singleton patched to a MagicMock so tests never accumulate state.
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
    from herbs_detection import monitoring as monitoring_mod

    from herbs_detection.model_registry import ENABLED_KEYS

    monkeypatch.setattr(api_mod, "_load_all", lambda: None)
    monkeypatch.setattr(monitoring_mod, "monitor", MagicMock())
    monkeypatch.setattr(api_mod, "metrics_store", MagicMock())

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
```

### Step 2: Run existing API tests to confirm no regressions

- [ ] Run:
```bash
cd backend && pytest tests/test_api.py -v
```
Expected: all existing tests pass (the MagicMock for metrics_store means `record_request()` calls are silently accepted).

### Step 3: Add imports and endpoints to main.py

- [ ] Open `backend/app/api/main.py`. After the line `from herbs_detection.monitoring import monitor`, add:

```python
from herbs_detection.metrics_store import metrics_store
```

- [ ] After the `@api.get("/models")` endpoint and before `@api.post("/predict")`, add the two new GET endpoints:

```python
@api.get("/metrics")
def get_metrics():
    """Live model health snapshot: KPIs, last 20 requests, class distribution, per-model stats."""
    return metrics_store.snapshot()


@api.get("/metrics/predictions")
def get_all_predictions():
    """Full prediction history as a flat list (for CSV export)."""
    return {"predictions": metrics_store.all_predictions()}
```

### Step 4: Wire record_request() in /predict

- [ ] In the `/predict` handler, replace the body of the `try` block (everything from `results = []` to before `finally:`) with:

```python
    collected: dict[str, tuple[str, float, float]] = {}
    results = []
    try:
        for key in keys:
            t0 = time.perf_counter()
            top3 = _predictors[key].predict_top3(tmp_path)[:top_k]
            latency_ms = (time.perf_counter() - t0) * 1000
            monitor.log_prediction(key, top3[0][0], top3[0][1], latency_ms, "predict")
            collected[key] = (top3[0][0], top3[0][1], latency_ms)
            results.append({
                "model": key,
                "top3": [{"class": c, "confidence": conf} for c, conf in top3],
            })
        metrics_store.record_request(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            predictions=collected,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"predictions": results}
```

### Step 5: Wire record_request() in /predict-batch

- [ ] In the `/predict-batch` handler, replace the `per_model = {}` line and the `try` block content (everything from `per_model = {}` to before `finally:`) with:

```python
    per_model: dict[str, list] = {}
    per_model_latency: dict[str, float] = {}
    try:
        for key in keys:
            t0 = time.perf_counter()
            preds = _predictors[key].predict_set(tmp_paths)
            latency_ms = (time.perf_counter() - t0) * 1000 / max(len(tmp_paths), 1)
            per_model[key] = preds
            per_model_latency[key] = latency_ms
            if preds:
                monitor.log_prediction(key, preds[0][0], preds[0][1], latency_ms, "predict-batch")
        for i in range(len(filenames)):
            metrics_store.record_request(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                predictions={
                    key: (per_model[key][i][0], per_model[key][i][1], per_model_latency[key])
                    for key in keys
                },
            )
    finally:
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)
```

### Step 6: Wire record_request() in /explore

- [ ] In the `/explore` handler, replace the body of the `try` block (everything from `results = []` to before `finally:`) with:

```python
    collected: dict[str, tuple[str, float, float]] = {}
    results = []
    try:
        for key in keys:
            t0 = time.perf_counter()
            top3 = _predictors[key].predict_top3(tmp_path)[:top_k]
            latency_ms = (time.perf_counter() - t0) * 1000
            monitor.log_prediction(key, top3[0][0], top3[0][1], latency_ms, "explore")
            collected[key] = (top3[0][0], top3[0][1], latency_ms)
            results.append({
                "model": key,
                "top_k": [
                    {"rank": i + 1, "class": c, "confidence": conf}
                    for i, (c, conf) in enumerate(top3)
                ],
            })
        metrics_store.record_request(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            predictions=collected,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"filename": file.filename, "predictions": results}
```

### Step 7: Run full test suite

- [ ] Run:
```bash
cd backend && pytest -v
```
Expected: all tests pass (MetricsStore is mocked in API tests; MetricsStore unit tests pass independently).

### Step 8: Commit

- [ ] Run:
```bash
git add backend/app/api/main.py backend/tests/conftest.py
git commit -m "feat: add /metrics endpoints and wire MetricsStore into prediction handlers"
```

---

## Task 3: Streamlit monitoring page

**Files:**
- Create: `frontend/pages/3_Monitoring.py`

### Step 1: Create the monitoring page

- [ ] Create `frontend/pages/3_Monitoring.py` with this full content:

```python
import csv
import io
import time

import pandas as pd
import requests
import streamlit as st

API_URL = "https://plant-predictor-966041648100.europe-west1.run.app"

st.set_page_config(page_title="Model Monitoring", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_metrics() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _fetch_all_predictions() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/metrics/predictions", timeout=10)
        r.raise_for_status()
        return r.json().get("predictions", [])
    except Exception:
        return []


def _build_table_html(rows: list[dict], model_keys: list[str]) -> str:
    header_cells = "".join(
        f'<th style="padding:6px 10px;text-align:left">{k}</th>'
        for k in ["Timestamp"] + model_keys
    )
    body_rows = []
    for row in rows:
        classes = [row[k]["class"] for k in model_keys if k in row]
        majority = max(set(classes), key=classes.count) if classes else ""
        cells = [f'<td style="padding:5px 10px">{row["timestamp"]}</td>']
        for k in model_keys:
            if k not in row:
                cells.append('<td style="padding:5px 10px">—</td>')
                continue
            val = row[k]
            conf: float = val["confidence"]
            cls: str = val["class"]
            if conf < 0.4:
                bg = "#ffcccc"
            elif cls != majority:
                bg = "#ffe8cc"
            else:
                bg = "transparent"
            cells.append(
                f'<td style="padding:5px 10px;background:{bg}">'
                f'{cls}<br><span style="font-size:11px;color:#555">{conf:.0%}</span></td>'
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        f'<thead><tr style="background:#1a1a2e;color:white">{header_cells}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _build_csv(predictions: list[dict]) -> str:
    if not predictions:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["timestamp", "model", "class", "confidence", "latency_ms"]
    )
    writer.writeheader()
    writer.writerows(predictions)
    return buf.getvalue()


def _agreement_rate(rows: list[dict], model_keys: list[str]) -> dict[str, float]:
    """Per-model % of requests where that model agreed with the majority class."""
    agreed: dict[str, int] = {k: 0 for k in model_keys}
    total: dict[str, int] = {k: 0 for k in model_keys}
    for row in rows:
        present = [k for k in model_keys if k in row]
        if len(present) < 2:
            continue
        classes = [row[k]["class"] for k in present]
        majority = max(set(classes), key=classes.count)
        for k in present:
            total[k] += 1
            if row[k]["class"] == majority:
                agreed[k] += 1
    return {
        k: round(agreed[k] / total[k] * 100, 1) if total[k] else 0.0
        for k in model_keys
    }


# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
data = _fetch_metrics()

# ---------------------------------------------------------------------------
# Header + status badge
# ---------------------------------------------------------------------------
col_title, col_badge = st.columns([8, 2])
with col_title:
    st.title("Model Monitoring")
with col_badge:
    if data:
        st.success("Connected")
    else:
        st.error("API unreachable")

if data is None:
    st.warning("API unreachable — retrying in 10s")
    if st.sidebar.toggle("Auto-refresh (10s)", value=True):
        time.sleep(10)
        st.rerun()
    st.stop()

kpis = data.get("kpis", {})
recent = data.get("recent_requests", [])
class_dist = data.get("class_distribution", {})
model_stats = data.get("model_stats", {})
model_keys = list(model_stats.keys())

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Requests", kpis.get("total_requests", 0))
c2.metric("Avg Latency", f"{kpis.get('avg_latency_ms', 0):.0f} ms")
c3.metric("Avg Confidence", f"{kpis.get('avg_confidence', 0):.1%}")
c4.metric("Low Conf Alerts", kpis.get("low_confidence_count", 0))

st.divider()

# ---------------------------------------------------------------------------
# Predictions table
# ---------------------------------------------------------------------------
st.subheader("Recent Predictions (last 20)")

if recent and model_keys:
    st.markdown(_build_table_html(recent, model_keys), unsafe_allow_html=True)
else:
    st.info("No predictions recorded yet.")

# CSV download (fetches full history)
all_preds = _fetch_all_predictions()
csv_data = _build_csv(all_preds)
st.download_button(
    label="Download all predictions (CSV)",
    data=csv_data,
    file_name="predictions.csv",
    mime="text/csv",
    disabled=not csv_data,
)

st.divider()

# ---------------------------------------------------------------------------
# 2x2 chart grid
# ---------------------------------------------------------------------------
if model_stats:
    left, right = st.columns(2)

    with left:
        st.subheader("Avg Latency per Model (ms)")
        latency_df = pd.DataFrame(
            {"Latency (ms)": {k: v["avg_latency_ms"] for k, v in model_stats.items()}}
        )
        st.bar_chart(latency_df)

        st.subheader("Class Distribution (top 10)")
        top10 = dict(list(class_dist.items())[:10])
        if top10:
            st.bar_chart(pd.DataFrame({"Count": top10}))
        else:
            st.info("No predictions yet.")

    with right:
        st.subheader("Avg Confidence per Model")
        conf_df = pd.DataFrame(
            {"Confidence": {k: v["avg_confidence"] for k, v in model_stats.items()}}
        )
        st.bar_chart(conf_df)

        st.subheader("Model Agreement Rate (%)")
        agreement = _agreement_rate(recent, model_keys)
        if any(v > 0 for v in agreement.values()):
            st.bar_chart(pd.DataFrame({"Agreement %": agreement}))
        else:
            st.info("Not enough multi-model requests to compute agreement.")
else:
    st.info("Charts will appear after the first prediction.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if st.sidebar.toggle("Auto-refresh (10s)", value=True):
    time.sleep(10)
    st.rerun()
```

### Step 2: Verify frontend can start (no import errors)

- [ ] Run:
```bash
cd frontend && python -c "import py_compile; py_compile.compile('pages/3_Monitoring.py', doraise=True)" && echo "OK"
```
Expected: `OK`

### Step 3: Manual smoke test

- [ ] Start the backend locally:
```bash
PYTHONPATH=backend/app/src uvicorn backend.app.api.main:api --reload --host 0.0.0.0 --port 8080
```
- [ ] In a second terminal, confirm the endpoints respond:
```bash
curl -s http://localhost:8080/metrics | python3 -m json.tool | head -20
curl -s http://localhost:8080/metrics/predictions | python3 -m json.tool
```
Expected: `/metrics` returns `{"kpis": {...}, "recent_requests": [], ...}`. `/metrics/predictions` returns `{"predictions": []}`.

- [ ] Start the frontend:
```bash
cd frontend && streamlit run main.py
```
- [ ] Open the browser, navigate to the Monitoring page, confirm it shows "Connected", 4 KPI cards at 0, an empty predictions table, and disabled CSV download button.

- [ ] POST a test prediction to generate data:
```bash
curl -s -X POST http://localhost:8080/predict \
  -F "file=@data/raw/all_images/dill_0.jpg" \
  -F "models=all" | python3 -m json.tool
```
- [ ] Refresh the Monitoring page (or wait 10s). Confirm total requests increments to 1 and the predictions table shows one row with 5 model columns.

### Step 4: Commit

- [ ] Run:
```bash
git add frontend/pages/3_Monitoring.py
git commit -m "feat: add live Monitoring page with KPI cards, predictions table, and charts"
```

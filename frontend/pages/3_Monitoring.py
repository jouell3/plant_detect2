import csv
import io
import time
from collections import Counter

import pandas as pd
import requests
import streamlit as st

# Local API URL - change to your deployed API endpoint if needed
#API_URL = "http://localhost:8080"

# Deployed API URL (replace with your actual endpoint)
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
        counts = Counter(classes)
        top_count = counts.most_common(1)[0][1] if counts else 0
        winners = [cls for cls, cnt in counts.items() if cnt == top_count]
        majority = winners[0] if len(winners) == 1 else ""
        cells = [f'<td style="padding:5px 10px">{row.get("timestamp", "—")}</td>']
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


@st.cache_data(ttl=30)
def _fetch_csv_cached() -> str:
    preds = _fetch_all_predictions()
    return _build_csv(preds)


def _build_csv(predictions: list[dict]) -> str:
    if not predictions:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["timestamp", "model", "class", "confidence", "latency_ms"],
        extrasaction="ignore",
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
        counts = Counter(classes)
        top_count = counts.most_common(1)[0][1]
        winners = [cls for cls, cnt in counts.items() if cnt == top_count]
        if len(winners) > 1:
            continue  # genuine tie — skip this row
        majority = winners[0]
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
auto_refresh = st.sidebar.toggle("Auto-refresh (10s)", value=True, key="auto_refresh")

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
    if auto_refresh:
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

# CSV download (fetches full history, cached for 30s to avoid re-fetch on every auto-refresh)
csv_data = _fetch_csv_cached()
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
if auto_refresh:
    time.sleep(10)
    st.rerun()

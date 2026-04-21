import csv
import io
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import inject_global_css

# Local API URL - change to your deployed API endpoint if needed
#API_URL = "http://localhost:8080"

# Deployed API URL (replace with your actual endpoint)
API_URL = "https://plant-predictor-966041648100.europe-west1.run.app"

_MODEL_WEIGHTS = {
    "convnext_tiny":     0.954,
    "efficientnet_b3":   0.929,
    "efficientnet_b4":   0.928,
    "mobilenetv3_large": 0.879,
    "resnet50":          0.862,
}


def _weighted_winner(row: dict, model_keys: list[str]) -> str:
    scores: dict[str, float] = defaultdict(float)
    for k in model_keys:
        if k in row:
            w = _MODEL_WEIGHTS.get(k, 1.0)
            scores[row[k]["class"]] += w * row[k]["confidence"]
    return max(scores, key=scores.get) if scores else ""


st.set_page_config(page_title="Model Monitoring", layout="wide")
inject_global_css()


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


def _kpi_card(label: str, value: str, icon: str, accent: str = "#2e7d32") -> str:
    return (
        f"<div style='background:white;border:1px solid #dde5dd;border-radius:14px;"
        f"padding:1.2rem 1rem;text-align:center;box-shadow:0 2px 8px rgba(26,46,35,0.07)'>"
        f"<div style='font-size:1.4rem;margin-bottom:6px'>{icon}</div>"
        f"<div style='font-size:2rem;font-weight:700;color:{accent};line-height:1.1;"
        f"font-family:DM Sans,sans-serif'>{value}</div>"
        f"<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;"
        f"color:#7a9e87;font-weight:600;margin-top:6px'>{label}</div>"
        f"</div>"
    )


def _build_table_html(rows: list[dict], model_keys: list[str]) -> str:
    header_cells = "".join(
        f'<th style="padding:8px 12px;text-align:left;font-size:0.74rem;'
        f'text-transform:uppercase;letter-spacing:0.8px;font-weight:600">{k}</th>'
        for k in ["Time"] + model_keys
    )
    body_rows = []
    for i, row in enumerate(rows):
        majority = _weighted_winner(row, model_keys)
        row_bg = "#ffffff" if i % 2 == 0 else "#f8f6f1"
        cells = [
            f'<td style="padding:8px 12px;color:#5a7a62;font-size:0.85rem;'
            f'font-family:monospace">{row.get("timestamp", "—")}</td>'
        ]
        for k in model_keys:
            if k not in row:
                cells.append('<td style="padding:8px 12px;color:#bdbdbd">—</td>')
                continue
            val = row[k]
            conf: float = val["confidence"]
            cls: str = val["class"]
            strong_consensus = cls == majority and majority != ""
            if conf < 0.4:
                bg = "#fdf0f2"
                text_color = "#c62828"
                weight = "500"
            elif strong_consensus:
                bg = row_bg
                text_color = "#1a2e23"
                weight = "700"
            else:
                bg = row_bg
                text_color = "#5a6672"
                weight = "400"
            cells.append(
                f'<td style="padding:8px 12px;background:{bg};white-space:nowrap">'
                f'<span style="font-weight:{weight};color:{text_color};font-size:0.96rem">{cls}</span>'
                f'<span style="font-size:0.88rem;color:#8a9ea8;margin-left:8px">{conf:.0%}</span>'
                f'</td>'
            )
        body_rows.append(f"<tr style='background:{row_bg}'>{''.join(cells)}</tr>")

    return (
        '<div style="border:1px solid #dde5dd;border-radius:12px;overflow:hidden">'
        '<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr style="background:#1a2e23;color:#c8dcc9">{header_cells}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
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
low_conf = kpis.get("low_confidence_count", 0)
uptime_s = kpis.get("uptime_seconds", 0)
uptime_str = f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m" if uptime_s >= 3600 else f"{uptime_s // 60}m {uptime_s % 60}s"

kpi_cols = st.columns(5)
kpi_data = [
    ("Total Requests", str(kpis.get("total_requests", 0)), "📊", "#2e7d32"),
    ("Avg Latency", f"{kpis.get('avg_latency_ms', 0):.0f} ms", "⚡", "#1565c0"),
    ("Avg Confidence", f"{kpis.get('avg_confidence', 0):.1%}", "🎯", "#6a1b9a"),
    ("Low Conf Alerts", str(low_conf), "⚠️", "#c62828" if low_conf > 0 else "#7a9e87"),
    ("Uptime", uptime_str, "🕐", "#2e7d32"),
]
for col, (label, value, icon, accent) in zip(kpi_cols, kpi_data):
    col.markdown(_kpi_card(label, value, icon, accent), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Predictions table
# ---------------------------------------------------------------------------
st.subheader("Recent Predictions (last 20)")

if recent and model_keys:
    st.markdown(_build_table_html(recent, model_keys), unsafe_allow_html=True)
    st.markdown(
        "<div style='margin-top:12px;background:white;border:1px solid #dde5dd;"
        "border-radius:10px;padding:10px 18px;display:flex;gap:24px;flex-wrap:wrap;"
        "align-items:center'>"
        "<span style='font-size:0.82rem;font-weight:700;color:#1a2e23;margin-right:2px'>"
        "Colour guide</span>"
        "<span style='font-size:0.84rem;color:#444;display:flex;align-items:center;gap:7px'>"
        "<span style='display:inline-block;width:16px;height:16px;background:#fdf0f2;"
        "border-radius:3px;flex-shrink:0'></span>"
        "Confidence below 40%</span>"
        "<span style='font-size:0.84rem;color:#444;display:flex;align-items:center;gap:7px'>"
        "<span style='font-size:0.9rem;font-weight:700;color:#1a2e23'>B</span>"
        "<strong>Bold name</strong> — weighted majority vote winner</span>"
        "<span style='font-size:0.84rem;color:#9aafb0'>Normal text = not the weighted vote winner</span>"
        "</div>",
        unsafe_allow_html=True,
    )
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
else:
    st.info("Charts will appear after the first prediction.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(10)
    st.rerun()

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

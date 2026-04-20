import os
import threading
import time

from loguru import logger


class WandbMonitor:
    def __init__(self):
        self._active = False
        self._run = None
        self._table = None
        self._step = 0
        self._lock = threading.Lock()

    def start(self, project: str, entity: str = "") -> None:
        try:
            import wandb
            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key, relogin=True)
            kwargs = {"project": project, "job_type": "serving"}
            if entity:
                kwargs["entity"] = entity
            self._run = wandb.init(**kwargs)
            self._table = wandb.Table(
                columns=["timestamp", "endpoint", "model", "top1_class", "confidence", "latency_ms"]
            )
            self._active = True
            logger.info("WandB monitoring run started: {}", self._run.id)
        except Exception as exc:
            logger.warning("WandB monitoring unavailable: {}. Logging to console only.", exc)
            self._active = False

    def finish(self) -> None:
        if not self._active:
            return
        try:
            self._run.log({"predictions": self._table})
            self._run.finish()
            logger.info("WandB monitoring run finished.")
        except Exception as exc:
            logger.warning("WandB finish failed: {}", exc)
        finally:
            self._active = False

    def log_artifact_download(self, artifact_name: str, duration_s: float, cache_hit: bool) -> None:
        if not self._active:
            return
        try:
            with self._lock:
                step = self._step
                self._step += 1
            self._run.log({
                f"startup/{artifact_name}/download_duration_s": duration_s,
                f"startup/{artifact_name}/cache_hit": int(cache_hit),
            }, step=step)
        except Exception as exc:
            logger.warning("WandB log_artifact_download failed: {}", exc)

    def log_prediction(self, model_key: str, top1_class: str, confidence: float,
                       latency_ms: float, endpoint: str) -> None:
        if not self._active:
            return
        try:
            with self._lock:
                step = self._step
                self._step += 1
            self._run.log({
                f"{model_key}/latency_ms": latency_ms,
                f"{model_key}/confidence": confidence,
            }, step=step)
            self._table.add_data(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                endpoint,
                model_key,
                top1_class,
                confidence,
                latency_ms,
            )
        except Exception as exc:
            logger.warning("WandB log_prediction failed: {}", exc)


monitor = WandbMonitor()

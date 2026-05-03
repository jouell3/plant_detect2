class WandbMonitor:
    def __init__(self):
        self._active = False

    def start(self, project: str, entity: str = "") -> None:
        self._active = False

    def finish(self) -> None:
        self._active = False

    def log_artifact_download(self, artifact_name: str, duration_s: float, cache_hit: bool) -> None:
        return

    def log_prediction(self, model_key: str, top1_class: str, confidence: float,
                       latency_ms: float, endpoint: str) -> None:
        return


monitor = WandbMonitor()

"""
Generic predictor for any timm-compatible classification model.
Replaces the separate model_*.py files - one class handles all 5 architectures.
"""
import pickle
import threading
from pathlib import Path

import timm
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms

from .model_registry import ModelConfig
from .wandb_loader import artifact_local_path, label_encoder_local_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimmPredictor:
    """Load a timm model from a wandb artifact and expose predict_top3/predict_set."""

    def __init__(self, cfg: ModelConfig, cache_root: Path | None = None):
        self._cfg        = cfg
        self._cache_root = cache_root
        self._model: torch.nn.Module | None = None
        self._load_error: Exception | None = None
        self._classes: list[str] = []
        self._img_size   = cfg.img_size
        self._ready      = threading.Event()

        self._preprocess = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load(self) -> None:
        """Download artifact (if needed) and load weights into memory."""
        try:
            local_dir = artifact_local_path(self._cfg.wandb_artifact, self._cache_root)

            # Check model artifact dir first, then fall back to shared preprocessor artifact
            encoder_files = list(local_dir.glob("*.pkl"))
            if not encoder_files:
                shared_dir = label_encoder_local_path(self._cache_root)
                encoder_files = list(shared_dir.glob("*.pkl"))
            if encoder_files:
                # sklearn LabelEncoder requires pickle — source is our own wandb registry
                with open(encoder_files[0], "rb") as f:
                    le = pickle.load(f)  # noqa: S301
                self._classes = list(le.classes_)
            else:
                classes_file = local_dir / "classes.txt"
                if classes_file.exists():
                    self._classes = classes_file.read_text().splitlines()
                else:
                    raise FileNotFoundError(
                        f"No label encoder (*.pkl) in {local_dir} or the shared "
                        "'label_encoder' preprocessor artifact, and no classes.txt found."
                    )

            num_classes = len(self._classes)
            pth_files   = list(local_dir.glob("*.pth"))
            if not pth_files:
                raise FileNotFoundError(f"No .pth file found in {local_dir}")

            self._model = timm.create_model(
                self._cfg.timm_name, pretrained=False, num_classes=num_classes
            )
            self._model.load_state_dict(torch.load(pth_files[0], map_location=DEVICE))
            self._model.to(DEVICE)
            self._model.train(False)   # sets inference mode (same as .eval())
            logger.info("{} ready. device={} classes={}",
                        self._cfg.key, DEVICE, num_classes)
        except Exception as exc:
            self._load_error = exc
            logger.error("Failed to load {}: {}", self._cfg.key, exc)
        finally:
            self._ready.set()

    def _check_ready(self) -> None:
        """Block until load() completes, then raise if it failed."""
        self._ready.wait()
        if self._load_error is not None:
            raise RuntimeError(
                f"Model {self._cfg.key} failed to load"
            ) from self._load_error

    def predict_top3(self, img_path: str) -> list[tuple[str, float]]:
        """Return top-3 (class_name, confidence) for a single image."""
        self._check_ready()
        tensor = self._preprocess(
            Image.open(img_path).convert("RGB")
        ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            proba = torch.softmax(self._model(tensor), dim=1).squeeze()
        k = min(3, len(self._classes))
        top3 = proba.topk(k)
        return [
            (self._classes[i.item()], round(p.item(), 4))
            for i, p in zip(top3.indices, top3.values)
        ]

    def predict_set(self, img_paths: list[str],
                    batch_size: int = 32) -> list[tuple[str, float]]:
        """Return top-1 (class_name, confidence) for each image in a batch."""
        self._check_ready()
        results = []
        for start in range(0, len(img_paths), batch_size):
            chunk = img_paths[start: start + batch_size]
            batch = torch.stack([
                self._preprocess(Image.open(p).convert("RGB")) for p in chunk
            ]).to(DEVICE)
            with torch.no_grad():
                proba = torch.softmax(self._model(batch), dim=1)
            confs, idxs = proba.max(dim=1)
            results.extend(
                (self._classes[i.item()], round(c.item(), 4))
                for i, c in zip(idxs, confs)
            )
        return results

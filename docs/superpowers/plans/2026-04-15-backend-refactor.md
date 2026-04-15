# Backend Refactor - 5-Model timm API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old GCS-backed 4-model backend with a wandb-backed 5-model API that serves predictions from all benchmark winners, supports per-request model selection, and runs fully locally before any Cloud Run deployment.

**Architecture:** A single generic `TimmPredictor` class handles all 5 architectures via timm, replacing the separate `model_*.py` files. A `wandb_loader.py` module downloads and locally caches model artifacts from wandb at startup, replacing the GCS download logic. The FastAPI layer gains a `models` form field so the frontend can request any subset of the 5 models per call.

**Tech Stack:** FastAPI, timm, wandb SDK, PyTorch, pytest + httpx (existing), loguru (existing)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `backend/app/src/herbs_detection/model_registry.py` | Single source of truth: 5 model configs (timm name, img size, wandb artifact) |
| **Create** | `backend/app/src/herbs_detection/wandb_loader.py` | Download wandb artifact to local cache; TTL-based invalidation |
| **Create** | `backend/app/src/herbs_detection/timm_predictor.py` | Generic load/predict class for any timm model; replaces all `model_*.py` files |
| **Modify** | `backend/app/api/main.py` | New endpoints with optional `models=` form field; startup loads all 5 |
| **Modify** | `backend/tests/conftest.py` | Stubs for the new `TimmPredictor` API |
| **Modify** | `backend/tests/test_api.py` | Updated tests for new response shape and model-selection behaviour |
| **Modify** | `backend/app/src/herbs_detection/__init__.py` | Export `TimmPredictor`, `MODEL_REGISTRY`, `ENABLED_KEYS` |
| **Keep (unchanged)** | All old `model_*.py` and `gcs_cache.py` | Not deleted yet - user may want them as reference |

---

## Task 1 - Model Registry

**Files:**
- Create: `backend/app/src/herbs_detection/model_registry.py`

- [ ] **Step 1: Create the registry file**

```python
# backend/app/src/herbs_detection/model_registry.py
"""
Single source of truth for all deployed model configurations.
Add or comment out entries here to enable/disable models globally.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    key: str            # identifier used in API requests and wandb artifact names
    timm_name: str      # timm.create_model() identifier
    img_size: int       # native input resolution
    wandb_artifact: str # artifact name in the wandb registry (without :tag)
    enabled: bool = True


MODEL_REGISTRY: list[ModelConfig] = [
    ModelConfig(
        key="convnext_tiny",
        timm_name="convnext_tiny",
        img_size=224,
        wandb_artifact="convnext_tiny_best",
    ),
    ModelConfig(
        key="efficientnet_b3",
        timm_name="efficientnet_b3",
        img_size=300,
        wandb_artifact="efficientnet_b3_best",
    ),
    ModelConfig(
        key="efficientnet_b4",
        timm_name="efficientnet_b4",
        img_size=380,
        wandb_artifact="efficientnet_b4_best",
    ),
    ModelConfig(
        key="mobilenetv3_large",
        timm_name="mobilenetv3_large_100",
        img_size=224,
        wandb_artifact="mobilenetv3_large_best",
    ),
    ModelConfig(
        key="resnet50",
        timm_name="resnet50",
        img_size=224,
        wandb_artifact="resnet50_best",
    ),
]

# Fast lookup by key
REGISTRY_BY_KEY: dict[str, ModelConfig] = {m.key: m for m in MODEL_REGISTRY}
ENABLED_KEYS: list[str] = [m.key for m in MODEL_REGISTRY if m.enabled]
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/src/herbs_detection/model_registry.py
git commit -m "feat: add 5-model registry (timm configs + wandb artifact names)"
```

---

## Task 2 - wandb Artifact Loader

**Files:**
- Create: `backend/app/src/herbs_detection/wandb_loader.py`
- Create: `backend/tests/test_wandb_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_wandb_loader.py
import time
from pathlib import Path

import pytest

from herbs_detection.wandb_loader import is_cache_valid


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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && pytest tests/test_wandb_loader.py -v
```
Expected: `ModuleNotFoundError` (file does not exist yet).

- [ ] **Step 3: Create the loader**

```python
# backend/app/src/herbs_detection/wandb_loader.py
"""
Download wandb model artifacts to a local cache directory.
Mirrors the TTL-based pattern from the old gcs_cache.py.
"""
import os
import shutil
import time
from pathlib import Path

from loguru import logger

_CACHE_MAX_AGE = int(os.getenv("WANDB_CACHE_MAX_AGE_SECONDS", str(3 * 3600)))
_WANDB_PROJECT = os.getenv("WANDB_PROJECT", "certification")
_WANDB_ENTITY  = os.getenv("WANDB_ENTITY", "")


def is_cache_valid(local_dir: Path, filenames: list[str],
                   max_age: int = _CACHE_MAX_AGE) -> bool:
    """Return True if all files exist and are fresher than max_age seconds."""
    if not local_dir.exists():
        return False
    now = time.time()
    for name in filenames:
        p = local_dir / name
        if not p.exists():
            logger.debug("Cache miss: {} not found.", p)
            return False
        if now - p.stat().st_mtime > max_age:
            logger.debug("Cache expired: {}.", p)
            return False
    logger.info("Cache hit for {}.", local_dir)
    return True


def artifact_local_path(artifact_name: str, cache_root: Path | None = None) -> Path:
    """
    Return a local directory containing the artifact files.
    Downloads from wandb if not cached; falls back to a .pth in cwd.

    Resolution order:
    1. Local cache (TTL-based) - skip download if a fresh copy exists
    2. wandb registry download
    3. Local fallback - look for {artifact_name}.pth in cwd
    """
    if cache_root is None:
        cache_root = Path.cwd() / "models" / "wandb"

    local_dir = cache_root / artifact_name
    pth_files = list(local_dir.glob("*.pth")) if local_dir.exists() else []

    if pth_files and is_cache_valid(local_dir, [pth_files[0].name]):
        return local_dir

    logger.info("Downloading wandb artifact: {}", artifact_name)
    try:
        import wandb
        entity_prefix = f"{_WANDB_ENTITY}/" if _WANDB_ENTITY else ""
        ref = f"{entity_prefix}{_WANDB_PROJECT}/{artifact_name}:latest"
        run = wandb.init(project=_WANDB_PROJECT, job_type="inference",
                         reinit="finish_previous")
        artifact = run.use_artifact(ref, type="model")
        local_dir.mkdir(parents=True, exist_ok=True)
        artifact.download(root=str(local_dir))
        run.finish()
        logger.info("Downloaded to {}.", local_dir)
    except Exception as exc:
        logger.warning("wandb download failed ({}). Trying local fallback.", exc)
        fallback = Path.cwd() / f"{artifact_name}.pth"
        if fallback.exists():
            local_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback, local_dir / fallback.name)
            logger.info("Using local fallback: {}.", fallback)
        else:
            raise FileNotFoundError(
                f"wandb download failed and no local fallback found for {artifact_name}. "
                f"Place {artifact_name}.pth in the working directory or set WANDB_API_KEY."
            ) from exc

    return local_dir
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && pytest tests/test_wandb_loader.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/app/src/herbs_detection/wandb_loader.py backend/tests/test_wandb_loader.py
git commit -m "feat: wandb artifact loader with TTL cache and local fallback"
```

---

## Task 3 - Generic TimmPredictor

**Files:**
- Create: `backend/app/src/herbs_detection/timm_predictor.py`
- Create: `backend/tests/test_timm_predictor.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_timm_predictor.py
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from torchvision import transforms

from herbs_detection.timm_predictor import TimmPredictor


@pytest.fixture
def dummy_predictor():
    """TimmPredictor with mocked internals - no file I/O or GPU needed."""
    classes = ["Basilic", "Menthe", "Thym", "Lavande"]
    p = TimmPredictor.__new__(TimmPredictor)
    p._classes   = classes
    p._img_size  = 224
    p._ready     = MagicMock()
    p._ready.wait = lambda: None
    p._model     = MagicMock(return_value=torch.zeros(1, len(classes)))
    p._preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return p


def test_predict_top3_returns_three_tuples(dummy_predictor, tmp_path):
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64)).save(img_path)
    result = dummy_predictor.predict_top3(str(img_path))
    assert len(result) == 3
    assert all(isinstance(name, str) and isinstance(conf, float)
               for name, conf in result)


def test_predict_set_returns_one_per_image(dummy_predictor, tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (64, 64)).save(p)
        paths.append(str(p))
    result = dummy_predictor.predict_set(paths)
    assert len(result) == 3
    assert all(isinstance(name, str) and isinstance(conf, float)
               for name, conf in result)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && pytest tests/test_timm_predictor.py -v
```
Expected: `ImportError` (file does not exist yet).

- [ ] **Step 3: Create TimmPredictor**

```python
# backend/app/src/herbs_detection/timm_predictor.py
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
from .wandb_loader import artifact_local_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimmPredictor:
    """Load a timm model from a wandb artifact and expose predict_top3/predict_set."""

    def __init__(self, cfg: ModelConfig, cache_root: Path | None = None):
        self._cfg        = cfg
        self._cache_root = cache_root
        self._model      = None
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
        local_dir = artifact_local_path(self._cfg.wandb_artifact, self._cache_root)

        # Load class names from label encoder or classes.txt
        encoder_files = list(local_dir.glob("*.pkl"))
        if encoder_files:
            with open(encoder_files[0], "rb") as f:
                le = pickle.load(f)
            self._classes = list(le.classes_)
        else:
            classes_file = local_dir / "classes.txt"
            if classes_file.exists():
                self._classes = classes_file.read_text().splitlines()
            else:
                raise FileNotFoundError(
                    f"No label encoder (*.pkl) or classes.txt in {local_dir}. "
                    "Save one alongside the .pth file in the wandb artifact."
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
        self._ready.set()
        logger.info("{} ready. device={} classes={}",
                    self._cfg.key, DEVICE, num_classes)

    def predict_top3(self, img_path: str) -> list[tuple[str, float]]:
        """Return top-3 (class_name, confidence) for a single image."""
        self._ready.wait()
        tensor = self._preprocess(
            Image.open(img_path).convert("RGB")
        ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            proba = torch.softmax(self._model(tensor), dim=1).squeeze()
        top3 = proba.topk(3)
        return [
            (self._classes[i.item()], round(p.item(), 4))
            for i, p in zip(top3.indices, top3.values)
        ]

    def predict_set(self, img_paths: list[str],
                    batch_size: int = 32) -> list[tuple[str, float]]:
        """Return top-1 (class_name, confidence) for each image in a batch."""
        self._ready.wait()
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
```

Note: `.train(False)` sets the module to inference mode identically to the commonly seen `.eval()` alias.

- [ ] **Step 4: Run tests**

```bash
cd backend && pytest tests/test_timm_predictor.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/app/src/herbs_detection/timm_predictor.py backend/tests/test_timm_predictor.py
git commit -m "feat: generic TimmPredictor replaces separate model_*.py files"
```

---

## Task 4 - Update `__init__.py`

**Files:**
- Modify: `backend/app/src/herbs_detection/__init__.py`

- [ ] **Step 1: Replace contents**

```python
# backend/app/src/herbs_detection/__init__.py
from .model_registry import MODEL_REGISTRY, REGISTRY_BY_KEY, ENABLED_KEYS
from .timm_predictor import TimmPredictor
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/src/herbs_detection/__init__.py
git commit -m "chore: export TimmPredictor and model registry from package"
```

---

## Task 5 - Rewrite `main.py`

**Files:**
- Modify: `backend/app/api/main.py`

**Response shapes:**

`POST /predict` single image:
```json
{
  "predictions": [
    {"model": "convnext_tiny", "top3": [{"class": "Basilic", "confidence": 0.95}]}
  ]
}
```

`POST /predict-batch` multiple images:
```json
[{"filename": "img1.jpg", "predictions": [{"model": "convnext_tiny", "class": "Basilic", "confidence": 0.95}]}]
```

`POST /explore` visual exploration:
```json
{
  "filename": "img1.jpg",
  "predictions": [
    {"model": "convnext_tiny", "top_k": [{"rank": 1, "class": "Basilic", "confidence": 0.95}]}
  ]
}
```

- [ ] **Step 1: Rewrite main.py**

```python
# backend/app/api/main.py
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger
import uvicorn

from herbs_detection.model_registry import MODEL_REGISTRY, REGISTRY_BY_KEY, ENABLED_KEYS
from herbs_detection.timm_predictor import TimmPredictor

_predictors: dict[str, TimmPredictor] = {}


def _load_all() -> None:
    for cfg in MODEL_REGISTRY:
        if cfg.enabled:
            p = TimmPredictor(cfg)
            _predictors[cfg.key] = p
            threading.Thread(target=p.load, daemon=True).start()


def _resolve_models(models_param: str) -> list[str]:
    if not models_param or models_param.strip().lower() == "all":
        return ENABLED_KEYS
    requested = [k.strip() for k in models_param.split(",") if k.strip()]
    unknown   = [k for k in requested if k not in REGISTRY_BY_KEY]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model key(s): {unknown}. Valid keys: {ENABLED_KEYS}",
        )
    return [k for k in requested if k in _predictors]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - loading models in background threads...")
    _load_all()
    yield
    logger.info("Shutting down.")


api = FastAPI(lifespan=lifespan)


@api.get("/")
def root():
    return {"message": "Plant Detect API", "models": ENABLED_KEYS}


@api.get("/models")
def list_models():
    return [
        {"key": cfg.key, "timm_name": cfg.timm_name, "img_size": cfg.img_size}
        for cfg in MODEL_REGISTRY if cfg.enabled
    ]


@api.post("/predict")
async def predict(
    file: UploadFile = File(...),
    models: str = Form(default="all"),
    top_k: int = Form(default=3),
):
    """Single image prediction. models='all' or comma-separated keys. top_k 1-10."""
    top_k = max(1, min(top_k, 10))
    keys  = _resolve_models(models)
    logger.info("predict | file={} models={} top_k={}", file.filename, keys, top_k)

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = [
            {
                "model": key,
                "top3": [
                    {"class": c, "confidence": conf}
                    for c, conf in _predictors[key].predict_top3(tmp_path)[:top_k]
                ],
            }
            for key in keys
        ]
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"predictions": results}


@api.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    models: str = Form(default="all"),
):
    """Batch prediction, top-1 per image per model."""
    keys = _resolve_models(models)
    logger.info("predict-batch | {} files models={}", len(files), keys)

    tmp_paths, filenames = [], []
    for f in files:
        suffix = Path(f.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await f.read())
            tmp_paths.append(tmp.name)
            filenames.append(f.filename)

    try:
        per_model = {key: _predictors[key].predict_set(tmp_paths) for key in keys}
    finally:
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)

    return [
        {
            "filename": fname,
            "predictions": [
                {"model": key,
                 "class":  per_model[key][i][0],
                 "confidence": per_model[key][i][1]}
                for key in keys
            ],
        }
        for i, fname in enumerate(filenames)
    ]


@api.post("/explore")
async def explore(
    file: UploadFile = File(...),
    models: str = Form(default="all"),
    top_k: int = Form(default=5),
):
    """
    Visual exploration: returns top-K predictions per model with rank.
    Designed for the side-by-side model comparison view in the frontend.
    """
    top_k = max(1, min(top_k, 10))
    keys  = _resolve_models(models)
    logger.info("explore | file={} models={} top_k={}", file.filename, keys, top_k)

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = [
            {
                "model": key,
                "top_k": [
                    {"rank": i + 1, "class": c, "confidence": conf}
                    for i, (c, conf) in enumerate(
                        _predictors[key].predict_top3(tmp_path)[:top_k]
                    )
                ],
            }
            for key in keys
        ]
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"filename": file.filename, "predictions": results}


if __name__ == "__main__":
    uvicorn.run("app.api.main:api", host="0.0.0.0", port=8080)
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/api/main.py
git commit -m "feat: rewrite API with 5 timm models, models= selection, /explore endpoint"
```

---

## Task 6 - Update Tests

**Files:**
- Modify: `backend/tests/conftest.py`
- Modify: `backend/tests/test_api.py`

- [ ] **Step 1: Rewrite conftest.py**

```python
# backend/tests/conftest.py
"""
All TimmPredictor instances replaced with stubs - no weights or GPU needed.
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
    from herbs_detection.model_registry import ENABLED_KEYS

    monkeypatch.setattr(api_mod, "_load_all", lambda: None)

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

- [ ] **Step 2: Rewrite test_api.py**

```python
# backend/tests/test_api.py
import pytest
from fastapi.testclient import TestClient

from app.api.main import api
from tests.conftest import make_image_bytes, TOP3_STUB, TOP1_STUB
from herbs_detection.model_registry import ENABLED_KEYS


@pytest.fixture
def client():
    with TestClient(api) as c:
        yield c


class TestRoot:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_lists_enabled_models(self, client):
        assert set(client.get("/").json()["models"]) == set(ENABLED_KEYS)


class TestListModels:
    def test_returns_all_enabled(self, client):
        data = client.get("/models").json()
        assert len(data) == len(ENABLED_KEYS)
        assert all("key" in m and "img_size" in m for m in data)


class TestPredict:
    def _post(self, client, models="all", top_k=3, suffix=".jpg"):
        img = make_image_bytes(suffix)
        return client.post(
            "/predict",
            files={"file": (f"plant{suffix}", img, "image/jpeg")},
            data={"models": models, "top_k": top_k},
        )

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_all_models_returned_by_default(self, client):
        returned = {p["model"] for p in self._post(client).json()["predictions"]}
        assert returned == set(ENABLED_KEYS)

    def test_single_model_selection(self, client):
        data = self._post(client, models="convnext_tiny").json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["model"] == "convnext_tiny"

    def test_multiple_model_selection(self, client):
        returned = {p["model"] for p in
                    self._post(client, models="convnext_tiny,resnet50").json()["predictions"]}
        assert returned == {"convnext_tiny", "resnet50"}

    def test_top3_default(self, client):
        assert len(self._post(client).json()["predictions"][0]["top3"]) == 3

    def test_top_k_one(self, client):
        assert len(self._post(client, top_k=1).json()["predictions"][0]["top3"]) == 1

    def test_prediction_shape(self, client):
        pred = self._post(client).json()["predictions"][0]
        assert "model" in pred and "top3" in pred
        assert "class" in pred["top3"][0] and "confidence" in pred["top3"][0]

    def test_unknown_model_returns_422(self, client):
        assert self._post(client, models="does_not_exist").status_code == 422

    def test_accepts_png(self, client):
        assert self._post(client, suffix=".png").status_code == 200


class TestPredictBatch:
    def _post(self, client, n=2, models="all"):
        img = make_image_bytes()
        files = [("files", (f"img{i}.jpg", img, "image/jpeg")) for i in range(n)]
        return client.post("/predict-batch", files=files, data={"models": models})

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_one_result_per_image(self, client):
        assert len(self._post(client, n=3).json()) == 3

    def test_result_has_filename(self, client):
        assert self._post(client, n=1).json()[0]["filename"] == "img0.jpg"

    def test_all_models_present(self, client):
        returned = {p["model"] for p in self._post(client, n=1).json()[0]["predictions"]}
        assert returned == set(ENABLED_KEYS)

    def test_model_selection(self, client):
        result = self._post(client, n=1, models="convnext_tiny").json()[0]
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["model"] == "convnext_tiny"


class TestExplore:
    def _post(self, client, top_k=3):
        img = make_image_bytes()
        return client.post(
            "/explore",
            files={"file": ("plant.jpg", img, "image/jpeg")},
            data={"top_k": top_k},
        )

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_contains_all_models(self, client):
        returned = {p["model"] for p in self._post(client).json()["predictions"]}
        assert returned == set(ENABLED_KEYS)

    def test_filename_in_response(self, client):
        assert self._post(client).json()["filename"] == "plant.jpg"

    def test_each_entry_has_rank(self, client):
        entries = self._post(client).json()["predictions"][0]["top_k"]
        assert entries[0]["rank"] == 1
        assert entries[1]["rank"] == 2
```

- [ ] **Step 3: Run the full test suite**

```bash
cd backend && pytest -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/conftest.py backend/tests/test_api.py
git commit -m "test: update full suite for 5-model registry and models= param"
```

---

## Task 7 - Local Development Setup

- [ ] **Step 1: Add wandb and timm to dev requirements**

Edit `backend/requirements-dev.txt` to add:
```
timm>=0.9.0
wandb>=0.17.0
```

Then install:
```bash
pip install -r backend/requirements-dev.txt
```

- [ ] **Step 2: Add credentials to `.env`**

```
WANDB_API_KEY=<your key from wandb.ai/settings>
WANDB_PROJECT=certification
WANDB_ENTITY=<your wandb username>
WANDB_CACHE_MAX_AGE_SECONDS=10800
```

- [ ] **Step 3: (Optional) Copy local checkpoints for offline use**

```bash
cp /path/to/convnext_tiny_best.pth     ./convnext_tiny_best.pth
cp /path/to/efficientnet_b3_best.pth   ./efficientnet_b3_best.pth
cp /path/to/efficientnet_b4_best.pth   ./efficientnet_b4_best.pth
cp /path/to/mobilenetv3_large_best.pth ./mobilenetv3_large_best.pth
cp /path/to/resnet50_best.pth          ./resnet50_best.pth
```

A `classes.txt` (59 class names, one per line, in training order) or `label_encoder.pkl` must also be present in each artifact directory alongside the `.pth`.

- [ ] **Step 4: Start the server**

```bash
PYTHONPATH=backend/app/src uvicorn backend.app.api.main:api --reload --host 0.0.0.0 --port 8080
```

Expected startup log:
```
INFO: Starting up - loading models in background threads...
INFO: Application startup complete.
INFO: Downloading wandb artifact: convnext_tiny_best
INFO: convnext_tiny ready. device=cpu classes=59
```

- [ ] **Step 5: Smoke test all endpoints**

```bash
# List available models
curl http://localhost:8080/models

# Single prediction - all 5 models
curl -X POST http://localhost:8080/predict \
  -F "file=@data/raw/all_images/dill_0.jpg"

# Single prediction - top-2 models, 1 result each
curl -X POST http://localhost:8080/predict \
  -F "file=@data/raw/all_images/dill_0.jpg" \
  -F "models=convnext_tiny,efficientnet_b3" \
  -F "top_k=1"

# Visual exploration - top-5 per model
curl -X POST http://localhost:8080/explore \
  -F "file=@data/raw/all_images/dill_0.jpg"

# Batch - winner model only
curl -X POST http://localhost:8080/predict-batch \
  -F "files=@data/raw/all_images/dill_0.jpg" \
  -F "files=@data/raw/all_images/dill_0.jpg" \
  -F "models=convnext_tiny"
```

- [ ] **Step 6: Commit dev requirements**

```bash
git add backend/requirements-dev.txt
git commit -m "chore: add timm and wandb to dev requirements"
```

---

## Self-Review

### Spec coverage

| Requirement | Task(s) |
|---|---|
| Keep all 5 models | Task 1 registry, Task 5 `_load_all` |
| Display results for all 5 | Task 5 `models=all` default |
| User chooses how many models | Task 5 `models=` form param + `top_k` |
| wandb replaces GCS | Task 2 `wandb_loader.py` |
| Local testing before Cloud Run | Task 7 |
| Visual exploration endpoint | Task 5 `/explore` |
| All tests pass | Tasks 2, 3, 6 |

### Known limitations to address as follow-up

- `predict_top3` returns exactly 3 items. If `/explore` requests `top_k > 3`, results are silently capped. Extend to accept a `k` parameter when the frontend actually requests more.
- Each wandb artifact must contain either a `label_encoder.pkl` or a `classes.txt` alongside the `.pth`. If training did not save one, re-upload the artifact with the file added.
- Old `model_*.py`, `model_*_gcs.py`, and `gcs_cache.py` are left in place. Remove in a cleanup commit once the new stack is confirmed working end-to-end.

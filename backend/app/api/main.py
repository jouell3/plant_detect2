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

"""
Resolve model artifacts from the local filesystem.

The backend now treats the checked-in or mounted model directory as the
authoritative source and fails fast if a required artifact is missing.
"""
import os
import shutil
import threading
import time
from pathlib import Path

from loguru import logger

_CACHE_MAX_AGE = int(os.getenv("WANDB_CACHE_MAX_AGE_SECONDS", str(3 * 3600)))
_WANDB_PROJECT = os.getenv("WANDB_PROJECT", "certification")
_WANDB_ENTITY  = os.getenv("WANDB_ENTITY", "")

_artifact_locks: dict[str, threading.Lock] = {}
_artifact_locks_lock = threading.Lock()


def _get_artifact_lock(name: str) -> threading.Lock:
    with _artifact_locks_lock:
        if name not in _artifact_locks:
            _artifact_locks[name] = threading.Lock()
        return _artifact_locks[name]


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


def artifact_local_path(artifact_name: str, cache_root: Path | None = None,
                        artifact_type: str = "model") -> Path:
    """
    Return a local directory containing the artifact files.

    Resolution order:
    1. Local artifact directory containing the expected file
    2. Flat file at the cache root, copied into the artifact directory
    3. Raise FileNotFoundError
    """
    if cache_root is None:
        cache_root = Path(os.getenv("MODEL_PATH", str(Path.cwd() / "models" / "wandb")))
    else:
        cache_root = Path(cache_root)

    # Validate artifact_name to prevent path traversal attacks
    safe_name = Path(artifact_name).name
    if not safe_name or safe_name != artifact_name:
        raise ValueError(f"artifact_name must be a plain name, got: {artifact_name!r}")

    local_dir = cache_root / safe_name

    with _get_artifact_lock(safe_name):
        if artifact_type == "model":
            expected_name = f"{safe_name}.pth"
        elif artifact_type == "preprocessor":
            expected_name = f"{safe_name}.pkl"
        else:
            raise ValueError(f"Unsupported artifact_type: {artifact_type!r}")

        local_file = local_dir / expected_name
        if local_file.exists():
            logger.info("Using local {} artifact: {}", artifact_type, local_file)
            return local_dir

        flat_file = cache_root / expected_name
        if flat_file.exists():
            local_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(flat_file, local_file)
            logger.info("Copied local {} artifact into {}", artifact_type, local_dir)
            return local_dir

        raise FileNotFoundError(
            f"Local {artifact_type} artifact '{artifact_name}' not found. Expected either "
            f"{local_file} or {flat_file}."
        )
    return local_dir


def label_encoder_local_path(cache_root: Path | None = None) -> Path:
    """
    Return the local directory containing label_encoder.pkl.

    The encoder is shared across all models and is expected to exist under the
    same local model root as the classifiers.
    """
    return artifact_local_path("label_encoder", cache_root, artifact_type="preprocessor")

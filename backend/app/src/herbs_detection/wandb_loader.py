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

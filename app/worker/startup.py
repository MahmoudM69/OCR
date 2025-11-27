"""
Worker startup module for pre-downloading models.

This module handles downloading and caching models at worker startup
so they're ready when jobs arrive.
"""
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

from app.config import settings
from app.services.model_status import (
    ModelStatus,
    get_model_status_service,
)

logger = logging.getLogger(__name__)

# Unique worker ID for this process
WORKER_ID = str(uuid.uuid4())

# Model ID mapping - maps engine names to HuggingFace model IDs
MODEL_IDS = {
    "qari": "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
    "got": "stepfun-ai/GOT-OCR2_0",
    "deepseek": "unsloth/DeepSeek-OCR",
}


def _get_model_cache_dir(model_id: str) -> Path:
    """Get the HuggingFace cache directory for a model."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_cache = Path(hf_home) / "hub"
    cache_dir_name = f"models--{model_id.replace('/', '--')}"
    return hub_cache / cache_dir_name


def _validate_model_download(model_id: str) -> bool:
    """
    Validate that a model was fully downloaded.

    Checks for the presence of model weight files (.safetensors or .bin).

    Args:
        model_id: HuggingFace model ID.

    Returns:
        True if model appears complete, False otherwise.
    """
    model_cache_dir = _get_model_cache_dir(model_id)
    blobs_dir = model_cache_dir / "blobs"

    if not blobs_dir.exists():
        logger.warning(f"No blobs directory found for {model_id}")
        return False

    # Check for incomplete files
    incomplete_files = list(blobs_dir.glob("*.incomplete"))
    if incomplete_files:
        logger.warning(f"Found {len(incomplete_files)} incomplete files for {model_id}")
        return False

    # Check total size of blobs - models should be at least 100MB
    total_size = sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file())
    min_size_mb = 100  # Minimum expected size in MB

    if total_size < min_size_mb * 1024 * 1024:
        logger.warning(
            f"Model {model_id} cache too small: {total_size / 1024 / 1024:.1f}MB "
            f"(expected at least {min_size_mb}MB)"
        )
        return False

    logger.info(f"Model {model_id} validated: {total_size / 1024 / 1024:.1f}MB in cache")
    return True


def _cleanup_incomplete_downloads(model_id: str) -> int:
    """
    Remove incomplete download files for a model.

    HuggingFace Hub creates .incomplete files during download.
    If the download is interrupted, these files can cause issues
    on retry.

    Args:
        model_id: HuggingFace model ID (e.g., 'unsloth/DeepSeek-OCR').

    Returns:
        Number of incomplete files removed.
    """
    model_cache_dir = _get_model_cache_dir(model_id)
    blobs_dir = model_cache_dir / "blobs"

    if not blobs_dir.exists():
        return 0

    removed = 0
    for incomplete_file in blobs_dir.glob("*.incomplete"):
        try:
            incomplete_file.unlink()
            logger.info(f"Removed incomplete download: {incomplete_file.name}")
            removed += 1
        except Exception as e:
            logger.warning(f"Failed to remove {incomplete_file}: {e}")

    if removed > 0:
        print(f"[STARTUP] Cleaned up {removed} incomplete download(s) for {model_id}", flush=True)

    return removed


def download_model(
    model_name: str,
    progress_callback: Callable[[float], None] | None = None,
) -> bool:
    """
    Download a model from HuggingFace Hub.

    Uses a distributed lock to prevent multiple workers from
    downloading the same model simultaneously.

    Args:
        model_name: Name of the model (e.g., 'qari', 'got').
        progress_callback: Optional callback for progress updates.

    Returns:
        True if download succeeded, False otherwise.
    """
    model_status_service = get_model_status_service()

    if model_name not in MODEL_IDS:
        logger.warning(f"Unknown model: {model_name}")
        model_status_service.set_status(
            model_name,
            ModelStatus.FAILED,
            f"Unknown model: {model_name}",
        )
        return False

    # Check if already ready
    if model_status_service.is_ready(model_name):
        print(f"[STARTUP] Model '{model_name}' already ready, skipping download", flush=True)
        return True

    # Try to acquire download lock
    if not model_status_service.acquire_download_lock(model_name, WORKER_ID):
        print(f"[STARTUP] Model '{model_name}' is being downloaded by another worker", flush=True)
        return False

    model_id = MODEL_IDS[model_name]

    # Set status to downloading (with TTL)
    model_status_service.set_status(model_name, ModelStatus.DOWNLOADING)
    model_status_service.set_progress(model_name, 0.0)

    try:
        print(f"[STARTUP] Downloading model '{model_name}' ({model_id})...", flush=True)

        # Clean up any incomplete downloads from previous attempts
        _cleanup_incomplete_downloads(model_id)

        # Download model (automatically resumes if interrupted)
        # force_download=True would skip cache and re-download everything
        local_dir = snapshot_download(
            model_id,
            local_files_only=False,
        )

        # Validate the download by checking for model weights
        if not _validate_model_download(model_id):
            raise RuntimeError(
                f"Model download incomplete - missing weight files. "
                f"Try deleting the cache and re-downloading."
            )

        # Mark as ready
        model_status_service.set_status(model_name, ModelStatus.READY)
        model_status_service.set_progress(model_name, 100.0)
        print(f"[STARTUP] Model '{model_name}' downloaded successfully to {local_dir}!", flush=True)
        return True

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to download model '{model_name}': {error_msg}")
        model_status_service.set_status(
            model_name,
            ModelStatus.FAILED,
            error_msg,
        )
        print(f"[STARTUP] Failed to download model '{model_name}': {error_msg}", flush=True)
        return False

    finally:
        # Always release the lock when done
        model_status_service.release_download_lock(model_name, WORKER_ID)


def preload_models_sync() -> None:
    """
    Synchronously download all configured models.

    Downloads models in the order specified in settings.preload_models.
    """
    models_to_load = settings.preload_models
    print(f"[STARTUP] Pre-loading models: {models_to_load}", flush=True)

    for model_name in models_to_load:
        download_model(model_name)

    print("[STARTUP] Model pre-loading complete!", flush=True)


def preload_models_async() -> threading.Thread:
    """
    Start model downloading in a background thread.

    Returns:
        The background thread (already started).
    """
    thread = threading.Thread(
        target=preload_models_sync,
        daemon=True,
        name="model-preloader",
    )
    thread.start()
    return thread


def cleanup_stale_downloads() -> None:
    """
    Clean up any stale downloading statuses from previous runs.

    This handles cases where a worker crashed mid-download.
    """
    model_status_service = get_model_status_service()
    cleaned = model_status_service.cleanup_stale_statuses()

    if cleaned:
        print(f"[STARTUP] Cleaned up stale statuses for: {cleaned}", flush=True)
    else:
        print("[STARTUP] No stale download statuses found", flush=True)


def init_worker() -> None:
    """
    Initialize the worker with model pre-loading.

    Call this at worker startup to begin downloading models
    in the background.
    """
    print(f"[STARTUP] Initializing worker (ID: {WORKER_ID})...", flush=True)

    # Clean up any stale statuses from previous crashes
    cleanup_stale_downloads()

    # Start model download in background
    preload_models_async()

    print("[STARTUP] Worker initialization started", flush=True)

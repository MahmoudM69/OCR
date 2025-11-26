"""
Worker startup module for pre-downloading models.

This module handles downloading and caching models at worker startup
so they're ready when jobs arrive.
"""
import logging
import threading
from typing import Callable

from huggingface_hub import snapshot_download

from app.config import settings
from app.services.model_status import (
    ModelStatus,
    get_model_status_service,
)

logger = logging.getLogger(__name__)

# Model ID mapping - maps engine names to HuggingFace model IDs
MODEL_IDS = {
    "qari": "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
    "got": "stepfun-ai/GOT-OCR2_0",
    "deepseek": "unsloth/DeepSeek-OCR",
}


def download_model(
    model_name: str,
    progress_callback: Callable[[float], None] | None = None,
) -> bool:
    """
    Download a model from HuggingFace Hub.

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

    model_id = MODEL_IDS[model_name]

    # Set status to downloading
    model_status_service.set_status(model_name, ModelStatus.DOWNLOADING)
    model_status_service.set_progress(model_name, 0.0)

    try:
        print(f"[STARTUP] Downloading model '{model_name}' ({model_id})...", flush=True)

        # Download with progress tracking
        snapshot_download(
            model_id,
            local_files_only=False,
        )

        # Mark as ready
        model_status_service.set_status(model_name, ModelStatus.READY)
        model_status_service.set_progress(model_name, 100.0)
        print(f"[STARTUP] Model '{model_name}' downloaded successfully!", flush=True)
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


def init_worker() -> None:
    """
    Initialize the worker with model pre-loading.

    Call this at worker startup to begin downloading models
    in the background.
    """
    print("[STARTUP] Initializing worker...", flush=True)

    # Start model download in background
    preload_models_async()

    print("[STARTUP] Worker initialization started", flush=True)

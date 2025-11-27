import asyncio
import gc
import logging
from pathlib import Path

import redis

from app.config import settings
from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry
from app.services.model_status import ModelStatus, get_model_status_service

logger = logging.getLogger(__name__)

# Redis key for tracking current model across processes
CURRENT_MODEL_KEY = "ocr:current_model"


class OCRModelManager:
    """
    Manages OCR model lifecycle and runtime switching.

    This class handles:
    - Loading and unloading models
    - Graceful model switching with memory cleanup
    - Tracking the currently active model
    - Discovering available models from the models directory
    """

    def __init__(self, models_dir: Path | None = None, redis_client: redis.Redis | None = None):
        """
        Initialize the model manager.

        Args:
            models_dir: Directory containing model subdirectories.
                       Defaults to settings.models_dir.
            redis_client: Redis client for state synchronization.
        """
        self.models_dir = models_dir or settings.models_dir
        self.redis = redis_client or redis.Redis.from_url(settings.redis_url)
        self._current_engine: BaseOCREngine | None = None
        self._lock = asyncio.Lock()
        self._processing = False

    @property
    def current_model(self) -> str | None:
        """Get the name of the currently loaded model from Redis."""
        model = self.redis.get(CURRENT_MODEL_KEY)
        if model:
            return model.decode('utf-8')
        return None

    @property
    def current_model_local(self) -> str | None:
        """Get the name of the locally loaded model (this process only)."""
        if self._current_engine and self._current_engine.is_loaded:
            return self._current_engine.name
        return None

    @property
    def is_processing(self) -> bool:
        """Check if currently processing an image."""
        return self._processing

    def list_available_models(self) -> list[str]:
        """
        List models available in the models directory.

        Scans the models directory for subdirectories that match
        registered engine names.

        Returns:
            List of available model names.
        """
        available = []

        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return available

        for path in self.models_dir.iterdir():
            if path.is_dir():
                model_name = path.name
                if OCREngineRegistry.is_registered(model_name):
                    available.append(model_name)
                else:
                    logger.debug(f"Found model dir '{model_name}' but no engine registered")

        return available

    async def load_model(self, model_name: str) -> None:
        """
        Load a specific model.

        Args:
            model_name: Name of the model to load.

        Raises:
            ValueError: If the model is not registered or not available.
            RuntimeError: If loading fails.
        """
        model_status_service = get_model_status_service()

        async with self._lock:
            if not OCREngineRegistry.is_registered(model_name):
                raise ValueError(f"No engine registered for model: {model_name}")

            # Model path for any local caching (models download from HuggingFace)
            model_path = self.models_dir / model_name
            model_path.mkdir(parents=True, exist_ok=True)

            engine = OCREngineRegistry.create_engine(model_name, model_path)
            if engine is None:
                model_status_service.set_status(model_name, ModelStatus.FAILED, "Failed to create engine")
                raise RuntimeError(f"Failed to create engine for: {model_name}")

            logger.info(f"Loading model: {model_name}")
            model_status_service.set_status(model_name, ModelStatus.LOADING)

            try:
                await engine.load()
            except Exception as e:
                model_status_service.set_status(model_name, ModelStatus.FAILED, str(e))
                raise

            self._current_engine = engine

            # Store current model in Redis for cross-process visibility
            self.redis.set(CURRENT_MODEL_KEY, model_name)
            model_status_service.set_status(model_name, ModelStatus.READY)

            logger.info(f"Model loaded successfully: {model_name}")

    async def unload_current(self) -> None:
        """Unload the currently loaded model and free memory."""
        async with self._lock:
            await self._unload_current_unsafe()

    async def _unload_current_unsafe(self) -> None:
        """Unload without acquiring lock (internal use)."""
        if self._current_engine is None:
            return

        model_name = self._current_engine.name
        logger.info(f"Unloading model: {model_name}")

        await self._current_engine.unload()
        self._current_engine = None

        # Clear from Redis
        self.redis.delete(CURRENT_MODEL_KEY)

        # Force garbage collection to free memory
        gc.collect()

        # Try to clear GPU memory if torch is available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
        except ImportError:
            pass

        logger.info(f"Model unloaded: {model_name}")

    async def switch_model(
        self, model_name: str, timeout: float = 30.0
    ) -> str | None:
        """
        Switch to a different model.

        This method:
        1. Waits for any current processing to complete
        2. Unloads the current model
        3. Cleans up memory
        4. Loads the new model

        Args:
            model_name: Name of the model to switch to.
            timeout: Maximum time to wait for current processing to finish.

        Returns:
            The name of the previous model, or None if no model was loaded.

        Raises:
            ValueError: If the model is not available.
            TimeoutError: If waiting for processing times out.
            RuntimeError: If switching fails.
        """
        model_status_service = get_model_status_service()

        # Validate model exists before waiting
        if not OCREngineRegistry.is_registered(model_name):
            raise ValueError(f"No engine registered for model: {model_name}")

        # Model path for any local caching (models download from HuggingFace)
        model_path = self.models_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        # Wait for current processing to finish
        wait_time = 0.0
        while self._processing and wait_time < timeout:
            await asyncio.sleep(0.1)
            wait_time += 0.1

        if self._processing:
            raise TimeoutError("Timed out waiting for current processing to complete")

        async with self._lock:
            # Store previous model name
            previous = self._current_engine.name if self._current_engine else None

            # Unload current model
            await self._unload_current_unsafe()

            # Load new model
            engine = OCREngineRegistry.create_engine(model_name, model_path)
            if engine is None:
                model_status_service.set_status(model_name, ModelStatus.FAILED, "Failed to create engine")
                raise RuntimeError(f"Failed to create engine for: {model_name}")

            logger.info(f"Loading model: {model_name}")
            model_status_service.set_status(model_name, ModelStatus.LOADING)

            try:
                await engine.load()
            except Exception as e:
                model_status_service.set_status(model_name, ModelStatus.FAILED, str(e))
                raise

            self._current_engine = engine

            # Store current model in Redis
            self.redis.set(CURRENT_MODEL_KEY, model_name)
            model_status_service.set_status(model_name, ModelStatus.READY)

            logger.info(f"Switched to model: {model_name}")

            return previous

    async def ensure_model_loaded(self) -> None:
        """
        Ensure the current model (from Redis) is loaded locally.

        This is called by the worker to sync with model switch commands.
        """
        target_model = self.current_model
        local_model = self.current_model_local

        if target_model is None:
            # No model should be loaded
            if local_model is not None:
                await self.unload_current()
            return

        if local_model != target_model:
            # Need to switch models
            logger.info(f"Syncing model: {local_model} -> {target_model}")
            await self.switch_model(target_model)

    async def extract_text(self, image_path: Path, engine: str | None = None) -> OCRResult:
        """
        Extract text from an image using the specified or current model.

        Args:
            image_path: Path to the image file.
            engine: Specific engine to use. If None, uses the current model from Redis.

        Returns:
            OCRResult containing extracted text.

        Raises:
            RuntimeError: If no model is loaded.
        """
        # If specific engine requested, ensure it's loaded
        if engine:
            local_model = self.current_model_local
            if local_model != engine:
                logger.info(f"Switching to requested engine: {engine}")
                await self.switch_model(engine)
        else:
            # Ensure we have the right model loaded from Redis
            await self.ensure_model_loaded()

        if self._current_engine is None or not self._current_engine.is_loaded:
            raise RuntimeError("No model is currently loaded")

        self._processing = True
        try:
            return await self._current_engine.extract_text(image_path)
        finally:
            self._processing = False


# Global instance for the worker process
_manager: OCRModelManager | None = None


def get_model_manager() -> OCRModelManager:
    """Get or create the global model manager instance."""
    global _manager
    if _manager is None:
        _manager = OCRModelManager()
    return _manager

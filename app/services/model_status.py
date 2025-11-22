"""
Service for tracking model download and readiness status.

Uses Redis to store model status across API and Worker processes.
"""
import json
import logging
from enum import Enum
from typing import Optional

import redis

from app.config import settings

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Status of a model."""
    NOT_FOUND = "not_found"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"
    LOADING = "loading"  # Downloaded but loading into GPU


class ModelStatusService:
    """Service for managing model status in Redis."""

    STATUS_PREFIX = "model:status:"
    PROGRESS_PREFIX = "model:progress:"

    def __init__(self, redis_client: redis.Redis | None = None):
        """Initialize the model status service."""
        self.redis = redis_client or redis.Redis.from_url(settings.redis_url)

    def _status_key(self, model_name: str) -> str:
        """Generate Redis key for model status."""
        return f"{self.STATUS_PREFIX}{model_name}"

    def _progress_key(self, model_name: str) -> str:
        """Generate Redis key for download progress."""
        return f"{self.PROGRESS_PREFIX}{model_name}"

    def set_status(
        self,
        model_name: str,
        status: ModelStatus,
        message: str = "",
    ) -> None:
        """
        Set the status of a model.

        Args:
            model_name: Name of the model.
            status: Current status.
            message: Optional message (e.g., error details).
        """
        data = {
            "status": status.value,
            "message": message,
        }
        self.redis.set(self._status_key(model_name), json.dumps(data))
        logger.info(f"Model '{model_name}' status: {status.value}")

    def get_status(self, model_name: str) -> tuple[ModelStatus, str]:
        """
        Get the status of a model.

        Args:
            model_name: Name of the model.

        Returns:
            Tuple of (status, message).
        """
        data = self.redis.get(self._status_key(model_name))
        if data is None:
            return ModelStatus.NOT_FOUND, ""

        try:
            parsed = json.loads(data)
            return ModelStatus(parsed["status"]), parsed.get("message", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            return ModelStatus.NOT_FOUND, ""

    def set_progress(self, model_name: str, progress: float) -> None:
        """
        Set download progress percentage.

        Args:
            model_name: Name of the model.
            progress: Progress percentage (0-100).
        """
        self.redis.set(self._progress_key(model_name), str(progress))

    def get_progress(self, model_name: str) -> float:
        """
        Get download progress percentage.

        Args:
            model_name: Name of the model.

        Returns:
            Progress percentage (0-100), or 0 if not found.
        """
        data = self.redis.get(self._progress_key(model_name))
        if data is None:
            return 0.0
        try:
            return float(data)
        except ValueError:
            return 0.0

    def is_ready(self, model_name: str) -> bool:
        """Check if a model is ready for use."""
        status, _ = self.get_status(model_name)
        return status == ModelStatus.READY

    def get_all_statuses(self) -> dict[str, dict]:
        """
        Get status of all tracked models.

        Returns:
            Dictionary mapping model names to their status info.
        """
        result = {}
        pattern = f"{self.STATUS_PREFIX}*"

        for key in self.redis.scan_iter(pattern):
            model_name = key.decode().replace(self.STATUS_PREFIX, "")
            status, message = self.get_status(model_name)
            progress = self.get_progress(model_name)

            result[model_name] = {
                "status": status.value,
                "message": message,
                "progress": progress,
            }

        return result


# Singleton instance
_model_status_service: ModelStatusService | None = None


def get_model_status_service() -> ModelStatusService:
    """Get or create the model status service singleton."""
    global _model_status_service
    if _model_status_service is None:
        _model_status_service = ModelStatusService()
    return _model_status_service

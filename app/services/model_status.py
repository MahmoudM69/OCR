"""
Service for tracking model download and readiness status.

Uses Redis to store model status across API and Worker processes.
"""
import json
import logging
import time
from enum import Enum

import redis

from app.config import settings

logger = logging.getLogger(__name__)

# TTL for transient statuses (downloading, loading) - 10 minutes
TRANSIENT_STATUS_TTL_SECONDS = 600

# Download lock TTL - 15 minutes (should be longer than expected download time)
DOWNLOAD_LOCK_TTL_SECONDS = 900


class ModelStatus(str, Enum):
    """Status of a model."""
    NOT_FOUND = "not_found"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"
    LOADING = "loading"  # Downloaded but loading into GPU


# Statuses that should auto-expire if not refreshed
TRANSIENT_STATUSES = {ModelStatus.DOWNLOADING, ModelStatus.LOADING}


class ModelStatusService:
    """Service for managing model status in Redis."""

    STATUS_PREFIX = "model:status:"
    PROGRESS_PREFIX = "model:progress:"
    LOCK_PREFIX = "model:download_lock:"

    def __init__(self, redis_client: redis.Redis | None = None):
        """Initialize the model status service."""
        self.redis = redis_client or redis.Redis.from_url(settings.redis_url)

    def _status_key(self, model_name: str) -> str:
        """Generate Redis key for model status."""
        return f"{self.STATUS_PREFIX}{model_name}"

    def _progress_key(self, model_name: str) -> str:
        """Generate Redis key for download progress."""
        return f"{self.PROGRESS_PREFIX}{model_name}"

    def _lock_key(self, model_name: str) -> str:
        """Generate Redis key for download lock."""
        return f"{self.LOCK_PREFIX}{model_name}"

    def set_status(
        self,
        model_name: str,
        status: ModelStatus,
        message: str = "",
    ) -> None:
        """
        Set the status of a model.

        Transient statuses (downloading, loading) are set with TTL
        so they auto-expire if not refreshed.

        Args:
            model_name: Name of the model.
            status: Current status.
            message: Optional message (e.g., error details).
        """
        data = {
            "status": status.value,
            "message": message,
            "timestamp": time.time(),
        }
        key = self._status_key(model_name)

        if status in TRANSIENT_STATUSES:
            # Transient statuses get TTL - they auto-expire if not refreshed
            self.redis.set(
                key,
                json.dumps(data),
                ex=TRANSIENT_STATUS_TTL_SECONDS,
            )
        else:
            # Permanent statuses (ready, failed) persist indefinitely
            self.redis.set(key, json.dumps(data))

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

    def acquire_download_lock(self, model_name: str, worker_id: str) -> bool:
        """
        Attempt to acquire a download lock for a model.

        Uses Redis SETNX for atomic lock acquisition.

        Args:
            model_name: Name of the model to lock.
            worker_id: Unique identifier for this worker.

        Returns:
            True if lock acquired, False if already locked by another worker.
        """
        lock_key = self._lock_key(model_name)
        lock_data = json.dumps({
            "worker_id": worker_id,
            "acquired_at": time.time(),
        })

        # SETNX with expiration - atomic operation
        acquired = self.redis.set(
            lock_key,
            lock_data,
            nx=True,  # Only set if not exists
            ex=DOWNLOAD_LOCK_TTL_SECONDS,
        )

        if acquired:
            logger.info(f"Acquired download lock for '{model_name}' (worker: {worker_id})")
        else:
            logger.debug(f"Download lock for '{model_name}' held by another worker")

        return bool(acquired)

    def release_download_lock(self, model_name: str, worker_id: str) -> bool:
        """
        Release a download lock if owned by this worker.

        Args:
            model_name: Name of the model.
            worker_id: Worker that should own the lock.

        Returns:
            True if lock was released, False if not owned by this worker.
        """
        lock_key = self._lock_key(model_name)
        lock_data = self.redis.get(lock_key)

        if lock_data is None:
            return True  # No lock exists

        try:
            parsed = json.loads(lock_data)
            if parsed.get("worker_id") == worker_id:
                self.redis.delete(lock_key)
                logger.info(f"Released download lock for '{model_name}'")
                return True
            else:
                logger.warning(
                    f"Cannot release lock for '{model_name}': "
                    f"owned by {parsed.get('worker_id')}, not {worker_id}"
                )
                return False
        except (json.JSONDecodeError, KeyError):
            # Corrupted lock data, delete it
            self.redis.delete(lock_key)
            return True

    def refresh_download_status(self, model_name: str, progress: float) -> None:
        """
        Refresh download status TTL and update progress.

        Call this periodically during download to prevent TTL expiration.

        Args:
            model_name: Name of the model.
            progress: Current download progress (0-100).
        """
        # Re-set the status to refresh TTL
        self.set_status(model_name, ModelStatus.DOWNLOADING)
        self.set_progress(model_name, progress)

    def cleanup_stale_statuses(self) -> list[str]:
        """
        Clean up any stale downloading/loading statuses on startup.

        This handles cases where a worker crashed mid-download.
        Redis TTL should handle most cases, but this provides
        immediate cleanup on startup.

        Returns:
            List of model names that were cleaned up.
        """
        cleaned = []
        pattern = f"{self.STATUS_PREFIX}*"

        for key in self.redis.scan_iter(pattern):
            model_name = key.decode().replace(self.STATUS_PREFIX, "")
            status, _ = self.get_status(model_name)

            if status in TRANSIENT_STATUSES:
                # Check if there's an active lock
                lock_key = self._lock_key(model_name)
                if not self.redis.exists(lock_key):
                    # No active lock but transient status - stale
                    logger.warning(
                        f"Cleaning up stale '{status.value}' status for '{model_name}'"
                    )
                    self.redis.delete(key)
                    self.redis.delete(self._progress_key(model_name))
                    cleaned.append(model_name)

        return cleaned

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

import logging

import httpx

from app.config import settings
from app.schemas.job import Job

logger = logging.getLogger(__name__)


class WebhookService:
    """Service for delivering webhook notifications."""

    def __init__(
        self,
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        """
        Initialize the webhook service.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
        """
        self.timeout = timeout or settings.webhook_timeout
        self.max_retries = max_retries or settings.webhook_max_retries

    async def deliver(self, job: Job) -> bool:
        """
        Deliver a webhook notification for a completed job.

        Args:
            job: The completed job to notify about.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        if not job.webhook_url:
            return True

        payload = {
            "job_id": job.id,
            "status": job.status.value,
            "filename": job.filename,
            "result": job.result.model_dump() if job.result else None,
            "error": job.error,
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        job.webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    logger.info(f"Webhook delivered for job {job.id}")
                    return True

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Webhook failed for job {job.id} (attempt {attempt + 1}): "
                    f"HTTP {e.response.status_code}"
                )
            except httpx.RequestError as e:
                logger.warning(
                    f"Webhook failed for job {job.id} (attempt {attempt + 1}): {e}"
                )

        logger.error(f"Webhook delivery failed for job {job.id} after {self.max_retries} attempts")
        return False


# Singleton instance
_webhook_service: WebhookService | None = None


def get_webhook_service() -> WebhookService:
    """Get or create the webhook service singleton."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service

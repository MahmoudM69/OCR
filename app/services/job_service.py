import logging
import time
import uuid
from datetime import datetime

import redis

from app.config import settings
from app.schemas.job import Job, JobResult, JobStatus

logger = logging.getLogger(__name__)

# Jobs stuck in "processing" for longer than this are considered stale
STALE_PROCESSING_THRESHOLD_SECONDS = 300  # 5 minutes


class JobService:
    """
    Service for managing OCR jobs in Redis.

    Handles job creation, status updates, and result storage.
    """

    JOB_PREFIX = "ocr:job:"

    def __init__(self, redis_client: redis.Redis | None = None):
        """
        Initialize the job service.

        Args:
            redis_client: Redis client instance. Creates one if not provided.
        """
        self.redis = redis_client or redis.Redis.from_url(settings.redis_url)

    def _job_key(self, job_id: str) -> str:
        """Generate Redis key for a job."""
        return f"{self.JOB_PREFIX}{job_id}"

    def _serialize_job(self, job: Job) -> str:
        """Serialize a job to JSON for storage."""
        return job.model_dump_json()

    def _deserialize_job(self, data: str) -> Job:
        """Deserialize a job from JSON."""
        return Job.model_validate_json(data)

    def create_job(
        self,
        filename: str | None = None,
        webhook_url: str | None = None,
    ) -> Job:
        """
        Create a new OCR job.

        Args:
            filename: Original filename of the uploaded image.
            webhook_url: Optional webhook URL to call on completion.

        Returns:
            The created job.
        """
        now = datetime.utcnow()
        job = Job(
            id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            filename=filename,
            webhook_url=webhook_url,
        )

        self.redis.setex(
            self._job_key(job.id),
            settings.job_result_ttl,
            self._serialize_job(job),
        )

        return job

    def get_job(self, job_id: str) -> Job | None:
        """
        Get a job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            The job if found, None otherwise.
        """
        data = self.redis.get(self._job_key(job_id))
        if data is None:
            return None
        return self._deserialize_job(data)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: str | None = None,
    ) -> Job | None:
        """
        Update a job's status.

        Args:
            job_id: The job identifier.
            status: New status.
            error: Optional error message (for failed status).

        Returns:
            The updated job, or None if not found.
        """
        job = self.get_job(job_id)
        if job is None:
            return None

        job.status = status
        job.updated_at = datetime.utcnow()
        if error:
            job.error = error

        self.redis.setex(
            self._job_key(job.id),
            settings.job_result_ttl,
            self._serialize_job(job),
        )

        return job

    def set_result(
        self,
        job_id: str,
        text: str,
        formatted_text: str = "",
        output_file_base64: str | None = None,
        confidence: float = 0.0,
        metadata: dict | None = None,
    ) -> Job | None:
        """
        Set the result of an OCR job.

        Args:
            job_id: The job identifier.
            text: Extracted text (plain).
            formatted_text: Formatted text output from model.
            output_file_base64: Base64-encoded content of the output text file.
            confidence: Confidence score.
            metadata: Additional metadata.

        Returns:
            The updated job, or None if not found.
        """
        job = self.get_job(job_id)
        if job is None:
            return None

        job.status = JobStatus.COMPLETED
        job.updated_at = datetime.utcnow()
        job.result = JobResult(
            text=text,
            formatted_text=formatted_text,
            output_file_base64=output_file_base64,
            confidence=confidence,
            metadata=metadata or {},
        )

        self.redis.setex(
            self._job_key(job.id),
            settings.job_result_ttl,
            self._serialize_job(job),
        )

        return job

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job.

        Args:
            job_id: The job identifier.

        Returns:
            True if deleted, False if not found.
        """
        return self.redis.delete(self._job_key(job_id)) > 0

    def cleanup_stale_processing_jobs(self) -> list[str]:
        """
        Clean up jobs stuck in "processing" status.

        This handles cases where the worker crashed mid-processing
        and the job status was never updated to failed.

        Returns:
            List of job IDs that were cleaned up.
        """
        cleaned = []
        pattern = f"{self.JOB_PREFIX}*"
        now = time.time()

        for key in self.redis.scan_iter(pattern):
            try:
                data = self.redis.get(key)
                if data is None:
                    continue

                job = self._deserialize_job(data)

                if job.status == JobStatus.PROCESSING:
                    # Check if job has been processing too long
                    updated_timestamp = job.updated_at.timestamp()
                    age_seconds = now - updated_timestamp

                    if age_seconds > STALE_PROCESSING_THRESHOLD_SECONDS:
                        logger.warning(
                            f"Cleaning up stale processing job {job.id} "
                            f"(stuck for {age_seconds:.0f}s)"
                        )
                        self.update_status(
                            job.id,
                            JobStatus.FAILED,
                            error="Job timed out - worker may have crashed",
                        )
                        cleaned.append(job.id)

            except Exception as e:
                logger.error(f"Error checking job {key}: {e}")

        return cleaned


# Singleton instance
_job_service: JobService | None = None


def get_job_service() -> JobService:
    """Get or create the job service singleton."""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service

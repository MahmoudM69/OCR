import json
import uuid
from datetime import datetime

import redis

from app.config import settings
from app.schemas.job import Job, JobResult, JobStatus


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
        confidence: float = 0.0,
        metadata: dict | None = None,
    ) -> Job | None:
        """
        Set the result of an OCR job.

        Args:
            job_id: The job identifier.
            text: Extracted text.
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


# Singleton instance
_job_service: JobService | None = None


def get_job_service() -> JobService:
    """Get or create the job service singleton."""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service

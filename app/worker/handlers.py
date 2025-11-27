"""
RQ exception and failure handlers.

These handlers are called when jobs fail unexpectedly,
including when the worker process is killed.
"""
import logging
from typing import Any

from rq.job import Job

from app.schemas.job import JobStatus
from app.services.job_service import get_job_service

logger = logging.getLogger(__name__)


def handle_job_failure(job: Job, *args, **kwargs) -> bool:
    """
    Handle job failure - called when a job raises an exception or worker dies.

    This is registered as an RQ exception handler. The signature must be
    flexible because RQ calls it with different arguments depending on
    the failure type:
    - Exception: (job, connection, type, value, traceback)
    - Worker death/cleanup: (job, connection, exc_instance) or variations

    Args:
        job: The failed RQ job.
        *args: Additional arguments (connection, exc_info, etc.)

    Returns:
        False to indicate the exception was handled and should not propagate.
    """
    error_msg = "Job failed unexpectedly"

    try:
        # Try to extract exception info - RQ passes different args in different scenarios
        if len(args) >= 3:
            # Could be (connection, type, value, traceback) or (connection, exc, ...)
            exc_info = args[1]

            # Check if it's an exception type (class) or an exception instance
            if isinstance(exc_info, type) and issubclass(exc_info, BaseException):
                # It's a type - args are (connection, type, value, traceback)
                exc_type = exc_info
                exc_value = args[2] if len(args) > 2 else None
                error_msg = f"{exc_type.__name__}: {exc_value}"
            elif isinstance(exc_info, BaseException):
                # It's an exception instance directly
                error_msg = f"{type(exc_info).__name__}: {exc_info}"
            else:
                error_msg = f"Job failed: {exc_info}"

        elif len(args) >= 2:
            # Could be (connection, exc_instance)
            exc_info = args[1]
            if isinstance(exc_info, BaseException):
                error_msg = f"{type(exc_info).__name__}: {exc_info}"

        logger.error(f"Job {job.id} failed: {error_msg}")

    except Exception as e:
        # If we can't parse the exception, just log what we can
        logger.error(f"Job {job.id} failed (could not parse exception: {e})")
        error_msg = "Job failed (details unavailable)"

    # Extract OCR job ID from the job args
    # process_ocr_job(job_id, image_path, engine)
    try:
        if job.func_name == "app.worker.tasks.process_ocr_job" and job.args:
            ocr_job_id = job.args[0]
            _mark_job_failed(ocr_job_id, error_msg)
    except Exception as e:
        logger.error(f"Failed to mark job as failed: {e}")

    # Return False to indicate we handled it (don't re-raise)
    return False


def handle_worker_death(job: Job, connection: Any, *args, **kwargs) -> None:
    """
    Handle worker death - called when worker process is killed.

    This handles cases like OOM kills where no exception is raised.

    Args:
        job: The job that was being processed.
        connection: Redis connection.
    """
    logger.error(f"Worker died while processing job {job.id}")

    # Extract OCR job ID from the job args
    if job.func_name == "app.worker.tasks.process_ocr_job" and job.args:
        ocr_job_id = job.args[0]
        _mark_job_failed(ocr_job_id, "Worker process terminated unexpectedly (possible OOM)")


def _mark_job_failed(job_id: str, error_msg: str) -> None:
    """
    Mark an OCR job as failed.

    Args:
        job_id: The OCR job ID.
        error_msg: Error message to store.
    """
    try:
        job_service = get_job_service()
        job_service.update_status(job_id, JobStatus.FAILED, error=error_msg)
        logger.info(f"Marked OCR job {job_id} as failed: {error_msg}")
    except Exception as e:
        logger.error(f"Failed to mark job {job_id} as failed: {e}")

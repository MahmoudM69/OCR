import asyncio
import logging
from pathlib import Path

# Import OCR engines to register them
import app.ocr  # noqa: F401

from app.ocr.manager import get_model_manager
from app.schemas.job import JobStatus
from app.services.job_service import get_job_service
from app.services.webhook_service import get_webhook_service

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine in the sync RQ context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def process_ocr_job(job_id: str, image_path: str) -> dict:
    """
    Process an OCR job.

    This is the RQ task that runs in the worker process.

    Args:
        job_id: The job identifier.
        image_path: Path to the uploaded image.

    Returns:
        Dictionary with job result or error.
    """
    job_service = get_job_service()
    model_manager = get_model_manager()
    webhook_service = get_webhook_service()

    # Update status to processing
    job = job_service.update_status(job_id, JobStatus.PROCESSING)
    if job is None:
        logger.error(f"Job not found: {job_id}")
        return {"error": "Job not found"}

    try:
        # Run OCR
        result = _run_async(model_manager.extract_text(Path(image_path)))

        # Store result
        job = job_service.set_result(
            job_id,
            text=result.text,
            confidence=result.confidence,
            metadata=result.metadata,
        )

        logger.info(f"OCR completed for job {job_id}")

        # Send webhook if configured
        if job and job.webhook_url:
            _run_async(webhook_service.deliver(job))

        return {
            "job_id": job_id,
            "status": "completed",
            "text": result.text,
        }

    except RuntimeError as e:
        # Model not loaded or other runtime error
        error_msg = str(e)
        logger.error(f"OCR failed for job {job_id}: {error_msg}")

        job = job_service.update_status(job_id, JobStatus.FAILED, error=error_msg)

        if job and job.webhook_url:
            _run_async(webhook_service.deliver(job))

        return {"error": error_msg}

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.exception(f"OCR failed for job {job_id}")

        job = job_service.update_status(job_id, JobStatus.FAILED, error=error_msg)

        if job and job.webhook_url:
            _run_async(webhook_service.deliver(job))

        return {"error": error_msg}


def switch_model_task(model_name: str) -> dict:
    """
    Switch the OCR model.

    This task runs in the worker to switch models.

    Args:
        model_name: Name of the model to switch to.

    Returns:
        Dictionary with result or error.
    """
    model_manager = get_model_manager()

    try:
        previous = model_manager.current_model
        _run_async(model_manager.switch_model(model_name))

        return {
            "previous_model": previous,
            "current_model": model_name,
            "message": f"Switched from {previous or 'none'} to {model_name}",
        }

    except ValueError as e:
        return {"error": str(e)}
    except TimeoutError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception(f"Failed to switch model to {model_name}")
        return {"error": f"Failed to switch model: {str(e)}"}

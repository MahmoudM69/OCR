import asyncio
import base64
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


def _read_file_as_base64(file_path: str | None) -> str | None:
    """Read a file and return its content as base64."""
    if not file_path:
        return None
    try:
        path = Path(file_path)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            return base64.b64encode(content.encode("utf-8")).decode("ascii")
    except Exception as e:
        logger.warning(f"Failed to read output file {file_path}: {e}")
    return None


def process_ocr_job(job_id: str, image_path: str, engine: str | None = None) -> dict:
    """
    Process an OCR job.

    This is the RQ task that runs in the worker process.

    Args:
        job_id: The job identifier.
        image_path: Path to the uploaded image.
        engine: OCR engine to use (e.g., 'qari', 'got'). Defaults to config default.

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
        # Run OCR with specified engine
        result = _run_async(model_manager.extract_text(Path(image_path), engine=engine))

        # Read output file and convert to base64
        output_file_base64 = _read_file_as_base64(result.output_file)

        # Store result
        job = job_service.set_result(
            job_id,
            text=result.text,
            formatted_text=result.formatted_text,
            output_file_base64=output_file_base64,
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


def _cleanup_pdf_files(job_id: str, page_image_paths: list[str]) -> None:
    """
    Clean up PDF job files (page images and original PDF).

    Args:
        job_id: The job identifier (used to find original PDF).
        page_image_paths: List of paths to page images.
    """
    from app.config import settings

    # Clean up page images
    for page_path in page_image_paths:
        try:
            Path(page_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete page image {page_path}: {e}")

    # Clean up original PDF file
    pdf_path = settings.uploads_dir / f"{job_id}.pdf"
    try:
        pdf_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to delete PDF file {pdf_path}: {e}")


def process_pdf_ocr_job(
    job_id: str,
    page_image_paths: list[str],
    engine: str | None = None,
) -> dict:
    """
    Process a multi-page PDF OCR job.

    This task processes each page of a PDF and combines results with page markers.

    Args:
        job_id: The job identifier.
        page_image_paths: List of paths to page images (in order).
        engine: OCR engine to use (e.g., 'qari', 'got'). Defaults to config default.

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
        _cleanup_pdf_files(job_id, page_image_paths)
        return {"error": "Job not found"}

    try:
        total_pages = len(page_image_paths)
        logger.info(f"Processing PDF job {job_id} with {total_pages} pages")

        # Process each page
        page_texts = []
        formatted_texts = []
        all_metadata = {"pages": [], "total_pages": total_pages}
        total_confidence = 0.0

        for page_num, page_path in enumerate(page_image_paths, start=1):
            logger.info(f"Processing page {page_num}/{total_pages} for job {job_id}")

            # Run OCR on this page
            result = _run_async(
                model_manager.extract_text(Path(page_path), engine=engine)
            )

            # Add page marker and text
            page_marker = f"=== Page {page_num} of {total_pages} ==="
            page_texts.append(f"{page_marker}\n{result.text}")

            if result.formatted_text:
                formatted_texts.append(f"{page_marker}\n{result.formatted_text}")

            # Collect metadata
            all_metadata["pages"].append(
                {
                    "page_number": page_num,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                }
            )

            total_confidence += result.confidence

        # Combine all page texts
        combined_text = "\n\n".join(page_texts)
        combined_formatted = "\n\n".join(formatted_texts) if formatted_texts else ""

        # Calculate average confidence
        avg_confidence = total_confidence / total_pages if total_pages > 0 else 0.0

        # Store combined result
        job = job_service.set_result(
            job_id,
            text=combined_text,
            formatted_text=combined_formatted,
            output_file_base64=None,
            confidence=avg_confidence,
            metadata=all_metadata,
        )

        logger.info(f"PDF OCR completed for job {job_id} ({total_pages} pages)")

        # Send webhook if configured
        if job and job.webhook_url:
            _run_async(webhook_service.deliver(job))

        # Clean up all PDF-related files
        _cleanup_pdf_files(job_id, page_image_paths)

        return {
            "job_id": job_id,
            "status": "completed",
            "pages": total_pages,
        }

    except RuntimeError as e:
        # Model not loaded or other runtime error
        error_msg = str(e)
        logger.error(f"PDF OCR failed for job {job_id}: {error_msg}")

        job = job_service.update_status(job_id, JobStatus.FAILED, error=error_msg)

        if job and job.webhook_url:
            _run_async(webhook_service.deliver(job))

        # Clean up files even on failure
        _cleanup_pdf_files(job_id, page_image_paths)

        return {"error": error_msg}

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.exception(f"PDF OCR failed for job {job_id}")

        job = job_service.update_status(job_id, JobStatus.FAILED, error=error_msg)

        if job and job.webhook_url:
            _run_async(webhook_service.deliver(job))

        # Clean up files even on failure
        _cleanup_pdf_files(job_id, page_image_paths)

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

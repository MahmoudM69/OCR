import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from rq import Queue
import redis

from app.config import settings
from app.schemas.job import JobStatusResponse
from app.schemas.ocr import OCRSubmitResponse
from app.services.job_service import get_job_service
from app.services.model_status import ModelStatus, get_model_status_service
from app.worker.tasks import process_ocr_job

router = APIRouter(prefix="/ocr", tags=["OCR"])


def get_queue() -> Queue:
    """Get RQ queue for job submission."""
    redis_client = redis.Redis.from_url(settings.redis_url)
    return Queue(connection=redis_client)


@router.post(
    "",
    response_model=OCRSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_ocr(
    file: UploadFile = File(..., description="Image file to process"),
    engine: str = Form(default=None, description="OCR engine to use (default: from config)"),
    webhook_url: str | None = Form(default=None, description="Webhook URL for completion notification"),
):
    """
    Submit an image for OCR processing.

    The image will be processed asynchronously. Use the returned job_id
    to poll for results or provide a webhook_url for notification.
    """
    # Use default engine if not specified
    model_name = engine or settings.default_model

    # Check model status
    model_status_service = get_model_status_service()
    model_status, message = model_status_service.get_status(model_name)
    progress = model_status_service.get_progress(model_name)

    if model_status == ModelStatus.DOWNLOADING:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' is still downloading ({progress:.0f}%). Please try again later.",
        )
    elif model_status == ModelStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' failed to load: {message}",
        )
    elif model_status == ModelStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' is not available. Check /models for available models.",
        )

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )

    # Create job
    job_service = get_job_service()
    job = job_service.create_job(
        filename=file.filename,
        webhook_url=webhook_url,
    )

    # Save uploaded file
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    file_ext = Path(file.filename).suffix if file.filename else ".png"
    image_path = settings.uploads_dir / f"{job.id}{file_ext}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Queue job with engine parameter
    queue = get_queue()
    queue.enqueue(
        process_ocr_job,
        job.id,
        str(image_path),
        model_name,
        job_timeout=settings.job_timeout,
    )

    return OCRSubmitResponse(
        job_id=job.id,
        message=f"Image submitted for processing with '{model_name}' engine",
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
)
async def get_job_status(job_id: str):
    """
    Get the status and result of an OCR job.

    Poll this endpoint to check if processing is complete
    and retrieve the extracted text.
    """
    job_service = get_job_service()
    job = job_service.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        filename=job.filename,
        result=job.result,
        error=job.error,
    )

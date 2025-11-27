import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from rq import Queue
import redis

from app.config import settings
from app.ocr.manager import CURRENT_MODEL_KEY
from app.ocr.registry import OCREngineRegistry
from app.schemas.job import JobStatusResponse
from app.schemas.ocr import OCRSubmitResponse
from app.services.job_service import get_job_service
from app.services.model_status import ModelStatus, get_model_status_service
from app.services.pdf_service import pdf_to_images, PDFProcessingError
from app.utils.file_validation import FileType, ValidationError, validate_file, validate_filename
from app.worker.tasks import process_ocr_job, process_pdf_ocr_job

router = APIRouter(prefix="/ocr", tags=["OCR"])


def get_current_model_from_redis() -> str | None:
    """Get the current model name from Redis."""
    try:
        client = redis.Redis.from_url(settings.redis_url)
        model = client.get(CURRENT_MODEL_KEY)
        if model:
            return model.decode('utf-8')
    except redis.RedisError:
        pass  # Fall back to default model
    return None


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
    # Use current model from Redis, or fall back to default
    model_name = engine or get_current_model_from_redis() or settings.default_model

    # Verify model is registered
    if not OCREngineRegistry.is_registered(model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' is not registered. Check /models for available models.",
        )

    # Check model status for issues that would prevent processing
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
    # Note: NOT_FOUND is OK - the worker will load the model on demand

    # Validate filename
    try:
        safe_filename = validate_filename(file.filename or "image.png")
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid filename: {e}",
        )

    # Validate file type and content
    try:
        validate_file(file.file, expected_type=FileType.IMAGE, max_size_mb=50)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {e}",
        )

    # Create job
    job_service = get_job_service()
    job = job_service.create_job(
        filename=safe_filename,
        webhook_url=webhook_url,
    )

    # Save uploaded file
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    file_ext = Path(safe_filename).suffix or ".png"
    image_path = settings.uploads_dir / f"{job.id}{file_ext}"

    file.file.seek(0)  # Reset after validation
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


@router.post(
    "/pdf",
    response_model=OCRSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_pdf_ocr(
    file: UploadFile = File(..., description="PDF file to process"),
    engine: str = Form(default=None, description="OCR engine to use (default: from config)"),
    webhook_url: str | None = Form(default=None, description="Webhook URL for completion notification"),
    dpi: int = Form(default=300, description="DPI for PDF page rendering (default: 300)"),
):
    """
    Submit a PDF for OCR processing.

    Each page of the PDF will be converted to an image and processed separately.
    The results will include page markers indicating where each page starts and ends.
    """
    # Use current model from Redis, or fall back to default (same as image endpoint)
    model_name = engine or get_current_model_from_redis() or settings.default_model

    # Verify model is registered
    if not OCREngineRegistry.is_registered(model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' is not registered. Check /models for available models.",
        )

    # Check model status for issues that would prevent processing
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
    # Note: NOT_FOUND is OK - the worker will load the model on demand

    # Validate DPI range
    if not (72 <= dpi <= 600):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="DPI must be between 72 and 600",
        )

    # Validate filename
    try:
        safe_filename = validate_filename(file.filename or "document.pdf")
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid filename: {e}",
        )

    # Validate file type and content
    try:
        validate_file(file.file, expected_type=FileType.PDF, max_size_mb=100)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid PDF file: {e}",
        )

    # Save uploaded PDF to temp location first (before creating job)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary ID for initial save
    temp_id = str(uuid.uuid4())
    temp_pdf_path = settings.uploads_dir / f"{temp_id}_temp.pdf"

    file.file.seek(0)  # Reset after validation
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert PDF to images (before creating job to avoid orphaned jobs)
    try:
        with open(temp_pdf_path, "rb") as pdf_file:
            pages = pdf_to_images(
                pdf_file,
                output_dir=settings.uploads_dir,
                job_id=temp_id,
                dpi=dpi,
            )
    except PDFProcessingError as e:
        # Clean up temp PDF file
        temp_pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process PDF: {e}",
        )

    # Now create job and rename files (with cleanup on failure)
    try:
        job_service = get_job_service()
        job = job_service.create_job(
            filename=safe_filename,
            webhook_url=webhook_url,
        )

        # Rename temp PDF to use actual job ID
        pdf_path = settings.uploads_dir / f"{job.id}.pdf"
        temp_pdf_path.rename(pdf_path)

        # Rename page images to use actual job ID
        renamed_pages = []
        for page in pages:
            old_path = page.image_path
            new_filename = old_path.name.replace(temp_id, job.id)
            new_path = old_path.parent / new_filename
            old_path.rename(new_path)
            page.image_path = new_path
            renamed_pages.append(page)

    except Exception as e:
        # Clean up temp files on failure
        temp_pdf_path.unlink(missing_ok=True)
        for page in pages:
            page.image_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {e}",
        )

    # Queue job for multi-page processing
    page_paths = [str(page.image_path) for page in renamed_pages]
    queue = get_queue()
    queue.enqueue(
        process_pdf_ocr_job,
        job.id,
        page_paths,
        model_name,
        job_timeout=settings.job_timeout * len(pages),  # Scale timeout by page count
    )

    return OCRSubmitResponse(
        job_id=job.id,
        message=f"PDF with {len(pages)} pages submitted for processing with '{model_name}' engine",
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

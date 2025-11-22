import redis
from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.ocr.manager import CURRENT_MODEL_KEY
from app.ocr.registry import OCREngineRegistry
from app.schemas.model import (
    CurrentModelResponse,
    ModelInfo,
    ModelsListResponse,
    SwitchModelRequest,
    SwitchModelResponse,
)
from app.services.model_status import get_model_status_service

router = APIRouter(prefix="/models", tags=["Models"])


def get_redis_client() -> redis.Redis:
    """Get Redis client."""
    return redis.Redis.from_url(settings.redis_url)


def get_current_model_from_redis() -> str | None:
    """Get the current model name from Redis."""
    client = get_redis_client()
    model = client.get(CURRENT_MODEL_KEY)
    if model:
        return model.decode('utf-8')
    return None


@router.get("", response_model=ModelsListResponse)
async def list_models():
    """
    List all available OCR models and their status.

    Returns registered models with download/ready status.
    """
    registered_models = OCREngineRegistry.list_registered()
    current = get_current_model_from_redis()
    model_status_service = get_model_status_service()

    models = []
    for name in registered_models:
        model_status, message = model_status_service.get_status(name)
        progress = model_status_service.get_progress(name)

        models.append(
            ModelInfo(
                name=name,
                is_loaded=(name == current),
                is_available=(model_status.value == "ready"),
                status=model_status.value,
                progress=progress,
                message=message,
            )
        )

    return ModelsListResponse(
        models=models,
        current_model=current,
        default_model=settings.default_model,
    )


@router.get("/current", response_model=CurrentModelResponse)
async def get_current_model():
    """
    Get the currently loaded model.

    Returns the name of the model currently set as active,
    or null if no model is set.
    """
    current = get_current_model_from_redis()
    return CurrentModelResponse(
        model=current,
        is_processing=False,  # API can't know if worker is processing
    )


@router.get("/{model_name}/status")
async def get_model_status(model_name: str):
    """
    Get the status of a specific model.

    Returns download status, progress, and any error messages.
    """
    if not OCREngineRegistry.is_registered(model_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )

    model_status_service = get_model_status_service()
    model_status, message = model_status_service.get_status(model_name)
    progress = model_status_service.get_progress(model_name)

    return {
        "model": model_name,
        "status": model_status.value,
        "progress": progress,
        "message": message,
        "is_ready": model_status.value == "ready",
    }


@router.post(
    "/switch",
    response_model=SwitchModelResponse,
    status_code=status.HTTP_200_OK,
)
async def switch_model(request: SwitchModelRequest):
    """
    Switch to a different OCR model.

    This sets the target model in Redis. The worker will automatically
    load/switch to this model when processing the next job.

    Note: The actual model loading happens in the worker process,
    so there may be a brief delay on the first job after switching.
    """
    # Validate model is registered
    if not OCREngineRegistry.is_registered(request.model):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {request.model}. Use GET /models to list available models.",
        )

    # Get previous model and set new one in Redis
    client = get_redis_client()
    previous = get_current_model_from_redis()
    client.set(CURRENT_MODEL_KEY, request.model)

    return SwitchModelResponse(
        previous_model=previous,
        current_model=request.model,
        message=f"Model set to {request.model}. Worker will load on next job.",
    )

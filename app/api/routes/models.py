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
    List all available OCR models.

    Returns registered models and indicates which ones are
    available (have model files in the models directory).
    """
    registered_models = OCREngineRegistry.list_registered()
    current = get_current_model_from_redis()

    # Check which models have files available
    available_models = []
    for name in registered_models:
        model_path = settings.models_dir / name
        if model_path.exists():
            available_models.append(name)

    models = []
    for name in registered_models:
        models.append(
            ModelInfo(
                name=name,
                is_loaded=(name == current),
                is_available=(name in available_models),
            )
        )

    return ModelsListResponse(
        models=models,
        current_model=current,
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

    # Validate model files exist
    model_path = settings.models_dir / request.model
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model files not found for: {request.model}",
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

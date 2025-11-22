import logging
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI

# Import OCR engines to register them
import app.ocr  # noqa: F401

from app.api.routes import health, models, ocr
from app.config import settings
from app.ocr.manager import CURRENT_MODEL_KEY
from app.ocr.registry import OCREngineRegistry

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting OCR API")
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {settings.models_dir}")
    logger.info(f"Uploads directory: {settings.uploads_dir}")

    # Set default model if none is set
    try:
        redis_client = redis.Redis.from_url(settings.redis_url)
        current_model = redis_client.get(CURRENT_MODEL_KEY)

        if current_model is None and settings.default_model:
            # Verify default model is registered
            if OCREngineRegistry.is_registered(settings.default_model):
                redis_client.set(CURRENT_MODEL_KEY, settings.default_model)
                logger.info(f"Set default model: {settings.default_model}")
            else:
                logger.warning(f"Default model '{settings.default_model}' is not registered")
        else:
            logger.info(f"Current model: {current_model.decode() if current_model else 'none'}")

        # Log available engines
        registered = OCREngineRegistry.list_registered()
        logger.info(f"Registered OCR engines: {registered}")

    except Exception as e:
        logger.error(f"Failed to set default model: {e}")

    yield

    # Shutdown
    logger.info("Shutting down OCR API")


app = FastAPI(
    title="OCR API",
    description="Asynchronous OCR processing API with runtime model switching",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(health.router)
app.include_router(ocr.router)
app.include_router(models.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "OCR API",
        "version": "1.0.0",
        "docs": "/docs",
    }

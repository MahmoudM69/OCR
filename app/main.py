import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Import OCR engines to register them
import app.ocr  # noqa: F401

from app.api.routes import health, models, ocr
from app.config import settings

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

# OCR API

Asynchronous OCR processing API with runtime model switching.

## Features

- **Async Processing**: Submit images and poll for results or receive webhooks
- **Multiple Models**: Support for different OCR engines via plugin interface
- **Runtime Switching**: Switch OCR models without restarting the service
- **Docker Ready**: Complete docker-compose setup with Redis

## Quick Start

```bash
# Create model directory for mock engine
mkdir -p data/models/mock

# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Submit an image for OCR
curl -X POST http://localhost:8000/ocr \
  -F "file=@image.png"

# Check job status
curl http://localhost:8000/ocr/{job_id}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ocr` | POST | Submit image for OCR processing |
| `/ocr/{job_id}` | GET | Get job status and results |
| `/models` | GET | List available models |
| `/models/current` | GET | Get currently loaded model |
| `/models/switch` | POST | Switch to a different model |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   FastAPI   │────▶│    Redis    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Worker    │
                                        │ (RQ + OCR)  │
                                        └─────────────┘
```

## Adding New OCR Engines

1. Create a new file in `app/ocr/` (e.g., `tesseract.py`)
2. Implement the `BaseOCREngine` interface:

```python
from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry

@OCREngineRegistry.register
class TesseractEngine(BaseOCREngine):
    @property
    def name(self) -> str:
        return "tesseract"

    async def load(self) -> None:
        # Load model from self.model_path
        self._loaded = True

    async def unload(self) -> None:
        # Clean up resources
        self._loaded = False

    async def extract_text(self, image_path: Path) -> OCRResult:
        self._ensure_loaded()
        # Perform OCR
        return OCRResult(text="...", confidence=0.95)
```

3. Import it in `app/ocr/__init__.py`
4. Place model files in `data/models/{engine_name}/`

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | redis | Redis hostname |
| `REDIS_PORT` | 6379 | Redis port |
| `MODELS_DIR` | /app/data/models | Models directory |
| `UPLOADS_DIR` | /app/data/uploads | Uploads directory |
| `JOB_TIMEOUT` | 300 | Job timeout in seconds |
| `WEBHOOK_TIMEOUT` | 30 | Webhook timeout in seconds |

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run API locally
uvicorn app.main:app --reload

# Run worker locally
rq worker --url redis://localhost:6379/0
```

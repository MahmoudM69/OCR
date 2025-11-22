# OCR API

Asynchronous OCR processing API with runtime model switching and GPU support.

## Features

- **Multiple OCR Engines**: GOT-OCR, QARI (Arabic), DeepSeek-OCR
- **Async Processing**: Submit images and poll for results or receive webhooks
- **Runtime Switching**: Switch OCR models without restarting the service
- **GPU Accelerated**: CUDA support with quantized models (4-bit/8-bit)
- **Docker Ready**: Complete docker-compose setup with Redis and HuggingFace cache

## Available Models

| Model | ID | Best For | Quantization |
|-------|-----|----------|--------------|
| **QARI** | `qari` | Arabic text (default) | 8-bit |
| **GOT-OCR 2.0** | `got` | General printed text | 4-bit |
| **DeepSeek** | `deepseek` | Multilingual documents | 4-bit |

## Quick Start

### 1. Build the base image (one-time, caches dependencies)
```bash
docker build -f Dockerfile.base -t ocr-base:latest .
```

### 2. Start all services
```bash
docker-compose up -d
```

### 3. Check health
```bash
curl http://localhost:8000/health
```

### 4. Submit an image for OCR
```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@image.png"
```

### 5. Check job status
```bash
curl http://localhost:8000/ocr/{job_id}
```

### 6. Switch models
```bash
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "got"}'
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
                                        │ (GPU + OCR) │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  HF Cache   │
                                        │  (Models)   │
                                        └─────────────┘
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | redis | Redis hostname |
| `REDIS_PORT` | 6379 | Redis port |
| `MODELS_DIR` | /app/data/models | Models directory |
| `UPLOADS_DIR` | /app/data/uploads | Uploads directory |
| `DEFAULT_MODEL` | qari | Default OCR model |
| `JOB_TIMEOUT` | 300 | Job timeout in seconds |

## GPU Requirements

- NVIDIA GPU with CUDA 12.4+ support
- Minimum 8GB VRAM recommended
- Docker with NVIDIA Container Toolkit

## Development

```bash
# Build base image
docker build -f Dockerfile.base -t ocr-base:latest .

# Build and start services
docker-compose up --build

# View logs
docker-compose logs -f worker

# Stop services
docker-compose down
```

FROM ocr-base:latest

WORKDIR /app

# Install engine-specific dependencies
# Note: --no-cache-dir removed to enable pip caching with volumes
RUN pip install \
    tiktoken==0.6.0 \
    verovio==4.3.1 \
    qwen-vl-utils

# Install HuggingFace model downloading dependencies
RUN pip install \
    huggingface_hub>=0.20.0 \
    hf-transfer>=0.1.4

# Install image processing and DeepSeek dependencies
RUN pip install \
    opencv-python-headless \
    addict \
    easydict \
    einops \
    matplotlib

# Copy application code
COPY app/ ./app/

# Create data directories
RUN mkdir -p /app/data/models /app/data/uploads

# Expose port
EXPOSE 8000

# Default command (API server)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

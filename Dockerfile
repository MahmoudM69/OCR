FROM ocr-base:latest

WORKDIR /app

# Install engine-specific dependencies
RUN pip install --no-cache-dir \
    tiktoken==0.6.0 \
    verovio==4.3.1 \
    qwen-vl-utils

# Copy application code
COPY app/ ./app/

# Create data directories
RUN mkdir -p /app/data/models /app/data/uploads

# Expose port
EXPOSE 8000

# Default command (API server)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

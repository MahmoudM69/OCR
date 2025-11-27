FROM ocr-base:latest

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install all dependencies from requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create data directories
RUN mkdir -p /app/data/models /app/data/uploads

# Expose port
EXPOSE 8000

# Default command (API server)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

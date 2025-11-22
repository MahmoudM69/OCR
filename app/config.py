from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Redis Settings
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0

    # Paths (mounted volumes)
    models_dir: Path = Path("/app/data/models")
    uploads_dir: Path = Path("/app/data/uploads")

    # Job Settings
    job_timeout: int = 300  # 5 minutes
    job_result_ttl: int = 3600  # 1 hour

    # Webhook Settings
    webhook_timeout: int = 30
    webhook_max_retries: int = 3

    # Default OCR model
    default_model: str = "qari"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

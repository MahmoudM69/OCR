from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class SplittingConfig(BaseModel):
    """Configuration for image splitting."""

    max_megapixels: float = Field(default=2.0, gt=0, description="Maximum megapixels before splitting is triggered")
    max_dimension: int = Field(default=2048, gt=0, description="Maximum dimension (width or height) before splitting")
    overlap_percent: float = Field(default=0.4, ge=0.0, le=1.0, description="Overlap percentage for grid fallback")
    min_gap_pixels: int = Field(default=10, ge=1, description="Minimum whitespace gap for content-aware splitting")
    gap_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Threshold for detecting whitespace")
    min_chunk_size: int = Field(default=256, gt=0, description="Minimum chunk dimension to avoid tiny chunks")
    target_chunk_size: int = Field(default=1024, gt=0, description="Target chunk size when splitting")


class PreprocessingConfig(BaseModel):
    """Configuration for image preprocessing."""

    enabled: bool = Field(default=True, description="Whether preprocessing is enabled")
    target_dpi: int = Field(default=300, ge=72, le=1200, description="Target DPI for scaling")
    denoise_strength: int = Field(default=10, ge=0, le=20, description="Strength of denoising (0-20)")
    binarization_method: Literal["otsu", "adaptive", "none"] = Field(
        default="adaptive", description="Binarization method"
    )
    auto_deskew: bool = Field(default=True, description="Whether to automatically deskew images")
    auto_invert: bool = Field(default=True, description="Whether to automatically invert dark backgrounds")

    # Thresholds for smart selection
    blur_threshold: float = Field(default=100.0, gt=0, description="Below this value, image is considered blurry")
    noise_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Above this value, denoising is applied")
    skew_threshold: float = Field(default=1.0, ge=0.0, le=45.0, description="Above this angle (degrees), deskewing is applied")
    contrast_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Below this value, contrast enhancement is applied")


class EngineConfig(BaseModel):
    """Per-engine configuration."""

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    """Preprocessing settings for this engine."""

    splitting: Optional[SplittingConfig] = None
    """Splitting settings (None = use global)."""


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
    job_timeout: int = 1200  # 20 minutes (for model downloads)
    job_result_ttl: int = 3600  # 1 hour

    # Webhook Settings
    webhook_timeout: int = 30
    webhook_max_retries: int = 3

    # OCR Model Settings
    default_model: str = "qari"
    preload_models: list[str] = ["qari"]  # Models to pre-download at startup

    # Global splitting settings
    splitting: SplittingConfig = SplittingConfig()

    # Per-engine configurations
    engine_configs: dict[str, EngineConfig] = {
        "qari": EngineConfig(
            preprocessing=PreprocessingConfig(
                binarization_method="none",  # Arabic text often better without binarization
                auto_deskew=True,
            ),
        ),
        "got": EngineConfig(
            preprocessing=PreprocessingConfig(
                binarization_method="adaptive",
                auto_deskew=True,
            ),
        ),
        "deepseek": EngineConfig(
            preprocessing=PreprocessingConfig(
                binarization_method="adaptive",
                auto_deskew=True,
            ),
        ),
    }

    def get_engine_config(self, engine_name: str) -> EngineConfig:
        """Get configuration for a specific engine."""
        # DEBUG: Log config access
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[Config] Getting config for engine: {engine_name}")
        
        config = self.engine_configs.get(engine_name, EngineConfig())
        if engine_name not in self.engine_configs:
            logger.warning(f"[Config] Engine '{engine_name}' not found, using default config")
        
        return config

    def get_splitting_config(self, engine_name: str) -> SplittingConfig:
        """Get splitting config for engine (engine-specific or global)."""
        # DEBUG: Log splitting config access
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[Config] Getting splitting config for engine: {engine_name}")
        
        engine_config = self.get_engine_config(engine_name)
        splitting_config = engine_config.splitting or self.splitting
        
        # DEBUG: Log which config is being used
        if engine_config.splitting:
            logger.debug(f"[Config] Using engine-specific splitting config for {engine_name}")
        else:
            logger.debug(f"[Config] Using global splitting config for {engine_name}")
            
        return splitting_config

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

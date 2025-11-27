"""
Image preprocessing module for OCR.

Provides a modular, extensible preprocessing pipeline with
smart step selection based on image quality analysis.
"""

from .base import PreprocessingStep, ImageAnalysis, PreprocessingConfig
from .analyzer import ImageQualityAnalyzer
from .pipeline import PreprocessingPipeline, create_pipeline

__all__ = [
    "PreprocessingStep",
    "ImageAnalysis",
    "PreprocessingConfig",
    "ImageQualityAnalyzer",
    "PreprocessingPipeline",
    "create_pipeline",
]

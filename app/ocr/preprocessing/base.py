"""
Base classes for image preprocessing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ImageAnalysis:
    """Analysis results for preprocessing decisions."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    is_grayscale: bool
    """Whether the image is already grayscale."""

    blur_score: float
    """Blur score (higher = sharper). Laplacian variance."""

    noise_level: float
    """Estimated noise level (0.0 to 1.0)."""

    skew_angle: float
    """Detected skew angle in degrees."""

    contrast_ratio: float
    """Contrast ratio (0.0 to 1.0)."""

    brightness: float
    """Average brightness (0.0 to 1.0)."""

    estimated_dpi: int
    """Estimated DPI based on common document sizes."""

    has_text: bool
    """Whether text-like content was detected."""

    is_inverted: bool
    """Whether the image appears to have inverted colors."""

    needs_denoising: bool = False
    """Whether denoising is recommended."""

    needs_deskewing: bool = False
    """Whether deskewing is recommended."""

    needs_contrast_enhancement: bool = False
    """Whether contrast enhancement is recommended."""


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    enabled: bool = True
    """Whether preprocessing is enabled."""

    target_dpi: int = 300
    """Target DPI for scaling."""

    denoise_strength: int = 10
    """Strength of denoising (0-20)."""

    binarization_method: str = "adaptive"
    """Binarization method: 'otsu', 'adaptive', or 'none'."""

    auto_deskew: bool = True
    """Whether to automatically deskew images."""

    auto_invert: bool = True
    """Whether to automatically invert dark backgrounds."""

    # Thresholds for smart selection
    blur_threshold: float = 100.0
    """Below this value, image is considered blurry."""

    noise_threshold: float = 0.1
    """Above this value, denoising is applied."""

    skew_threshold: float = 1.0
    """Above this angle (degrees), deskewing is applied."""

    contrast_threshold: float = 0.3
    """Below this value, contrast enhancement is applied."""


@dataclass
class StepResult:
    """Result of applying a preprocessing step."""

    image: np.ndarray
    """Processed image."""

    applied: bool
    """Whether the step was actually applied."""

    step_name: str
    """Name of the step."""

    metadata: dict = field(default_factory=dict)
    """Additional information about the processing."""


class PreprocessingStep(ABC):
    """
    Abstract base class for preprocessing steps.

    Each step decides whether to apply itself based on image analysis.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the step with configuration."""
        self.config = config or PreprocessingConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this preprocessing step."""
        pass

    @abstractmethod
    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """
        Determine if this step should be applied.

        Args:
            image: Current image state.
            analysis: Image analysis results.

        Returns:
            True if this step should be applied.
        """
        pass

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessing step.

        Args:
            image: Input image as numpy array.

        Returns:
            Processed image.
        """
        pass

    def process(
        self,
        image: np.ndarray,
        analysis: ImageAnalysis,
        force: bool = False,
    ) -> StepResult:
        """
        Process the image, applying step if needed.

        Args:
            image: Input image.
            analysis: Image analysis results.
            force: Force application regardless of analysis.

        Returns:
            StepResult with processed image and metadata.
        """
        should_apply = force or self.should_apply(image, analysis)

        if should_apply:
            processed = self.apply(image)
            return StepResult(
                image=processed,
                applied=True,
                step_name=self.name,
                metadata={"forced": force},
            )

        return StepResult(
            image=image,
            applied=False,
            step_name=self.name,
            metadata={"reason": "not_needed"},
        )

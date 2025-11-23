"""
Image scaling/DPI normalization step.
"""

from typing import Optional
import numpy as np
import cv2

from ..base import PreprocessingStep, PreprocessingConfig, ImageAnalysis


class ScalingStep(PreprocessingStep):
    """
    Scales images to target DPI for optimal OCR performance.

    Most OCR engines perform best at 300 DPI. This step
    upscales low-resolution images to improve recognition.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize with configuration."""
        super().__init__(config)
        self.target_dpi = self.config.target_dpi
        self._scale_factor: float = 1.0

    @property
    def name(self) -> str:
        return "scaling"

    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """Apply if estimated DPI is below target."""
        if analysis.estimated_dpi >= self.target_dpi:
            return False

        # Calculate scale factor
        self._scale_factor = self.target_dpi / analysis.estimated_dpi

        # Don't scale if factor is too small to matter
        if self._scale_factor < 1.1:
            return False

        # Don't upscale too much (diminishing returns)
        if self._scale_factor > 3.0:
            self._scale_factor = 3.0

        return True

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Scale image to target DPI."""
        if self._scale_factor <= 1.0:
            return image

        height, width = image.shape[:2]
        new_width = int(width * self._scale_factor)
        new_height = int(height * self._scale_factor)

        # Use INTER_CUBIC for upscaling (better quality)
        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC,
        )

    def scale_to_dpi(
        self,
        image: np.ndarray,
        current_dpi: int,
        target_dpi: Optional[int] = None,
    ) -> np.ndarray:
        """
        Scale image to specific DPI.

        Args:
            image: Input image.
            current_dpi: Current DPI of the image.
            target_dpi: Target DPI (defaults to config target).

        Returns:
            Scaled image.
        """
        target = target_dpi or self.target_dpi

        if current_dpi >= target:
            return image

        scale_factor = target / current_dpi

        if scale_factor > 3.0:
            scale_factor = 3.0

        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC,
        )

    def scale_by_factor(
        self,
        image: np.ndarray,
        factor: float,
    ) -> np.ndarray:
        """
        Scale image by a specific factor.

        Args:
            image: Input image.
            factor: Scale factor (>1 = upscale, <1 = downscale).

        Returns:
            Scaled image.
        """
        if factor == 1.0:
            return image

        height, width = image.shape[:2]
        new_width = int(width * factor)
        new_height = int(height * factor)

        interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )

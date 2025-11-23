"""
Noise removal step.
"""

from typing import Optional
import numpy as np
import cv2

from ..base import PreprocessingStep, PreprocessingConfig, ImageAnalysis


class NoiseRemovalStep(PreprocessingStep):
    """
    Removes noise from images using non-local means denoising.

    Uses OpenCV's fastNlMeansDenoising for grayscale images
    or fastNlMeansDenoisingColored for color images.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize with configuration."""
        super().__init__(config)
        self.strength = self.config.denoise_strength

    @property
    def name(self) -> str:
        return "noise_removal"

    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """Apply if noise level exceeds threshold."""
        return analysis.needs_denoising

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising."""
        if len(image.shape) == 2:
            # Grayscale
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=self.strength,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        else:
            # Color
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=self.strength,
                hForColorComponents=self.strength,
                templateWindowSize=7,
                searchWindowSize=21,
            )

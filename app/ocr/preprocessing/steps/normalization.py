"""
Pixel intensity normalization step.
"""

import numpy as np
import cv2

from ..base import PreprocessingStep, ImageAnalysis


class NormalizationStep(PreprocessingStep):
    """
    Normalizes pixel intensity values.

    Stretches the histogram to use the full 0-255 range,
    improving contrast for images with limited dynamic range.
    """

    @property
    def name(self) -> str:
        return "normalization"

    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """Apply if contrast is low or image doesn't use full range."""
        if analysis.needs_contrast_enhancement:
            return True

        # Also apply if image uses less than 80% of dynamic range
        if len(image.shape) == 2:
            min_val, max_val = image.min(), image.max()
            range_used = (max_val - min_val) / 255.0
            return range_used < 0.8

        return False

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel intensities to 0-255 range."""
        # Convert to float for normalization
        normalized = np.zeros(image.shape, dtype=np.float32)

        # Apply MINMAX normalization
        cv2.normalize(
            image.astype(np.float32),
            normalized,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        return normalized.astype(np.uint8)

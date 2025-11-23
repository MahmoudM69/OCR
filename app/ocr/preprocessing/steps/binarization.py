"""
Binarization/thresholding step.
"""

from typing import Optional
import numpy as np
import cv2

from ..base import PreprocessingStep, PreprocessingConfig, ImageAnalysis


class BinarizationStep(PreprocessingStep):
    """
    Converts image to binary (black and white) using thresholding.

    Supports multiple methods:
    - 'otsu': Otsu's automatic thresholding
    - 'adaptive': Adaptive gaussian thresholding
    - 'none': Skip binarization
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize with configuration."""
        super().__init__(config)
        self.method = self.config.binarization_method

    @property
    def name(self) -> str:
        return "binarization"

    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """Apply based on configuration method."""
        if self.method == "none":
            return False

        # Don't binarize if image is already binary
        if len(image.shape) == 2:
            unique_values = len(np.unique(image))
            if unique_values <= 2:
                return False

        return True

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply binarization."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        if self.method == "otsu":
            return self._apply_otsu(gray)
        elif self.method == "adaptive":
            return self._apply_adaptive(gray)
        else:
            return gray

    def _apply_otsu(self, gray: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding."""
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary

    def _apply_adaptive(self, gray: np.ndarray) -> np.ndarray:
        """Apply adaptive gaussian thresholding."""
        # Calculate block size based on image size
        height, width = gray.shape
        block_size = max(11, min(101, min(width, height) // 20))

        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            11,  # C constant
        )

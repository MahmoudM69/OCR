"""
Grayscale conversion step.
"""

import numpy as np
import cv2

from ..base import PreprocessingStep, ImageAnalysis


class GrayscaleStep(PreprocessingStep):
    """
    Converts color images to grayscale.

    This is typically the first step in preprocessing as many
    subsequent operations work on grayscale images.
    """

    @property
    def name(self) -> str:
        return "grayscale"

    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """Apply if image is not already grayscale."""
        return not analysis.is_grayscale

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(image.shape) == 2:
            return image

        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA to grayscale
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:
                # RGB to grayscale
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

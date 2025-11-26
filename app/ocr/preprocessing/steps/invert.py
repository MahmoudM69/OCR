"""
Image inversion step for light-on-dark text.
"""

import numpy as np

from ..base import PreprocessingStep, ImageAnalysis


class InvertStep(PreprocessingStep):
    """
    Inverts image colors when light text on dark background is detected.

    This helps OCR engines that expect dark text on light background
    to process inverted documents (e.g., white text on black).
    """

    @property
    def name(self) -> str:
        return "invert"

    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """Apply if auto_invert is enabled and image is detected as inverted."""
        return self.config.auto_invert and analysis.is_inverted

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Invert pixel values (255 - value for each pixel)."""
        return 255 - image

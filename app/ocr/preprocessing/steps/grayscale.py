"""
Grayscale conversion step.
"""

import numpy as np
import cv2

from ..base import PreprocessingStep, ImageAnalysis


class GrayscaleStep(PreprocessingStep):
    """
    Converts color images to grayscale, or removes alpha channel when preserving color.

    This is typically the first step in preprocessing as many
    subsequent operations work on grayscale images.

    When preserve_color=True (for VLMs), this step converts RGBA to RGB
    instead of grayscale, since models expect 3-channel images.
    """

    @property
    def name(self) -> str:
        return "grayscale"

    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """
        Apply if:
        - preserve_color=True AND image has alpha channel (RGBA) → need to strip alpha
        - preserve_color=False AND image is not grayscale → need to convert to grayscale
        """
        preserve_color = getattr(self.config, 'preserve_color', False)

        if preserve_color:
            # Even with preserve_color, we need to remove alpha channel if present
            # Models expect RGB (3 channels), not RGBA (4 channels)
            has_alpha = len(image.shape) == 3 and image.shape[2] == 4
            return has_alpha
        else:
            return not analysis.is_grayscale

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale or RGB depending on preserve_color setting."""
        if len(image.shape) == 2:
            return image

        preserve_color = getattr(self.config, 'preserve_color', False)

        if len(image.shape) == 3:
            if preserve_color:
                # Convert RGBA to RGB (preserve color, just remove alpha)
                if image.shape[2] == 4:
                    return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                # Already RGB, no change needed
                return image
            else:
                # Convert to grayscale
                if image.shape[2] == 4:
                    return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                elif image.shape[2] == 3:
                    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

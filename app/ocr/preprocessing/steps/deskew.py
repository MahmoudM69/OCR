"""
Deskew/rotation correction step.
"""

from typing import Optional
import numpy as np
import cv2

from ..base import PreprocessingStep, PreprocessingConfig, ImageAnalysis


class DeskewStep(PreprocessingStep):
    """
    Corrects image skew/rotation.

    Uses the skew angle detected during analysis to rotate
    the image back to horizontal alignment.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize with configuration."""
        super().__init__(config)
        self._detected_angle: float = 0.0

    @property
    def name(self) -> str:
        return "deskew"

    def should_apply(self, image: np.ndarray, analysis: ImageAnalysis) -> bool:
        """Apply if skew angle exceeds threshold."""
        if not self.config.auto_deskew:
            return False

        # Store angle for apply method
        self._detected_angle = analysis.skew_angle

        return analysis.needs_deskewing

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Rotate image to correct skew."""
        if abs(self._detected_angle) < 0.1:
            return image

        return self._rotate_image(image, -self._detected_angle)

    def _rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        background_color: int = 255,
    ) -> np.ndarray:
        """
        Rotate image by given angle.

        Args:
            image: Input image.
            angle: Rotation angle in degrees (positive = counter-clockwise).
            background_color: Color to fill exposed areas.

        Returns:
            Rotated image.
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box size
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])

        new_width = int(height * sin_angle + width * cos_angle)
        new_height = int(height * cos_angle + width * sin_angle)

        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2

        # Determine border color
        if len(image.shape) == 3:
            border_color = (background_color, background_color, background_color)
        else:
            border_color = background_color

        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color,
        )

        return rotated

    def deskew_with_angle(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Deskew image with a specific angle.

        Args:
            image: Input image.
            angle: Angle to correct (will be negated for correction).

        Returns:
            Deskewed image.
        """
        return self._rotate_image(image, -angle)

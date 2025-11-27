"""
Image quality analyzer for preprocessing decisions.
"""

from typing import Optional
import numpy as np
import cv2

from .base import ImageAnalysis, PreprocessingConfig


class ImageQualityAnalyzer:
    """
    Analyzes image quality to determine which preprocessing steps are needed.

    Measures:
    - Blur (Laplacian variance)
    - Noise level (local variance estimation)
    - Skew angle (Hough transform)
    - Contrast ratio
    - Brightness
    - DPI estimation
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the analyzer."""
        self.config = config or PreprocessingConfig()

    def analyze(self, image: np.ndarray) -> ImageAnalysis:
        """
        Perform comprehensive image analysis.

        Args:
            image: Image as numpy array (RGB or grayscale).

        Returns:
            ImageAnalysis with all quality metrics.
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            is_grayscale = False
        else:
            gray = image
            is_grayscale = True

        height, width = gray.shape

        # Measure blur
        blur_score = self._measure_blur(gray)

        # Measure noise
        noise_level = self._measure_noise(gray)

        # Detect skew
        skew_angle = self._detect_skew(gray)

        # Measure contrast
        contrast_ratio = self._measure_contrast(gray)

        # Measure brightness
        brightness = self._measure_brightness(gray)

        # Estimate DPI
        estimated_dpi = self._estimate_dpi(width, height)

        # Detect text presence
        has_text = self._detect_text(gray)

        # Check if inverted
        is_inverted = self._is_inverted(gray)

        # Determine recommendations
        needs_denoising = noise_level > self.config.noise_threshold
        needs_deskewing = abs(skew_angle) > self.config.skew_threshold
        needs_contrast_enhancement = contrast_ratio < self.config.contrast_threshold

        return ImageAnalysis(
            width=width,
            height=height,
            is_grayscale=is_grayscale,
            blur_score=blur_score,
            noise_level=noise_level,
            skew_angle=skew_angle,
            contrast_ratio=contrast_ratio,
            brightness=brightness,
            estimated_dpi=estimated_dpi,
            has_text=has_text,
            is_inverted=is_inverted,
            needs_denoising=needs_denoising,
            needs_deskewing=needs_deskewing,
            needs_contrast_enhancement=needs_contrast_enhancement,
        )

    def _measure_blur(self, gray: np.ndarray) -> float:
        """
        Measure image sharpness using Laplacian variance.

        Higher values indicate sharper images.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    def _measure_noise(self, gray: np.ndarray) -> float:
        """
        Estimate noise level using local variance method.

        Returns normalized noise level (0.0 to 1.0).
        """
        # Use small kernel for local variance
        kernel_size = 5
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray.astype(np.float32) ** 2, (kernel_size, kernel_size))
        variance = sqr_mean - mean ** 2

        # Normalize: typical noise variance is 100-1000 for 8-bit images
        noise_estimate = np.median(np.sqrt(np.abs(variance)))
        normalized = min(1.0, noise_estimate / 50.0)

        return float(normalized)

    def _detect_skew(self, gray: np.ndarray) -> float:
        """
        Detect skew angle using Hough transform.

        Returns angle in degrees.
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return 0.0

        # Collect angles
        angles = []
        for line in lines[:50]:  # Limit to first 50 lines
            rho, theta = line[0]
            # Convert to degrees, relative to horizontal
            angle = np.degrees(theta) - 90

            # Only consider near-horizontal or near-vertical lines
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # Use median to be robust to outliers
        skew = float(np.median(angles))

        return skew

    def _measure_contrast(self, gray: np.ndarray) -> float:
        """
        Measure contrast using Michelson contrast formula.

        Returns contrast ratio (0.0 to 1.0).
        """
        min_val = float(gray.min())
        max_val = float(gray.max())

        if max_val + min_val == 0:
            return 0.0

        contrast = (max_val - min_val) / (max_val + min_val)
        return contrast

    def _measure_brightness(self, gray: np.ndarray) -> float:
        """
        Measure average brightness.

        Returns normalized brightness (0.0 to 1.0).
        """
        return float(gray.mean() / 255.0)

    def _estimate_dpi(self, width: int, height: int) -> int:
        """
        Estimate DPI based on common document sizes.

        Assumes standard document sizes (A4, Letter).
        """
        # Common document dimensions at various DPIs
        # A4: 210mm x 297mm
        # Letter: 8.5in x 11in

        # Calculate pixels per inch assuming A4 width (210mm = 8.27in)
        # or Letter width (8.5in)
        a4_width_inches = 8.27
        letter_width_inches = 8.5

        # Use the larger dimension as the likely width
        larger_dim = max(width, height)
        smaller_dim = min(width, height)

        # Estimate based on A4 proportions
        aspect_ratio = larger_dim / smaller_dim if smaller_dim > 0 else 1.0

        # A4 aspect ratio is about 1.414, Letter is about 1.294
        if 1.35 < aspect_ratio < 1.50:
            # Likely A4
            dpi = int(smaller_dim / a4_width_inches)
        elif 1.25 < aspect_ratio < 1.35:
            # Likely Letter
            dpi = int(smaller_dim / letter_width_inches)
        else:
            # Unknown, use conservative estimate
            dpi = int(smaller_dim / 8.0)

        # Clamp to reasonable range
        return max(72, min(600, dpi))

    def _detect_text(self, gray: np.ndarray) -> bool:
        """
        Detect if image likely contains text.

        Uses edge density and pattern analysis.
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size

        # Text typically has edge density between 0.01 and 0.3
        return 0.01 < edge_density < 0.3

    def _is_inverted(self, gray: np.ndarray) -> bool:
        """
        Check if image has inverted colors (light text on dark background).
        """
        # Sample edges vs background
        edges = cv2.Canny(gray, 50, 150)

        # Get intensity at edge locations
        edge_intensity = gray[edges > 0].mean() if np.any(edges > 0) else 128

        # Get background intensity (non-edge areas)
        background_intensity = gray[edges == 0].mean() if np.any(edges == 0) else 128

        # If edges are brighter than background, likely inverted
        return edge_intensity > background_intensity + 30

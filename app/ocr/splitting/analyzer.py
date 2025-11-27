"""
Image analysis for split decisions.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


@dataclass
class ImageAnalysis:
    """Analysis results for split decision making."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    megapixels: float
    """Image size in megapixels."""

    has_horizontal_gaps: bool
    """Whether horizontal whitespace gaps were detected."""

    has_vertical_gaps: bool
    """Whether vertical whitespace gaps were detected."""

    horizontal_gap_positions: list[int]
    """Y positions of horizontal gaps."""

    vertical_gap_positions: list[int]
    """X positions of vertical gaps."""

    estimated_columns: int
    """Estimated number of text columns."""

    estimated_rows: int
    """Estimated number of text rows/sections."""

    is_mostly_white: bool
    """Whether the image is mostly whitespace."""

    content_density: float
    """Ratio of content to total area (0.0 to 1.0)."""


class SplitAnalyzer:
    """Analyzes images to determine optimal splitting strategy."""

    def __init__(
        self,
        gap_threshold: float = 0.95,
        min_gap_pixels: int = 10,
        smoothing_kernel: int = 5,
    ):
        """
        Initialize the analyzer.

        Args:
            gap_threshold: Threshold for detecting whitespace (0.0 to 1.0).
            min_gap_pixels: Minimum gap width to consider.
            smoothing_kernel: Kernel size for smoothing projections.
        """
        self.gap_threshold = gap_threshold
        self.min_gap_pixels = min_gap_pixels
        self.smoothing_kernel = smoothing_kernel

    def analyze(self, image: np.ndarray) -> ImageAnalysis:
        """
        Analyze an image for splitting.

        Args:
            image: Image as numpy array (grayscale or color).

        Returns:
            ImageAnalysis with detection results.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        height, width = gray.shape
        megapixels = (width * height) / 1_000_000

        # Binarize for projection analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate projections
        h_projection = self._calculate_horizontal_projection(binary)
        v_projection = self._calculate_vertical_projection(binary)

        # Find gaps
        h_gaps = self._find_gaps(h_projection, height, is_horizontal=True)
        v_gaps = self._find_gaps(v_projection, width, is_horizontal=False)

        # Estimate structure
        estimated_columns = len(v_gaps) + 1 if v_gaps else 1
        estimated_rows = len(h_gaps) + 1 if h_gaps else 1

        # Calculate content density
        content_pixels = np.sum(binary > 0)
        total_pixels = width * height
        content_density = content_pixels / total_pixels if total_pixels > 0 else 0

        is_mostly_white = content_density < 0.1

        return ImageAnalysis(
            width=width,
            height=height,
            megapixels=megapixels,
            has_horizontal_gaps=len(h_gaps) > 0,
            has_vertical_gaps=len(v_gaps) > 0,
            horizontal_gap_positions=h_gaps,
            vertical_gap_positions=v_gaps,
            estimated_columns=estimated_columns,
            estimated_rows=estimated_rows,
            is_mostly_white=is_mostly_white,
            content_density=content_density,
        )

    def _calculate_horizontal_projection(self, binary: np.ndarray) -> np.ndarray:
        """
        Calculate horizontal projection profile (sum along rows).

        Args:
            binary: Binary image.

        Returns:
            1D array of row sums.
        """
        projection = np.sum(binary, axis=1).astype(np.float32)

        # Normalize to 0-1 range
        max_val = projection.max()
        if max_val > 0:
            projection = projection / max_val

        # Smooth to reduce noise
        if self.smoothing_kernel > 1:
            kernel = np.ones(self.smoothing_kernel) / self.smoothing_kernel
            projection = np.convolve(projection, kernel, mode='same')

        return projection

    def _calculate_vertical_projection(self, binary: np.ndarray) -> np.ndarray:
        """
        Calculate vertical projection profile (sum along columns).

        Args:
            binary: Binary image.

        Returns:
            1D array of column sums.
        """
        projection = np.sum(binary, axis=0).astype(np.float32)

        # Normalize to 0-1 range
        max_val = projection.max()
        if max_val > 0:
            projection = projection / max_val

        # Smooth to reduce noise
        if self.smoothing_kernel > 1:
            kernel = np.ones(self.smoothing_kernel) / self.smoothing_kernel
            projection = np.convolve(projection, kernel, mode='same')

        return projection

    def _find_gaps(
        self,
        projection: np.ndarray,
        dimension: int,
        is_horizontal: bool,
    ) -> list[int]:
        """
        Find gap positions in a projection profile.

        Args:
            projection: Normalized projection profile.
            dimension: Total dimension (height for horizontal, width for vertical).
            is_horizontal: Whether this is horizontal projection.

        Returns:
            List of gap center positions.
        """
        # Find regions below threshold (whitespace)
        threshold = 1 - self.gap_threshold  # Invert: low projection = whitespace
        is_gap = projection < threshold

        # Find gap regions
        gaps = []
        in_gap = False
        gap_start = 0

        for i, is_g in enumerate(is_gap):
            if is_g and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_g and in_gap:
                gap_end = i
                gap_width = gap_end - gap_start

                if gap_width >= self.min_gap_pixels:
                    # Use center of gap
                    gap_center = gap_start + gap_width // 2

                    # Don't split too close to edges
                    edge_margin = dimension * 0.1
                    if edge_margin < gap_center < dimension - edge_margin:
                        gaps.append(gap_center)

                in_gap = False

        return gaps

    def find_optimal_split_points(
        self,
        image: np.ndarray,
        target_chunks: int,
        prefer_horizontal: bool = True,
    ) -> tuple[list[int], list[int]]:
        """
        Find optimal split points for a target number of chunks.

        Args:
            image: Image as numpy array.
            target_chunks: Target number of chunks.
            prefer_horizontal: Prefer horizontal splits (for RTL/LTR text).

        Returns:
            Tuple of (horizontal_splits, vertical_splits).
        """
        analysis = self.analyze(image)

        h_splits = []
        v_splits = []

        # Determine grid needed
        import math
        sqrt_chunks = math.sqrt(target_chunks)

        if prefer_horizontal:
            rows = math.ceil(sqrt_chunks)
            cols = math.ceil(target_chunks / rows)
        else:
            cols = math.ceil(sqrt_chunks)
            rows = math.ceil(target_chunks / cols)

        # Try to use detected gaps first
        if rows > 1 and analysis.has_horizontal_gaps:
            h_splits = self._select_best_gaps(
                analysis.horizontal_gap_positions,
                rows - 1,
                analysis.height,
            )

        if cols > 1 and analysis.has_vertical_gaps:
            v_splits = self._select_best_gaps(
                analysis.vertical_gap_positions,
                cols - 1,
                analysis.width,
            )

        return h_splits, v_splits

    def _select_best_gaps(
        self,
        gaps: list[int],
        num_needed: int,
        dimension: int,
    ) -> list[int]:
        """
        Select best gaps for even distribution.

        Args:
            gaps: Available gap positions.
            num_needed: Number of splits needed.
            dimension: Total dimension.

        Returns:
            Selected gap positions.
        """
        if not gaps or num_needed <= 0:
            return []

        if len(gaps) <= num_needed:
            return sorted(gaps)

        # Select gaps closest to ideal positions
        ideal_spacing = dimension / (num_needed + 1)
        ideal_positions = [ideal_spacing * (i + 1) for i in range(num_needed)]

        selected = []
        remaining_gaps = list(gaps)

        for ideal in ideal_positions:
            if not remaining_gaps:
                break

            # Find closest gap
            closest = min(remaining_gaps, key=lambda g: abs(g - ideal))
            selected.append(closest)
            remaining_gaps.remove(closest)

        return sorted(selected)

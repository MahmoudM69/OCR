"""
Projection profile-based image splitter.

Uses horizontal and vertical projection profiles to find whitespace gaps
between text regions for content-aware splitting.
"""

from typing import Optional
import numpy as np
import cv2

from .base import BaseSplitter, SplitResult, SplitConfig, ImageChunk
from .analyzer import SplitAnalyzer


class ProjectionSplitter(BaseSplitter):
    """
    Splits images using projection profile analysis.

    This splitter analyzes the horizontal and vertical projections of pixel
    intensities to find natural whitespace gaps between text regions.
    It's ideal for documents with clear paragraph or column separations.
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """Initialize the projection splitter."""
        super().__init__(config)
        self.analyzer = SplitAnalyzer(
            gap_threshold=self.config.gap_threshold,
            min_gap_pixels=self.config.min_gap_pixels,
        )

    @property
    def name(self) -> str:
        return "projection"

    def can_split(self, image: np.ndarray) -> bool:
        """
        Check if projection-based splitting can find valid split points.

        Args:
            image: Image as numpy array.

        Returns:
            True if whitespace gaps were detected.
        """
        if not self.needs_splitting(image):
            return False

        analysis = self.analyzer.analyze(image)
        return analysis.has_horizontal_gaps or analysis.has_vertical_gaps

    def split(self, image: np.ndarray) -> SplitResult:
        """
        Split image using projection profile analysis.

        Args:
            image: Image as numpy array.

        Returns:
            SplitResult with chunks split at whitespace gaps.
        """
        if not self.needs_splitting(image):
            return self._create_single_chunk_result(image)

        analysis = self.analyzer.analyze(image)
        height, width = image.shape[:2]

        # Calculate how many chunks we need
        target_chunks = self._calculate_target_chunks(width, height)

        # Find split points
        h_splits, v_splits = self._find_split_points(
            image, analysis, target_chunks
        )

        # If no splits found, return single chunk
        if not h_splits and not v_splits:
            return self._create_single_chunk_result(image)

        # Create chunks
        chunks = self._create_chunks(image, h_splits, v_splits)

        rows = len(h_splits) + 1
        cols = len(v_splits) + 1

        return SplitResult(
            chunks=chunks,
            grid_shape=(rows, cols),
            original_size=(width, height),
            split_method=self.name,
            was_split=True,
            metadata={
                "horizontal_splits": h_splits,
                "vertical_splits": v_splits,
                "analysis": {
                    "has_horizontal_gaps": analysis.has_horizontal_gaps,
                    "has_vertical_gaps": analysis.has_vertical_gaps,
                    "content_density": analysis.content_density,
                },
            },
        )

    def _calculate_target_chunks(self, width: int, height: int) -> int:
        """Calculate target number of chunks based on image size."""
        megapixels = (width * height) / 1_000_000
        target_mp = self.config.max_megapixels

        # Calculate chunks needed to get below threshold
        chunks_needed = int(np.ceil(megapixels / target_mp))

        # Also consider max dimension
        dim_chunks_w = int(np.ceil(width / self.config.max_dimension))
        dim_chunks_h = int(np.ceil(height / self.config.max_dimension))
        dim_chunks = dim_chunks_w * dim_chunks_h

        return max(chunks_needed, dim_chunks, 2)

    def _find_split_points(
        self,
        image: np.ndarray,
        analysis,
        target_chunks: int,
    ) -> tuple[list[int], list[int]]:
        """
        Find optimal split points using detected gaps.

        Args:
            image: Image as numpy array.
            analysis: ImageAnalysis results.
            target_chunks: Target number of chunks.

        Returns:
            Tuple of (horizontal_splits, vertical_splits).
        """
        height, width = image.shape[:2]

        # Calculate grid dimensions
        import math
        sqrt_chunks = math.sqrt(target_chunks)

        # Prefer horizontal splits for text documents (preserves reading order)
        rows = int(math.ceil(sqrt_chunks))
        cols = int(math.ceil(target_chunks / rows))

        h_splits = []
        v_splits = []

        # Use detected gaps if available
        if rows > 1 and analysis.has_horizontal_gaps:
            h_splits = self._select_splits_from_gaps(
                analysis.horizontal_gap_positions,
                rows - 1,
                height,
            )

        if cols > 1 and analysis.has_vertical_gaps:
            v_splits = self._select_splits_from_gaps(
                analysis.vertical_gap_positions,
                cols - 1,
                width,
            )

        # Ensure minimum chunk size
        h_splits = self._filter_by_min_size(h_splits, height)
        v_splits = self._filter_by_min_size(v_splits, width)

        return h_splits, v_splits

    def _select_splits_from_gaps(
        self,
        gaps: list[int],
        num_splits: int,
        dimension: int,
    ) -> list[int]:
        """
        Select best split positions from detected gaps.

        Args:
            gaps: Detected gap positions.
            num_splits: Number of splits needed.
            dimension: Total dimension.

        Returns:
            Selected split positions.
        """
        if not gaps or num_splits <= 0:
            return []

        if len(gaps) <= num_splits:
            return sorted(gaps)

        # Calculate ideal positions for even distribution
        ideal_spacing = dimension / (num_splits + 1)
        ideal_positions = [ideal_spacing * (i + 1) for i in range(num_splits)]

        # Greedily select gaps closest to ideal positions
        selected = []
        available = list(gaps)

        for ideal in ideal_positions:
            if not available:
                break

            closest = min(available, key=lambda g: abs(g - ideal))
            selected.append(closest)
            available.remove(closest)

        return sorted(selected)

    def _filter_by_min_size(
        self,
        splits: list[int],
        dimension: int,
    ) -> list[int]:
        """
        Filter splits to ensure minimum chunk size.

        Args:
            splits: Split positions.
            dimension: Total dimension.

        Returns:
            Filtered split positions.
        """
        if not splits:
            return []

        min_size = self.config.min_chunk_size
        filtered = []
        prev_pos = 0

        for split in sorted(splits):
            # Check if chunk before this split is big enough
            if split - prev_pos >= min_size:
                # Check if chunk after this split would be big enough
                remaining = dimension - split
                if remaining >= min_size:
                    filtered.append(split)
                    prev_pos = split

        return filtered

    def _create_chunks(
        self,
        image: np.ndarray,
        h_splits: list[int],
        v_splits: list[int],
    ) -> list[ImageChunk]:
        """
        Create image chunks from split positions.

        Args:
            image: Original image.
            h_splits: Horizontal split positions.
            v_splits: Vertical split positions.

        Returns:
            List of ImageChunk objects.
        """
        height, width = image.shape[:2]

        # Create boundary arrays including edges
        y_boundaries = [0] + sorted(h_splits) + [height]
        x_boundaries = [0] + sorted(v_splits) + [width]

        chunks = []
        index = 0

        for row, (y_start, y_end) in enumerate(zip(y_boundaries[:-1], y_boundaries[1:])):
            for col, (x_start, x_end) in enumerate(zip(x_boundaries[:-1], x_boundaries[1:])):
                # Extract chunk
                chunk_image = image[y_start:y_end, x_start:x_end].copy()

                chunk = ImageChunk(
                    image=chunk_image,
                    index=index,
                    row=row,
                    col=col,
                    x_offset=x_start,
                    y_offset=y_start,
                    width=x_end - x_start,
                    height=y_end - y_start,
                )

                chunks.append(chunk)
                index += 1

        return chunks

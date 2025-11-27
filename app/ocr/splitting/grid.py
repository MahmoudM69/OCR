"""
Grid-based image splitter with configurable overlap.

This is the fallback splitter when content-aware methods can't find
natural split points. Uses significant overlap to allow deduplication
of text at boundaries.
"""

from typing import Optional
import numpy as np

from .base import BaseSplitter, SplitResult, SplitConfig, ImageChunk


class GridSplitter(BaseSplitter):
    """
    Splits images using a regular grid with overlap.

    This splitter is used as a fallback when projection and component-based
    methods fail to find suitable split points. It uses configurable overlap
    (default 40%) to ensure text at boundaries appears in multiple chunks,
    allowing post-processing to deduplicate.
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """Initialize the grid splitter."""
        super().__init__(config)

    @property
    def name(self) -> str:
        return "grid"

    def can_split(self, image: np.ndarray) -> bool:
        """
        Grid splitting can always be applied if image needs splitting.

        Args:
            image: Image as numpy array.

        Returns:
            True if the image needs splitting.
        """
        return self.needs_splitting(image)

    def split(self, image: np.ndarray) -> SplitResult:
        """
        Split image using a regular grid with overlap.

        Args:
            image: Image as numpy array.

        Returns:
            SplitResult with overlapping grid chunks.
        """
        if not self.needs_splitting(image):
            return self._create_single_chunk_result(image)

        height, width = image.shape[:2]

        # Calculate grid dimensions
        rows, cols = self._calculate_grid_size(width, height)

        # Calculate chunk sizes with overlap
        chunks = self._create_overlapping_chunks(image, rows, cols)

        return SplitResult(
            chunks=chunks,
            grid_shape=(rows, cols),
            original_size=(width, height),
            split_method=self.name,
            was_split=True,
            metadata={
                "overlap_percent": self.config.overlap_percent,
                "rows": rows,
                "cols": cols,
            },
        )

    def _calculate_grid_size(self, width: int, height: int) -> tuple[int, int]:
        """
        Calculate optimal grid dimensions.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            Tuple of (rows, cols).
        """
        target_size = self.config.target_chunk_size
        max_dim = self.config.max_dimension

        # Use the smaller of target_chunk_size and max_dimension
        effective_max = min(target_size, max_dim)

        # Calculate based on dimensions
        rows = int(np.ceil(height / effective_max))
        cols = int(np.ceil(width / effective_max))

        # Also check megapixels
        megapixels = (width * height) / 1_000_000
        target_mp = self.config.max_megapixels

        if megapixels > target_mp:
            chunks_needed = int(np.ceil(megapixels / target_mp))
            import math
            min_splits = math.ceil(math.sqrt(chunks_needed))
            rows = max(rows, min_splits)
            cols = max(cols, min_splits)

        return max(rows, 1), max(cols, 1)

    def _create_overlapping_chunks(
        self,
        image: np.ndarray,
        rows: int,
        cols: int,
    ) -> list[ImageChunk]:
        """
        Create chunks with overlap.

        Args:
            image: Original image.
            rows: Number of rows.
            cols: Number of columns.

        Returns:
            List of overlapping ImageChunk objects.
        """
        height, width = image.shape[:2]
        overlap = self.config.overlap_percent

        # Calculate base chunk sizes
        base_chunk_height = height / rows
        base_chunk_width = width / cols

        # Calculate overlap in pixels
        overlap_y = int(base_chunk_height * overlap)
        overlap_x = int(base_chunk_width * overlap)

        chunks = []
        index = 0

        for row in range(rows):
            for col in range(cols):
                # Calculate boundaries with overlap
                y_start = int(row * base_chunk_height)
                y_end = int((row + 1) * base_chunk_height)
                x_start = int(col * base_chunk_width)
                x_end = int((col + 1) * base_chunk_width)

                # Extend with overlap (but not at edges)
                actual_overlap_top = 0
                actual_overlap_bottom = 0
                actual_overlap_left = 0
                actual_overlap_right = 0

                if row > 0:
                    actual_overlap_top = overlap_y
                    y_start = max(0, y_start - overlap_y)

                if row < rows - 1:
                    actual_overlap_bottom = overlap_y
                    y_end = min(height, y_end + overlap_y)

                if col > 0:
                    actual_overlap_left = overlap_x
                    x_start = max(0, x_start - overlap_x)

                if col < cols - 1:
                    actual_overlap_right = overlap_x
                    x_end = min(width, x_end + overlap_x)

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
                    overlap_top=actual_overlap_top,
                    overlap_bottom=actual_overlap_bottom,
                    overlap_left=actual_overlap_left,
                    overlap_right=actual_overlap_right,
                )

                chunks.append(chunk)
                index += 1

        return chunks

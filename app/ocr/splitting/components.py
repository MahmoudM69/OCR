"""
Connected component-based image splitter.

Uses connected component analysis to find text regions and splits
at boundaries that don't intersect any text component.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

from .base import BaseSplitter, SplitResult, SplitConfig, ImageChunk


@dataclass
class TextRegion:
    """Represents a detected text region."""

    x: int
    y: int
    width: int
    height: int
    area: int

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2


class ComponentSplitter(BaseSplitter):
    """
    Splits images using connected component analysis.

    This splitter detects text regions as connected components and finds
    split lines that don't intersect any text. It's useful for complex
    layouts where projection profiles may not find clear gaps.
    """

    def __init__(
        self,
        config: Optional[SplitConfig] = None,
        min_component_area: int = 100,
        dilation_kernel: int = 5,
    ):
        """
        Initialize the component splitter.

        Args:
            config: Split configuration.
            min_component_area: Minimum area for a component to be considered text.
            dilation_kernel: Kernel size for dilating components to merge close ones.
        """
        super().__init__(config)
        self.min_component_area = min_component_area
        self.dilation_kernel = dilation_kernel

    @property
    def name(self) -> str:
        return "components"

    def can_split(self, image: np.ndarray) -> bool:
        """
        Check if component-based splitting can find valid split points.

        Args:
            image: Image as numpy array.

        Returns:
            True if valid split lines can be found.
        """
        if not self.needs_splitting(image):
            return False

        regions = self._detect_text_regions(image)
        if not regions:
            return False

        height, width = image.shape[:2]
        target_chunks = self._calculate_target_chunks(width, height)

        h_splits, v_splits = self._find_split_lines(regions, width, height, target_chunks)

        return bool(h_splits or v_splits)

    def split(self, image: np.ndarray) -> SplitResult:
        """
        Split image avoiding text regions.

        Args:
            image: Image as numpy array.

        Returns:
            SplitResult with chunks that don't cut through text.
        """
        if not self.needs_splitting(image):
            return self._create_single_chunk_result(image)

        height, width = image.shape[:2]

        # Detect text regions
        regions = self._detect_text_regions(image)

        if not regions:
            # No text detected, fall back to single chunk
            return self._create_single_chunk_result(image)

        # Calculate target chunks
        target_chunks = self._calculate_target_chunks(width, height)

        # Find split lines that don't intersect text
        h_splits, v_splits = self._find_split_lines(
            regions, width, height, target_chunks
        )

        if not h_splits and not v_splits:
            # No valid splits found
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
                "num_text_regions": len(regions),
            },
        )

    def _detect_text_regions(self, image: np.ndarray) -> list[TextRegion]:
        """
        Detect text regions using connected component analysis.

        Args:
            image: Image as numpy array.

        Returns:
            List of detected text regions.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate to merge nearby components (words into lines)
        kernel = np.ones((self.dilation_kernel, self.dilation_kernel), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )

        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter small components
            if area >= self.min_component_area:
                regions.append(TextRegion(x=x, y=y, width=w, height=h, area=area))

        return regions

    def _calculate_target_chunks(self, width: int, height: int) -> int:
        """Calculate target number of chunks based on image size."""
        megapixels = (width * height) / 1_000_000
        target_mp = self.config.max_megapixels

        chunks_needed = int(np.ceil(megapixels / target_mp))

        dim_chunks_w = int(np.ceil(width / self.config.max_dimension))
        dim_chunks_h = int(np.ceil(height / self.config.max_dimension))
        dim_chunks = dim_chunks_w * dim_chunks_h

        return max(chunks_needed, dim_chunks, 2)

    def _find_split_lines(
        self,
        regions: list[TextRegion],
        width: int,
        height: int,
        target_chunks: int,
    ) -> tuple[list[int], list[int]]:
        """
        Find split lines that don't intersect any text region.

        Args:
            regions: Detected text regions.
            width: Image width.
            height: Image height.
            target_chunks: Target number of chunks.

        Returns:
            Tuple of (horizontal_splits, vertical_splits).
        """
        import math
        sqrt_chunks = math.sqrt(target_chunks)

        # Calculate grid
        rows = int(math.ceil(sqrt_chunks))
        cols = int(math.ceil(target_chunks / rows))

        # Find horizontal splits
        h_splits = []
        if rows > 1:
            h_splits = self._find_horizontal_splits(regions, height, rows - 1)

        # Find vertical splits
        v_splits = []
        if cols > 1:
            v_splits = self._find_vertical_splits(regions, width, cols - 1)

        return h_splits, v_splits

    def _find_horizontal_splits(
        self,
        regions: list[TextRegion],
        height: int,
        num_splits: int,
    ) -> list[int]:
        """
        Find horizontal split positions that don't intersect text.

        Args:
            regions: Text regions.
            height: Image height.
            num_splits: Number of splits needed.

        Returns:
            List of y positions for horizontal splits.
        """
        # Create occupancy map
        occupied = np.zeros(height, dtype=bool)
        for region in regions:
            # Mark region with padding
            padding = self.config.min_gap_pixels // 2
            y_start = max(0, region.y - padding)
            y_end = min(height, region.y_end + padding)
            occupied[y_start:y_end] = True

        # Find candidate positions (unoccupied)
        candidates = np.where(~occupied)[0]

        if len(candidates) == 0:
            return []

        # Find ideal positions
        ideal_spacing = height / (num_splits + 1)
        ideal_positions = [int(ideal_spacing * (i + 1)) for i in range(num_splits)]

        # Select best candidates
        splits = []
        for ideal in ideal_positions:
            if len(candidates) == 0:
                break

            # Find closest candidate to ideal position
            distances = np.abs(candidates - ideal)
            best_idx = np.argmin(distances)

            # Only use if reasonably close to ideal
            if distances[best_idx] < ideal_spacing * 0.5:
                split_pos = candidates[best_idx]

                # Ensure minimum chunk size
                if self._is_valid_split(splits, split_pos, height):
                    splits.append(split_pos)

        return sorted(splits)

    def _find_vertical_splits(
        self,
        regions: list[TextRegion],
        width: int,
        num_splits: int,
    ) -> list[int]:
        """
        Find vertical split positions that don't intersect text.

        Args:
            regions: Text regions.
            width: Image width.
            num_splits: Number of splits needed.

        Returns:
            List of x positions for vertical splits.
        """
        # Create occupancy map
        occupied = np.zeros(width, dtype=bool)
        for region in regions:
            padding = self.config.min_gap_pixels // 2
            x_start = max(0, region.x - padding)
            x_end = min(width, region.x_end + padding)
            occupied[x_start:x_end] = True

        # Find candidate positions
        candidates = np.where(~occupied)[0]

        if len(candidates) == 0:
            return []

        # Find ideal positions
        ideal_spacing = width / (num_splits + 1)
        ideal_positions = [int(ideal_spacing * (i + 1)) for i in range(num_splits)]

        # Select best candidates
        splits = []
        for ideal in ideal_positions:
            if len(candidates) == 0:
                break

            distances = np.abs(candidates - ideal)
            best_idx = np.argmin(distances)

            if distances[best_idx] < ideal_spacing * 0.5:
                split_pos = candidates[best_idx]

                if self._is_valid_split(splits, split_pos, width):
                    splits.append(split_pos)

        return sorted(splits)

    def _is_valid_split(
        self,
        existing_splits: list[int],
        new_split: int,
        dimension: int,
    ) -> bool:
        """
        Check if a new split position is valid.

        Args:
            existing_splits: Already selected splits.
            new_split: Candidate split position.
            dimension: Total dimension.

        Returns:
            True if the split is valid.
        """
        min_size = self.config.min_chunk_size

        # Check distance from edges
        if new_split < min_size or new_split > dimension - min_size:
            return False

        # Check distance from existing splits
        for split in existing_splits:
            if abs(split - new_split) < min_size:
                return False

        return True

    def _create_chunks(
        self,
        image: np.ndarray,
        h_splits: list[int],
        v_splits: list[int],
    ) -> list[ImageChunk]:
        """Create image chunks from split positions."""
        height, width = image.shape[:2]

        y_boundaries = [0] + sorted(h_splits) + [height]
        x_boundaries = [0] + sorted(v_splits) + [width]

        chunks = []
        index = 0

        for row, (y_start, y_end) in enumerate(zip(y_boundaries[:-1], y_boundaries[1:])):
            for col, (x_start, x_end) in enumerate(zip(x_boundaries[:-1], x_boundaries[1:])):
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

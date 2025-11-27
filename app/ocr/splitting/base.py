"""
Base classes for image splitting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class ImageChunk:
    """Represents a single chunk of a split image."""

    image: np.ndarray
    """The chunk image data as numpy array."""

    index: int
    """Sequential index of this chunk."""

    row: int
    """Row position in the grid (0-indexed)."""

    col: int
    """Column position in the grid (0-indexed)."""

    x_offset: int
    """X offset from original image origin."""

    y_offset: int
    """Y offset from original image origin."""

    width: int
    """Width of the chunk."""

    height: int
    """Height of the chunk."""

    overlap_top: int = 0
    """Pixels of overlap with chunk above."""

    overlap_bottom: int = 0
    """Pixels of overlap with chunk below."""

    overlap_left: int = 0
    """Pixels of overlap with chunk to the left."""

    overlap_right: int = 0
    """Pixels of overlap with chunk to the right."""

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) bounds in original image."""
        return (self.x_offset, self.y_offset, self.width, self.height)

    def to_pil(self) -> Image.Image:
        """Convert chunk to PIL Image."""
        return Image.fromarray(self.image)

    def save(self, path: Path) -> Path:
        """Save chunk to file."""
        self.to_pil().save(path)
        return path


@dataclass
class SplitResult:
    """Result of splitting an image."""

    chunks: list[ImageChunk]
    """List of image chunks."""

    grid_shape: tuple[int, int]
    """Shape of the grid (rows, cols)."""

    original_size: tuple[int, int]
    """Original image size (width, height)."""

    split_method: str
    """Method used for splitting (projection, components, grid)."""

    was_split: bool
    """Whether the image was actually split."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata about the split."""

    @property
    def num_chunks(self) -> int:
        """Total number of chunks."""
        return len(self.chunks)

    @property
    def rows(self) -> int:
        """Number of rows in the grid."""
        return self.grid_shape[0]

    @property
    def cols(self) -> int:
        """Number of columns in the grid."""
        return self.grid_shape[1]


@dataclass
class SplitConfig:
    """Configuration for image splitting."""

    enabled: bool = True
    """Whether splitting is enabled (False to skip splitting entirely)."""

    max_megapixels: float = 2.0
    """Maximum megapixels before splitting is triggered."""

    max_dimension: int = 2048
    """Maximum dimension (width or height) before splitting."""

    overlap_percent: float = 0.4
    """Overlap percentage for grid fallback (0.0 to 1.0)."""

    min_gap_pixels: int = 10
    """Minimum whitespace gap to consider for splitting."""

    gap_threshold: float = 0.95
    """Threshold for detecting whitespace (0.0 to 1.0)."""

    min_chunk_size: int = 256
    """Minimum chunk dimension to avoid tiny chunks."""

    target_chunk_size: int = 1024
    """Target chunk size when splitting."""

    prefer_horizontal_splits: bool = False
    """Prefer horizontal strips over vertical cuts (better for RTL text like Arabic)."""


class BaseSplitter(ABC):
    """Abstract base class for image splitters."""

    def __init__(self, config: Optional[SplitConfig] = None):
        """Initialize splitter with configuration."""
        self.config = config or SplitConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this splitting method."""
        pass

    @abstractmethod
    def can_split(self, image: np.ndarray) -> bool:
        """
        Check if this splitter can handle the given image.

        Args:
            image: Image as numpy array.

        Returns:
            True if this splitter can find valid split points.
        """
        pass

    @abstractmethod
    def split(self, image: np.ndarray) -> SplitResult:
        """
        Split the image into chunks.

        Args:
            image: Image as numpy array.

        Returns:
            SplitResult containing the chunks and metadata.
        """
        pass

    def needs_splitting(self, image: np.ndarray) -> bool:
        """
        Check if the image needs to be split based on size.

        Args:
            image: Image as numpy array.

        Returns:
            True if the image exceeds size thresholds.
        """
        height, width = image.shape[:2]
        megapixels = (width * height) / 1_000_000

        return (
            megapixels > self.config.max_megapixels or
            width > self.config.max_dimension or
            height > self.config.max_dimension
        )

    def _create_single_chunk_result(self, image: np.ndarray) -> SplitResult:
        """Create a result for an image that doesn't need splitting."""
        height, width = image.shape[:2]

        chunk = ImageChunk(
            image=image,
            index=0,
            row=0,
            col=0,
            x_offset=0,
            y_offset=0,
            width=width,
            height=height,
        )

        return SplitResult(
            chunks=[chunk],
            grid_shape=(1, 1),
            original_size=(width, height),
            split_method="none",
            was_split=False,
        )

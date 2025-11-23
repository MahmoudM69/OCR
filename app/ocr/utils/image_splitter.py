"""
DEPRECATED: This module is deprecated in favor of app.ocr.splitting.

Use the new content-aware splitting module instead:
    from app.ocr.splitting import SmartSplitter, create_splitter

The new module provides:
- Projection profile splitting (finds whitespace gaps)
- Connected component splitting (avoids cutting through text)
- Grid fallback with configurable overlap
- RTL/LTR aware result merging with deduplication

This module is kept for backward compatibility only.
"""

import logging
import math
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Issue deprecation warning on import
warnings.warn(
    "app.ocr.utils.image_splitter is deprecated. "
    "Use app.ocr.splitting.SmartSplitter instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Thresholds for auto-splitting
MAX_MEGAPIXELS = 2.0
MAX_DIMENSION = 2048


@dataclass
class ImageInfo:
    """Information about an image."""

    width: int
    height: int
    megapixels: float
    needs_splitting: bool
    suggested_grid: tuple[int, int]  # (rows, cols)


def get_image_info(image_path: Path) -> ImageInfo:
    """
    Analyze an image and determine if it needs splitting.

    Args:
        image_path: Path to the image file.

    Returns:
        ImageInfo with dimensions and splitting recommendation.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        megapixels = (width * height) / 1_000_000

        needs_splitting = (
            megapixels > MAX_MEGAPIXELS
            or width > MAX_DIMENSION
            or height > MAX_DIMENSION
        )

        # Calculate suggested grid based on image size
        if needs_splitting:
            # Target ~1MP per chunk
            target_chunks = math.ceil(megapixels / 1.0)
            # Prefer more columns for wide images, more rows for tall
            aspect_ratio = width / height
            if aspect_ratio > 1.5:  # Wide image
                cols = min(4, math.ceil(math.sqrt(target_chunks * aspect_ratio)))
                rows = math.ceil(target_chunks / cols)
            elif aspect_ratio < 0.67:  # Tall image
                rows = min(4, math.ceil(math.sqrt(target_chunks / aspect_ratio)))
                cols = math.ceil(target_chunks / rows)
            else:  # Square-ish
                rows = cols = math.ceil(math.sqrt(target_chunks))
            suggested_grid = (max(1, rows), max(1, cols))
        else:
            suggested_grid = (1, 1)

        return ImageInfo(
            width=width,
            height=height,
            megapixels=megapixels,
            needs_splitting=needs_splitting,
            suggested_grid=suggested_grid,
        )


def split_image_grid(
    image_path: Path,
    rows: int = 2,
    cols: int = 2,
    overlap: float = 0.1,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Split an image into a grid of overlapping chunks.

    Args:
        image_path: Path to the image file.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        overlap: Overlap ratio between chunks (0.0 to 0.5).
        output_dir: Directory to save chunks. Creates temp dir if None.

    Returns:
        List of paths to chunk images, ordered left-to-right, top-to-bottom.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ocr_chunks_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        width, height = img.size

        # Calculate chunk dimensions with overlap
        chunk_width = width // cols
        chunk_height = height // rows
        overlap_x = int(chunk_width * overlap)
        overlap_y = int(chunk_height * overlap)

        chunks = []
        for row in range(rows):
            for col in range(cols):
                # Calculate boundaries with overlap
                x1 = max(0, col * chunk_width - overlap_x)
                y1 = max(0, row * chunk_height - overlap_y)
                x2 = min(width, (col + 1) * chunk_width + overlap_x)
                y2 = min(height, (row + 1) * chunk_height + overlap_y)

                # Crop and save chunk
                chunk = img.crop((x1, y1, x2, y2))
                chunk_path = output_dir / f"chunk_{row}_{col}.png"
                chunk.save(chunk_path)
                chunks.append(chunk_path)

                logger.debug(f"Created chunk {row},{col}: {x1},{y1} to {x2},{y2}")

        logger.info(f"Split image into {len(chunks)} chunks ({rows}x{cols})")
        return chunks


def split_image_by_columns(
    image_path: Path,
    num_columns: int = 2,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Split an image vertically into columns (for newspaper-style documents).

    Args:
        image_path: Path to the image file.
        num_columns: Number of vertical columns.
        output_dir: Directory to save chunks.

    Returns:
        List of paths to column images, ordered left-to-right.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ocr_columns_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        width, height = img.size
        column_width = width // num_columns

        columns = []
        for i in range(num_columns):
            x1 = i * column_width
            x2 = width if i == num_columns - 1 else (i + 1) * column_width

            column = img.crop((x1, 0, x2, height))
            column_path = output_dir / f"column_{i}.png"
            column.save(column_path)
            columns.append(column_path)

        logger.info(f"Split image into {num_columns} columns")
        return columns

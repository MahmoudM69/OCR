import logging
import shutil
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


async def process_image_chunks(
    chunk_paths: list[Path],
    process_func: Callable[[Path], str],
    reading_order: str = "ltr",
    grid_shape: tuple[int, int] | None = None,
) -> str:
    """
    Process multiple image chunks and combine results.

    Args:
        chunk_paths: List of paths to chunk images.
        process_func: Async function that takes image path and returns text.
        reading_order: "ltr" (left-to-right) or "rtl" (right-to-left).
        grid_shape: (rows, cols) if chunks are from a grid split.

    Returns:
        Combined text from all chunks.
    """
    results = []

    for i, chunk_path in enumerate(chunk_paths):
        try:
            text = await process_func(chunk_path)
            if text and text.strip():
                results.append(text.strip())
                logger.debug(f"Processed chunk {i}: {len(text)} chars")
        except Exception as e:
            logger.error(f"Failed to process chunk {i}: {e}")
            continue

    if not results:
        return ""

    # Handle RTL reading order for Arabic text
    if reading_order == "rtl" and grid_shape:
        rows, cols = grid_shape
        reordered = []
        for row in range(rows):
            row_start = row * cols
            row_end = row_start + cols
            row_chunks = results[row_start:row_end]
            # Reverse each row for RTL
            reordered.extend(reversed(row_chunks))
        results = reordered

    # Combine with section markers
    if len(results) == 1:
        return results[0]

    combined = []
    for i, text in enumerate(results):
        combined.append(f"--- Section {i + 1} ---\n{text}")

    return "\n\n".join(combined)


def cleanup_temp_chunks(chunk_dir: Path) -> None:
    """
    Remove temporary chunk directory and its contents.

    Args:
        chunk_dir: Path to the temporary directory.
    """
    if chunk_dir.exists() and chunk_dir.is_dir():
        try:
            shutil.rmtree(chunk_dir)
            logger.debug(f"Cleaned up temp directory: {chunk_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

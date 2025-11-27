"""
Smart image splitter that uses a hybrid approach.

Tries projection profile first, falls back to connected components,
and finally uses grid with overlap as last resort.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

from .base import BaseSplitter, SplitResult, SplitConfig
from .projection import ProjectionSplitter
from .components import ComponentSplitter
from .grid import GridSplitter


class SmartSplitter:
    """
    Intelligent image splitter using a hybrid approach.

    Split strategy priority:
    1. Projection profile - Find natural whitespace gaps
    2. Connected components - Find gaps between text regions
    3. Grid with overlap - Fallback with deduplication support
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize the smart splitter.

        Args:
            config: Shared configuration for all splitters.
        """
        self.config = config or SplitConfig()

        # Initialize splitters in priority order
        self._splitters: list[BaseSplitter] = [
            ProjectionSplitter(self.config),
            ComponentSplitter(self.config),
            GridSplitter(self.config),
        ]

    @property
    def splitters(self) -> list[BaseSplitter]:
        """Get the list of splitters in priority order."""
        return self._splitters

    def split(self, image: Union[np.ndarray, Image.Image, Path, str]) -> SplitResult:
        """
        Split an image using the best available method.

        Args:
            image: Image as numpy array, PIL Image, or path.

        Returns:
            SplitResult from the first successful splitter.
        """
        # Convert to numpy array if needed
        img_array = self._to_numpy(image)

        # Check if splitting is needed
        if not self._needs_splitting(img_array):
            return self._create_single_chunk_result(img_array)

        # Try each splitter in order
        for splitter in self._splitters:
            if splitter.can_split(img_array):
                result = splitter.split(img_array)
                if result.was_split:
                    return result

        # This shouldn't happen as GridSplitter always works
        # but return single chunk as safety
        return self._create_single_chunk_result(img_array)

    def split_with_method(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        method: str,
    ) -> SplitResult:
        """
        Split using a specific method.

        Args:
            image: Image to split.
            method: Method name ('projection', 'components', 'grid').

        Returns:
            SplitResult from the specified splitter.

        Raises:
            ValueError: If method is not recognized.
        """
        img_array = self._to_numpy(image)

        for splitter in self._splitters:
            if splitter.name == method:
                return splitter.split(img_array)

        raise ValueError(
            f"Unknown split method: {method}. "
            f"Available: {[s.name for s in self._splitters]}"
        )

    def analyze(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
    ) -> dict:
        """
        Analyze an image and return splitting recommendations.

        Args:
            image: Image to analyze.

        Returns:
            Dictionary with analysis results and recommendations.
        """
        img_array = self._to_numpy(image)
        height, width = img_array.shape[:2]
        megapixels = (width * height) / 1_000_000

        needs_split = self._needs_splitting(img_array)

        # Check which methods can split
        available_methods = []
        recommended_method = None

        for splitter in self._splitters:
            can_split = splitter.can_split(img_array) if needs_split else False
            available_methods.append({
                "name": splitter.name,
                "can_split": can_split,
            })
            if can_split and recommended_method is None:
                recommended_method = splitter.name

        return {
            "width": width,
            "height": height,
            "megapixels": megapixels,
            "needs_splitting": needs_split,
            "max_megapixels": self.config.max_megapixels,
            "max_dimension": self.config.max_dimension,
            "available_methods": available_methods,
            "recommended_method": recommended_method,
        }

    def _to_numpy(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
    ) -> np.ndarray:
        """Convert various image types to numpy array."""
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            return np.array(image)

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _needs_splitting(self, image: np.ndarray) -> bool:
        """Check if image needs splitting based on size."""
        height, width = image.shape[:2]
        megapixels = (width * height) / 1_000_000

        return (
            megapixels > self.config.max_megapixels or
            width > self.config.max_dimension or
            height > self.config.max_dimension
        )

    def _create_single_chunk_result(self, image: np.ndarray) -> SplitResult:
        """Create result for image that doesn't need splitting."""
        from .base import ImageChunk

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


def create_splitter(
    max_megapixels: float = 2.0,
    max_dimension: int = 2048,
    overlap_percent: float = 0.4,
    **kwargs,
) -> SmartSplitter:
    """
    Factory function to create a configured SmartSplitter.

    Args:
        max_megapixels: Maximum megapixels before splitting.
        max_dimension: Maximum dimension before splitting.
        overlap_percent: Overlap for grid fallback.
        **kwargs: Additional SplitConfig options.

    Returns:
        Configured SmartSplitter.
    """
    config = SplitConfig(
        max_megapixels=max_megapixels,
        max_dimension=max_dimension,
        overlap_percent=overlap_percent,
        **kwargs,
    )
    return SmartSplitter(config)

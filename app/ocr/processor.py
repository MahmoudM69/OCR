"""
Unified image processor for OCR.

Combines preprocessing and content-aware splitting into a single
easy-to-use interface for OCR engines.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union, Awaitable
import numpy as np
from PIL import Image

from app.config import settings, SplittingConfig, PreprocessingConfig

from .preprocessing import PreprocessingPipeline, create_pipeline
from .preprocessing.base import PreprocessingConfig as PipelinePreprocessingConfig
from .splitting import (
    SmartSplitter,
    SplitResult,
    SplitConfig,
    ResultMerger,
    MergeConfig,
    ChunkResult,
)


logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    """Result of processing an image through preprocessing and splitting."""

    chunks: list[np.ndarray]
    """List of preprocessed image chunks (or single image if not split)."""

    was_split: bool
    """Whether the image was split."""

    split_result: Optional[SplitResult]
    """Split result metadata (None if not split)."""

    grid_shape: tuple[int, int]
    """Grid shape (1, 1) if not split."""

    preprocessing_applied: list[str]
    """Names of preprocessing steps that were applied."""

    original_size: tuple[int, int]
    """Original image size (width, height)."""

    metadata: dict = field(default_factory=dict)
    """Additional processing metadata."""


class ImageProcessor:
    """
    Unified image processor for OCR engines.

    Combines:
    1. Preprocessing (denoise, deskew, binarize, etc.)
    2. Content-aware splitting (projection, components, grid fallback)
    3. Result merging with deduplication
    """

    def __init__(
        self,
        engine_name: str,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        splitting_config: Optional[SplittingConfig] = None,
    ):
        """
        Initialize the processor.

        Args:
            engine_name: Name of the OCR engine (for config lookup).
            preprocessing_config: Override preprocessing config.
            splitting_config: Override splitting config.
        """
        self.engine_name = engine_name

        # Get configs from settings if not provided
        engine_config = settings.get_engine_config(engine_name)

        # Setup preprocessing
        if preprocessing_config:
            self._preprocessing_config = preprocessing_config
        else:
            self._preprocessing_config = engine_config.preprocessing

        # Convert to pipeline config
        pipeline_config = PipelinePreprocessingConfig(
            enabled=self._preprocessing_config.enabled,
            target_dpi=self._preprocessing_config.target_dpi,
            denoise_strength=self._preprocessing_config.denoise_strength,
            binarization_method=self._preprocessing_config.binarization_method,
            auto_deskew=self._preprocessing_config.auto_deskew,
            blur_threshold=self._preprocessing_config.blur_threshold,
            noise_threshold=self._preprocessing_config.noise_threshold,
            skew_threshold=self._preprocessing_config.skew_threshold,
            contrast_threshold=self._preprocessing_config.contrast_threshold,
        )
        self._preprocessor = PreprocessingPipeline(pipeline_config)

        # Setup splitting
        if splitting_config:
            self._splitting_config = splitting_config
        else:
            self._splitting_config = settings.get_splitting_config(engine_name)

        split_config = SplitConfig(
            max_megapixels=self._splitting_config.max_megapixels,
            max_dimension=self._splitting_config.max_dimension,
            overlap_percent=self._splitting_config.overlap_percent,
            min_gap_pixels=self._splitting_config.min_gap_pixels,
            gap_threshold=self._splitting_config.gap_threshold,
            min_chunk_size=self._splitting_config.min_chunk_size,
            target_chunk_size=self._splitting_config.target_chunk_size,
        )
        self._splitter = SmartSplitter(split_config)

    def process(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
    ) -> ProcessedImage:
        """
        Process an image through preprocessing and splitting.

        Args:
            image: Input image.

        Returns:
            ProcessedImage with chunks ready for OCR.
        """
        # Convert to numpy array
        img_array = self._to_numpy(image)
        original_size = (img_array.shape[1], img_array.shape[0])

        # Check if splitting is needed BEFORE preprocessing
        # (we need to split first, then preprocess each chunk)
        split_result = self._splitter.split(img_array)

        if split_result.was_split:
            # Preprocess each chunk
            processed_chunks = []
            all_preprocessing_steps = set()

            for chunk in split_result.chunks:
                preproc_result = self._preprocessor.process(chunk.image)
                processed_chunks.append(preproc_result.image)
                all_preprocessing_steps.update(preproc_result.steps_applied)

            return ProcessedImage(
                chunks=processed_chunks,
                was_split=True,
                split_result=split_result,
                grid_shape=split_result.grid_shape,
                preprocessing_applied=list(all_preprocessing_steps),
                original_size=original_size,
                metadata={
                    "split_method": split_result.split_method,
                    "num_chunks": len(processed_chunks),
                },
            )
        else:
            # Single image - just preprocess
            preproc_result = self._preprocessor.process(img_array)

            return ProcessedImage(
                chunks=[preproc_result.image],
                was_split=False,
                split_result=None,
                grid_shape=(1, 1),
                preprocessing_applied=preproc_result.steps_applied,
                original_size=original_size,
            )

    async def process_with_ocr(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        ocr_func: Callable[[np.ndarray], Awaitable[str]],
        rtl: bool = False,
    ) -> tuple[str, dict]:
        """
        Process image and run OCR on all chunks.

        Args:
            image: Input image.
            ocr_func: Async function that takes numpy array and returns text.
            rtl: Whether text is right-to-left (for Arabic).

        Returns:
            Tuple of (merged_text, metadata).
        """
        # DEBUG: Log input validation
        logger.debug(f"[ImageProcessor] Processing image type: {type(image)}, RTL: {rtl}")
        
        # Process image
        processed = self.process(image)
        
        # DEBUG: Log processing results
        logger.debug(f"[ImageProcessor] Processed {len(processed.chunks)} chunks, was_split: {processed.was_split}")

        if not processed.was_split:
            # Single chunk - just run OCR
            try:
                text = await ocr_func(processed.chunks[0])
            except Exception as e:
                logger.error(f"OCR failed on single chunk: {e}")
                raise RuntimeError(f"OCR processing failed: {e}") from e
            return text, {
                "was_split": False,
                "preprocessing_applied": processed.preprocessing_applied,
            }

        # Multiple chunks - run OCR on each and merge
        chunk_results = []
        for i, (chunk_image, chunk_info) in enumerate(
            zip(processed.chunks, processed.split_result.chunks)
        ):
            try:
                text = await ocr_func(chunk_image)
            except Exception as e:
                logger.error(f"OCR failed on chunk {i} (row={chunk_info.row}, col={chunk_info.col}): {e}")
                raise RuntimeError(f"OCR processing failed on chunk {i}: {e}") from e
            chunk_results.append(ChunkResult(
                chunk=chunk_info,
                text=text,
            ))

        # Merge results
        merger = ResultMerger(MergeConfig(rtl=rtl))
        merged_text = merger.merge(chunk_results, processed.split_result)

        return merged_text, {
            "was_split": True,
            "split_method": processed.split_result.split_method,
            "grid_shape": processed.grid_shape,
            "num_chunks": len(processed.chunks),
            "preprocessing_applied": processed.preprocessing_applied,
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
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            return np.array(image)

        raise TypeError(f"Unsupported image type: {type(image)}")


def create_processor(
    engine_name: str,
    **kwargs,
) -> ImageProcessor:
    """
    Factory function to create a processor for a specific engine.

    Args:
        engine_name: Name of the OCR engine.
        **kwargs: Override configuration options.

    Returns:
        Configured ImageProcessor.
    """
    return ImageProcessor(engine_name, **kwargs)

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
import cv2
from PIL import Image

from app.config import settings, SplittingConfig, PreprocessingConfig

from .preprocessing import PreprocessingPipeline, create_pipeline
from .preprocessing.base import PreprocessingConfig as PipelinePreprocessingConfig
from .preprocessing.analyzer import ImageQualityAnalyzer
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

        # Convert to pipeline config (disable per-chunk deskew if we do global deskew)
        # Global deskew is applied before splitting for consistent alignment
        pipeline_config = PipelinePreprocessingConfig(
            enabled=self._preprocessing_config.enabled,
            target_dpi=self._preprocessing_config.target_dpi,
            max_scale_factor=getattr(self._preprocessing_config, 'max_scale_factor', 3.0),
            denoise_strength=self._preprocessing_config.denoise_strength,
            binarization_method=self._preprocessing_config.binarization_method,
            auto_deskew=False,  # Disable per-chunk deskew (we do global deskew)
            auto_invert=getattr(self._preprocessing_config, 'auto_invert', True),
            preserve_color=getattr(self._preprocessing_config, 'preserve_color', False),
            blur_threshold=self._preprocessing_config.blur_threshold,
            noise_threshold=self._preprocessing_config.noise_threshold,
            skew_threshold=self._preprocessing_config.skew_threshold,
            contrast_threshold=self._preprocessing_config.contrast_threshold,
        )
        self._preprocessor = PreprocessingPipeline(pipeline_config)

        # Store deskew setting for global application
        self._global_deskew = self._preprocessing_config.auto_deskew
        self._skew_threshold = self._preprocessing_config.skew_threshold

        # Setup splitting
        if splitting_config:
            self._splitting_config = splitting_config
        else:
            self._splitting_config = settings.get_splitting_config(engine_name)

        split_config = SplitConfig(
            enabled=getattr(self._splitting_config, 'enabled', True),
            max_megapixels=self._splitting_config.max_megapixels,
            max_dimension=self._splitting_config.max_dimension,
            overlap_percent=self._splitting_config.overlap_percent,
            min_gap_pixels=self._splitting_config.min_gap_pixels,
            gap_threshold=self._splitting_config.gap_threshold,
            min_chunk_size=self._splitting_config.min_chunk_size,
            target_chunk_size=self._splitting_config.target_chunk_size,
            prefer_horizontal_splits=getattr(self._splitting_config, 'prefer_horizontal_splits', False),
        )
        self._splitter = SmartSplitter(split_config)

    def process(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
    ) -> ProcessedImage:
        """
        Process an image through preprocessing and splitting.

        Flow:
        1. Apply global deskew (if enabled) - ensures consistent alignment
        2. Split image if needed (based on size thresholds)
        3. Apply per-chunk preprocessing (without deskew)

        Args:
            image: Input image.

        Returns:
            ProcessedImage with chunks ready for OCR.
        """
        # Convert to numpy array
        img_array = self._to_numpy(image)
        original_size = (img_array.shape[1], img_array.shape[0])
        global_steps_applied = []

        # Step 1: Apply global deskew BEFORE splitting
        # This ensures all chunks have consistent text alignment
        if self._global_deskew:
            img_array, was_deskewed = self._apply_global_deskew(img_array)
            if was_deskewed:
                global_steps_applied.append("global_deskew")

        # Step 2: Check if splitting is needed (or disabled)
        splitting_enabled = getattr(self._splitter.config, 'enabled', True)

        if not splitting_enabled:
            # Splitting disabled - process as single image
            preproc_result = self._preprocessor.process(img_array)
            all_steps = global_steps_applied + preproc_result.steps_applied

            return ProcessedImage(
                chunks=[preproc_result.image],
                was_split=False,
                split_result=None,
                grid_shape=(1, 1),
                preprocessing_applied=all_steps,
                original_size=original_size,
                metadata={"splitting_disabled": True},
            )

        # Step 3: Split if needed
        split_result = self._splitter.split(img_array)

        if split_result.was_split:
            # Step 4: Preprocess each chunk (without deskew - already done globally)
            processed_chunks = []
            all_preprocessing_steps = set(global_steps_applied)

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
            all_steps = global_steps_applied + preproc_result.steps_applied

            return ProcessedImage(
                chunks=[preproc_result.image],
                was_split=False,
                split_result=None,
                grid_shape=(1, 1),
                preprocessing_applied=all_steps,
                original_size=original_size,
            )

    def _apply_global_deskew(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Apply deskew to the entire image before splitting.

        This ensures consistent text alignment across all chunks.

        Args:
            image: Input image.

        Returns:
            Tuple of (deskewed_image, was_deskewed).
        """
        # Analyze image to get skew angle
        analyzer = ImageQualityAnalyzer(self._preprocessor.config)
        analysis = analyzer.analyze(image)

        # Check if deskew is needed
        if abs(analysis.skew_angle) < self._skew_threshold:
            return image, False

        logger.debug(f"[ImageProcessor] Applying global deskew: {analysis.skew_angle:.2f} degrees")

        # Apply rotation to correct skew
        angle = -analysis.skew_angle
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box size
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int(height * sin_angle + width * cos_angle)
        new_height = int(height * cos_angle + width * sin_angle)

        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2

        # Determine background color (white for documents)
        if len(image.shape) == 3:
            border_color = (255, 255, 255)
        else:
            border_color = 255

        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color,
        )

        return rotated, True

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

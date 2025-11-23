"""
Preprocessing pipeline orchestrator.

Chains preprocessing steps and applies them based on image analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

from .base import PreprocessingStep, PreprocessingConfig, ImageAnalysis, StepResult
from .analyzer import ImageQualityAnalyzer
from .steps import (
    GrayscaleStep,
    NormalizationStep,
    NoiseRemovalStep,
    BinarizationStep,
    DeskewStep,
    ScalingStep,
)


@dataclass
class PipelineResult:
    """Result of running the preprocessing pipeline."""

    image: np.ndarray
    """Preprocessed image."""

    original_size: tuple[int, int]
    """Original image size (width, height)."""

    final_size: tuple[int, int]
    """Final image size after preprocessing."""

    analysis: ImageAnalysis
    """Initial image analysis."""

    steps_applied: list[str]
    """Names of steps that were applied."""

    steps_skipped: list[str]
    """Names of steps that were skipped."""

    step_results: list[StepResult] = field(default_factory=list)
    """Detailed results for each step."""

    @property
    def was_modified(self) -> bool:
        """Whether the image was modified by any step."""
        return len(self.steps_applied) > 0


class PreprocessingPipeline:
    """
    Orchestrates preprocessing steps for OCR optimization.

    Steps are applied in order, with each step deciding whether
    to apply itself based on image analysis.

    Default step order:
    1. Grayscale conversion
    2. Noise removal
    3. Deskew
    4. Normalization
    5. Binarization
    6. Scaling
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        steps: Optional[list[PreprocessingStep]] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Preprocessing configuration.
            steps: Custom list of steps. If None, uses default steps.
        """
        self.config = config or PreprocessingConfig()
        self.analyzer = ImageQualityAnalyzer(self.config)

        if steps is not None:
            self._steps = steps
        else:
            self._steps = self._create_default_steps()

    def _create_default_steps(self) -> list[PreprocessingStep]:
        """Create the default preprocessing steps."""
        return [
            GrayscaleStep(self.config),
            NoiseRemovalStep(self.config),
            DeskewStep(self.config),
            NormalizationStep(self.config),
            BinarizationStep(self.config),
            ScalingStep(self.config),
        ]

    @property
    def steps(self) -> list[PreprocessingStep]:
        """Get the list of preprocessing steps."""
        return self._steps

    def process(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        force_all: bool = False,
    ) -> PipelineResult:
        """
        Process an image through the preprocessing pipeline.

        Args:
            image: Input image as numpy array, PIL Image, or path.
            force_all: Force all steps to apply regardless of analysis.

        Returns:
            PipelineResult with processed image and metadata.
        """
        # Convert to numpy array
        img_array = self._to_numpy(image)
        original_size = (img_array.shape[1], img_array.shape[0])

        # Skip if preprocessing is disabled
        if not self.config.enabled:
            return PipelineResult(
                image=img_array,
                original_size=original_size,
                final_size=original_size,
                analysis=self.analyzer.analyze(img_array),
                steps_applied=[],
                steps_skipped=[s.name for s in self._steps],
            )

        # Analyze image
        analysis = self.analyzer.analyze(img_array)

        # Apply steps
        current_image = img_array
        steps_applied = []
        steps_skipped = []
        step_results = []

        for step in self._steps:
            result = step.process(current_image, analysis, force=force_all)
            step_results.append(result)

            if result.applied:
                current_image = result.image
                steps_applied.append(step.name)
            else:
                steps_skipped.append(step.name)

        final_size = (current_image.shape[1], current_image.shape[0])

        return PipelineResult(
            image=current_image,
            original_size=original_size,
            final_size=final_size,
            analysis=analysis,
            steps_applied=steps_applied,
            steps_skipped=steps_skipped,
            step_results=step_results,
        )

    def add_step(self, step: PreprocessingStep, position: Optional[int] = None) -> None:
        """
        Add a preprocessing step.

        Args:
            step: Step to add.
            position: Position in the pipeline. If None, adds at end.
        """
        if position is None:
            self._steps.append(step)
        else:
            self._steps.insert(position, step)

    def remove_step(self, name: str) -> bool:
        """
        Remove a step by name.

        Args:
            name: Name of step to remove.

        Returns:
            True if step was found and removed.
        """
        for i, step in enumerate(self._steps):
            if step.name == name:
                self._steps.pop(i)
                return True
        return False

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


def create_pipeline(
    enabled: bool = True,
    target_dpi: int = 300,
    denoise_strength: int = 10,
    binarization_method: str = "adaptive",
    auto_deskew: bool = True,
    **kwargs,
) -> PreprocessingPipeline:
    """
    Factory function to create a configured preprocessing pipeline.

    Args:
        enabled: Whether preprocessing is enabled.
        target_dpi: Target DPI for scaling.
        denoise_strength: Denoising strength.
        binarization_method: 'otsu', 'adaptive', or 'none'.
        auto_deskew: Whether to auto-deskew.
        **kwargs: Additional PreprocessingConfig options.

    Returns:
        Configured PreprocessingPipeline.
    """
    config = PreprocessingConfig(
        enabled=enabled,
        target_dpi=target_dpi,
        denoise_strength=denoise_strength,
        binarization_method=binarization_method,
        auto_deskew=auto_deskew,
        **kwargs,
    )
    return PreprocessingPipeline(config)


def create_pipeline_for_engine(engine_name: str) -> PreprocessingPipeline:
    """
    Create a pipeline with engine-specific defaults.

    Args:
        engine_name: Name of the OCR engine.

    Returns:
        Pipeline configured for the specific engine.
    """
    # Engine-specific defaults
    engine_configs = {
        "qari": PreprocessingConfig(
            binarization_method="none",  # Arabic text often better without binarization
            auto_deskew=True,
        ),
        "got": PreprocessingConfig(
            binarization_method="adaptive",
            auto_deskew=True,
        ),
        "deepseek": PreprocessingConfig(
            binarization_method="adaptive",
            auto_deskew=True,
        ),
    }

    config = engine_configs.get(engine_name, PreprocessingConfig())
    return PreprocessingPipeline(config)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BoundingBox:
    """Bounding box for detected text region."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class TextBlock:
    """A block of detected text with optional position info."""

    text: str
    confidence: float = 0.0
    bounding_box: BoundingBox | None = None


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    formatted_text: str = ""  # Formatted/structured output from model
    output_file: str | None = None  # Path to output file (for internal use)
    blocks: list[TextBlock] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseOCREngine(ABC):
    """
    Abstract base class for OCR engines.

    All OCR implementations must inherit from this class and implement
    the required methods for lifecycle management and text extraction.
    """

    def __init__(self, model_path: Path):
        """
        Initialize the OCR engine.

        Args:
            model_path: Path to the directory containing model files.
        """
        self.model_path = model_path
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this OCR engine.

        Returns:
            Engine name (e.g., 'tesseract', 'easyocr', 'paddleocr').
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._loaded

    @abstractmethod
    async def load(self) -> None:
        """
        Load the model into memory.

        This method should load all necessary model files from self.model_path
        and prepare the engine for inference.

        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model loading fails.
        """
        pass

    @abstractmethod
    async def unload(self) -> None:
        """
        Unload the model and free memory.

        This method should release all resources held by the model,
        including GPU memory if applicable.
        """
        pass

    @abstractmethod
    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image_path: Path to the image file to process.

        Returns:
            OCRResult containing extracted text and metadata.

        Raises:
            RuntimeError: If the model is not loaded.
            FileNotFoundError: If the image file doesn't exist.
        """
        pass

    def _ensure_loaded(self) -> None:
        """Raise an error if the model is not loaded."""
        if not self._loaded:
            raise RuntimeError(f"Model '{self.name}' is not loaded. Call load() first.")

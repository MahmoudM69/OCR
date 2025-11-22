import asyncio
import logging
from pathlib import Path

from app.ocr.base import BaseOCREngine, OCRResult, TextBlock, BoundingBox
from app.ocr.registry import OCREngineRegistry

logger = logging.getLogger(__name__)


@OCREngineRegistry.register
class MockOCREngine(BaseOCREngine):
    """
    Mock OCR engine for testing.

    This engine simulates OCR processing with configurable delays
    and returns placeholder text. Useful for testing the API without
    needing actual OCR models installed.
    """

    @property
    def name(self) -> str:
        return "mock"

    async def load(self) -> None:
        """Simulate model loading."""
        logger.info(f"Loading mock model from {self.model_path}")
        await asyncio.sleep(0.5)  # Simulate loading time
        self._loaded = True
        logger.info("Mock model loaded")

    async def unload(self) -> None:
        """Simulate model unloading."""
        logger.info("Unloading mock model")
        await asyncio.sleep(0.1)  # Simulate unloading time
        self._loaded = False
        logger.info("Mock model unloaded")

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Simulate OCR text extraction.

        Returns placeholder text based on the image filename.
        """
        self._ensure_loaded()

        logger.info(f"Processing image: {image_path}")
        await asyncio.sleep(1.0)  # Simulate processing time

        # Generate mock result
        filename = image_path.name
        mock_text = f"Mock OCR result for image: {filename}\n\nThis is placeholder text that would be replaced by actual OCR output."

        blocks = [
            TextBlock(
                text=f"Mock OCR result for image: {filename}",
                confidence=0.95,
                bounding_box=BoundingBox(x=10, y=10, width=500, height=30),
            ),
            TextBlock(
                text="This is placeholder text that would be replaced by actual OCR output.",
                confidence=0.92,
                bounding_box=BoundingBox(x=10, y=50, width=500, height=30),
            ),
        ]

        return OCRResult(
            text=mock_text,
            blocks=blocks,
            confidence=0.93,
            metadata={
                "engine": "mock",
                "image_path": str(image_path),
            },
        )

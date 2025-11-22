import gc
import logging
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "stepfun-ai/GOT-OCR2_0"


@OCREngineRegistry.register
class GOTOCREngine(BaseOCREngine):
    """
    GOT-OCR 2.0 (General OCR Theory) engine.

    A general-purpose OCR model that works well with printed text
    in various languages and formats.
    """

    def __init__(self, model_path: Path):
        super().__init__(model_path)
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "got"

    async def load(self) -> None:
        """Load the GOT-OCR model with 4-bit quantization."""
        if self._loaded:
            return

        logger.info(f"Loading GOT-OCR model: {MODEL_ID}")

        try:
            # Configure 4-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )

            # Load model with 4-bit quantization
            self._model = AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

            self._model.eval()
            # Disable cache to avoid DynamicCache compatibility issues
            self._model.config.use_cache = False
            # Also set it on generation_config if available
            if hasattr(self._model, 'generation_config'):
                self._model.generation_config.use_cache = False
            self._loaded = True
            logger.info("GOT-OCR model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load GOT-OCR model: {e}")
            raise RuntimeError(f"Failed to load GOT-OCR model: {e}")

    async def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if not self._loaded:
            return

        logger.info("Unloading GOT-OCR model")

        try:
            if self._model is not None:
                del self._model
                self._model = None

            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self._loaded = False
            logger.info("GOT-OCR model unloaded")

        except Exception as e:
            logger.error(f"Error unloading GOT-OCR model: {e}")

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image using GOT-OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text.
        """
        self._ensure_loaded()

        logger.info(f"Processing image with GOT-OCR: {image_path}")

        try:
            # Use the model's built-in chat method for OCR
            # Mode can be 'ocr' (plain text) or 'format' (formatted)
            # Disable cache explicitly to avoid DynamicCache issues
            with torch.inference_mode():
                result = self._model.chat(
                    self._tokenizer,
                    str(image_path),
                    ocr_type="ocr",
                )

            # Clean up the result
            text = result.strip() if result else ""

            return OCRResult(
                text=text,
                confidence=0.0,  # GOT doesn't provide confidence scores
                metadata={
                    "engine": self.name,
                    "model_id": MODEL_ID,
                    "image_path": str(image_path),
                },
            )

        except Exception as e:
            logger.error(f"GOT-OCR processing failed: {e}")
            raise RuntimeError(f"OCR processing failed: {e}")

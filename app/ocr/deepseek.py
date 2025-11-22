import gc
import logging
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "unsloth/DeepSeek-OCR"

# DeepSeek OCR prompt
OCR_PROMPT = """Extract all text from this image accurately.
Rules:
- Extract text exactly as it appears
- Maintain paragraph structure
- Do not add any commentary
- If no text is found, respond with "No text found"
"""


@OCREngineRegistry.register
class DeepSeekOCREngine(BaseOCREngine):
    """
    DeepSeek-OCR engine.

    A general-purpose OCR model with good multilingual support.
    """

    def __init__(self, model_path: Path):
        super().__init__(model_path)
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "deepseek"

    async def load(self) -> None:
        """Load the DeepSeek-OCR model with 4-bit quantization."""
        if self._loaded:
            return

        logger.info(f"Loading DeepSeek-OCR model: {MODEL_ID}")

        try:
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )

            # Load model with 4-bit quantization
            self._model = AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                load_in_4bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            self._model.eval()
            self._loaded = True
            logger.info("DeepSeek-OCR model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load DeepSeek-OCR model: {e}")
            raise RuntimeError(f"Failed to load DeepSeek-OCR model: {e}")

    async def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if not self._loaded:
            return

        logger.info("Unloading DeepSeek-OCR model")

        try:
            if self._model is not None:
                del self._model
                self._model = None

            if self._processor is not None:
                del self._processor
                self._processor = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self._loaded = False
            logger.info("DeepSeek-OCR model unloaded")

        except Exception as e:
            logger.error(f"Error unloading DeepSeek-OCR model: {e}")

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image using DeepSeek-OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text.
        """
        self._ensure_loaded()

        logger.info(f"Processing image with DeepSeek-OCR: {image_path}")

        try:
            from PIL import Image

            # Load and prepare image
            image = Image.open(image_path).convert("RGB")

            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process inputs
            inputs = self._processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)

            # Generate
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                )

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return OCRResult(
                text=output_text.strip(),
                confidence=0.0,
                metadata={
                    "engine": self.name,
                    "model_id": MODEL_ID,
                    "image_path": str(image_path),
                },
            )

        except Exception as e:
            logger.error(f"DeepSeek-OCR processing failed: {e}")
            raise RuntimeError(f"OCR processing failed: {e}")

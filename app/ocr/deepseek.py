import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry
from app.ocr.processor import ImageProcessor

logger = logging.getLogger(__name__)


# Store reference to original get_imports before any patching
from transformers.dynamic_module_utils import get_imports as _original_get_imports


def _patched_get_imports(filename: str | os.PathLike) -> list[str]:
    """
    Workaround for HuggingFace's get_imports not respecting conditional imports.

    The original get_imports uses regex and flags conditionally-imported packages
    as hard requirements. This patch removes packages that are optional.
    """
    imports = _original_get_imports(filename)
    # Remove packages that are conditionally imported in DeepSeek model
    optional_packages = ["flash_attn", "addict", "einops", "easydict", "matplotlib"]
    for pkg in optional_packages:
        if pkg in imports:
            imports.remove(pkg)
    return imports

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
        self._image_processor: Optional[ImageProcessor] = None

    @property
    def name(self) -> str:
        return "deepseek"

    async def load(self) -> None:
        """Load the DeepSeek-OCR model with 4-bit quantization."""
        if self._loaded:
            return

        logger.info(f"Loading DeepSeek-OCR model: {MODEL_ID}")

        try:
            # Import inside try block to catch any import errors
            from transformers import AutoModel, AutoProcessor

            # Use monkey-patch to bypass HuggingFace's broken import check
            # that doesn't respect conditional imports
            with patch(
                "transformers.dynamic_module_utils.get_imports",
                _patched_get_imports,
            ):
                # Load processor
                self._processor = AutoProcessor.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                )

                # Load model with 4-bit quantization
                # Using AutoModel instead of AutoModelForVision2Seq because
                # the model has a custom config class not recognized by the latter
                self._model = AutoModel.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    load_in_4bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            self._model.eval()
            self._loaded = True

            # Initialize image processor for preprocessing and splitting
            self._image_processor = ImageProcessor(self.name)

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

    async def _process_single_image(self, pil_image: Image.Image) -> str:
        """Process a single PIL image and return text."""
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
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
            images=[pil_image],
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

        return output_text.strip()

    async def _process_chunk(self, chunk: np.ndarray) -> str:
        """Process a single preprocessed chunk and return text."""
        pil_image = Image.fromarray(chunk).convert("RGB")
        return await self._process_single_image(pil_image)

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image using DeepSeek-OCR.

        Uses preprocessing and content-aware splitting for large images.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text.
        """
        self._ensure_loaded()

        logger.info(f"Processing image with DeepSeek-OCR: {image_path}")

        try:
            # Use the image processor for preprocessing and splitting
            text, metadata = await self._image_processor.process_with_ocr(
                image_path,
                self._process_chunk,
                rtl=False,  # DeepSeek is for general text (LTR)
            )

            logger.info(
                f"Processing complete. Split: {metadata.get('was_split', False)}, "
                f"Method: {metadata.get('split_method', 'none')}"
            )

            # Save output to file
            output_file_path = self._save_output_file(image_path, text)

            return OCRResult(
                text=text,
                formatted_text=text,  # DeepSeek doesn't have separate formatting
                output_file=output_file_path,
                confidence=0.0,
                metadata={
                    "engine": self.name,
                    "model_id": MODEL_ID,
                    "image_path": str(image_path),
                    "was_split": metadata.get("was_split", False),
                    "split_method": metadata.get("split_method"),
                    "grid_shape": metadata.get("grid_shape"),
                    "preprocessing_applied": metadata.get("preprocessing_applied", []),
                },
            )

        except Exception as e:
            logger.error(f"DeepSeek-OCR processing failed: {e}")
            raise RuntimeError(f"OCR processing failed: {e}")

    def _save_output_file(self, image_path: Path, text: str) -> str:
        """Save OCR output to a text file."""
        output_dir = image_path.parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{image_path.stem}_ocr.txt"
        output_path = output_dir / output_filename

        output_path.write_text(text, encoding="utf-8")
        logger.info(f"Saved OCR output to: {output_path}")

        return str(output_path)

import gc
import logging
from pathlib import Path

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry
from app.ocr.utils.image_splitter import get_image_info, split_image_grid
from app.ocr.utils.ocr_processor import process_image_chunks, cleanup_temp_chunks

logger = logging.getLogger(__name__)

MODEL_ID = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"

# Arabic OCR prompt
ARABIC_OCR_PROMPT = """أنت خبير في التعرف الضوئي على النصوص العربية. استخرج النص العربي بدقة من الصورة.
قواعد:
- استخرج النص العربي فقط
- حافظ على تنسيق الفقرات
- لا تضف أي تعليقات
- إذا لم يوجد نص، أجب بـ "لا يوجد نص"
"""


@OCREngineRegistry.register
class QariOCREngine(BaseOCREngine):
    """
    Qari-OCR v0.3 engine specialized for Arabic text.

    Features:
    - Optimized for Arabic script recognition
    - Supports chunked processing for large images
    - RTL reading order support
    - Anti-hallucination settings
    """

    def __init__(self, model_path: Path):
        super().__init__(model_path)
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "qari"

    async def load(self) -> None:
        """Load the Qari-OCR model with 8-bit quantization."""
        if self._loaded:
            return

        logger.info(f"Loading Qari-OCR model: {MODEL_ID}")

        try:
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )

            # Load model with 8-bit quantization (better for Arabic accuracy)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                load_in_8bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            self._model.eval()
            self._loaded = True
            logger.info("Qari-OCR model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Qari-OCR model: {e}")
            raise RuntimeError(f"Failed to load Qari-OCR model: {e}")

    async def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if not self._loaded:
            return

        logger.info("Unloading Qari-OCR model")

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
            logger.info("Qari-OCR model unloaded")

        except Exception as e:
            logger.error(f"Error unloading Qari-OCR model: {e}")

    async def _process_single_image(self, image_path: Path) -> str:
        """Process a single image (or chunk) and return text."""
        from qwen_vl_utils import process_vision_info

        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": ARABIC_OCR_PROMPT},
                ],
            }
        ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        # Generate with anti-hallucination settings
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
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

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image using Qari-OCR.

        Automatically splits large images into chunks for better accuracy.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text.
        """
        self._ensure_loaded()

        logger.info(f"Processing image with Qari-OCR: {image_path}")

        try:
            # Check if image needs splitting
            image_info = get_image_info(image_path)

            if image_info.needs_splitting:
                logger.info(
                    f"Image is {image_info.megapixels:.1f}MP, splitting into "
                    f"{image_info.suggested_grid[0]}x{image_info.suggested_grid[1]} grid"
                )

                # Split image into chunks
                rows, cols = image_info.suggested_grid
                chunk_paths = split_image_grid(
                    image_path,
                    rows=rows,
                    cols=cols,
                    overlap=0.1,
                )

                try:
                    # Process chunks with RTL reading order for Arabic
                    text = await process_image_chunks(
                        chunk_paths,
                        self._process_single_image,
                        reading_order="rtl",
                        grid_shape=(rows, cols),
                    )
                finally:
                    # Cleanup temp chunks
                    if chunk_paths:
                        cleanup_temp_chunks(chunk_paths[0].parent)
            else:
                # Process single image
                text = await self._process_single_image(image_path)

            return OCRResult(
                text=text,
                confidence=0.0,
                metadata={
                    "engine": self.name,
                    "model_id": MODEL_ID,
                    "image_path": str(image_path),
                    "was_chunked": image_info.needs_splitting,
                    "grid_shape": image_info.suggested_grid if image_info.needs_splitting else None,
                },
            )

        except Exception as e:
            logger.error(f"Qari-OCR processing failed: {e}")
            raise RuntimeError(f"OCR processing failed: {e}")

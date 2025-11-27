import gc
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry
from app.ocr.processor import ImageProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"

# Arabic OCR prompt - simplified for better accuracy
ARABIC_OCR_PROMPT = """استخرج النص من الصورة. اكتب النص العربي فقط بدون أي علامات HTML أو رموز أو حروف إنجليزية. النص العربي فقط."""


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
        self._image_processor: Optional[ImageProcessor] = None

    @property
    def name(self) -> str:
        return "qari"

    async def load(self) -> None:
        """Load the Qari-OCR model at full precision (float16)."""
        if self._loaded:
            return

        logger.info(f"Loading Qari-OCR model (full precision float16): {MODEL_ID}")

        try:
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )

            # Load model at full precision (float16) for maximum accuracy
            # The 2B model should fit in 16GB VRAM with float16 (~4GB)
            # Using both torch_dtype and dtype to ensure compatibility
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            self._model.eval()
            self._loaded = True

            # Initialize image processor for preprocessing and splitting
            self._image_processor = ImageProcessor(self.name)

            # Print model info to verify precision (using print for visibility)
            param = next(self._model.parameters())
            print(f"[QARI] Model dtype: {param.dtype}", flush=True)
            print(f"[QARI] Model device: {param.device}", flush=True)
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1024**3
                print(f"[QARI] GPU memory allocated: {mem_gb:.2f} GB", flush=True)

            # Check if model has quantization config
            if hasattr(self._model.config, 'quantization_config'):
                print(f"[QARI] Quantization config: {self._model.config.quantization_config}", flush=True)
            else:
                print("[QARI] No quantization config - model is at full precision", flush=True)

            print("[QARI] Model loaded successfully", flush=True)
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

    async def _process_chunk(self, chunk: np.ndarray) -> str:
        """Process a single preprocessed chunk and return text."""
        tmp_path = None
        try:
            # Save chunk to temp file for processing
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                Image.fromarray(chunk).save(tmp_path)

            return await self._process_single_image(tmp_path)
        finally:
            # Cleanup temp file
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image using Qari-OCR.

        Uses content-aware splitting and preprocessing for optimal accuracy.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text.
        """
        self._ensure_loaded()

        logger.info(f"Processing image with Qari-OCR: {image_path}")

        try:
            # Use the new image processor for preprocessing and splitting
            text, metadata = await self._image_processor.process_with_ocr(
                image_path,
                self._process_chunk,
                rtl=True,  # Arabic is RTL
            )

            logger.info(
                f"Processing complete. Split: {metadata.get('was_split', False)}, "
                f"Method: {metadata.get('split_method', 'none')}"
            )

            # Save output to file
            output_file_path = self._save_output_file(image_path, text)

            return OCRResult(
                text=text,
                formatted_text=text,  # Arabic text doesn't need special formatting
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
            logger.error(f"Qari-OCR processing failed: {e}")
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

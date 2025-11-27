import gc
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry
from app.ocr.processor import ImageProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "stepfun-ai/GOT-OCR2_0"


def _patch_dynamic_cache():
    """
    Patch DynamicCache for compatibility with older GOT model code.

    The model's custom code expects older transformers cache API.
    """
    if hasattr(DynamicCache, '_patched'):
        return

    original_init = DynamicCache.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Add seen_tokens if missing
        if not hasattr(self, 'seen_tokens'):
            self.seen_tokens = 0

    # Add get_max_length method if missing
    if not hasattr(DynamicCache, 'get_max_length'):
        def get_max_length(self):
            """Return max length of cached keys."""
            if hasattr(self, 'key_cache') and self.key_cache:
                return self.key_cache[0].shape[2] if len(self.key_cache) > 0 else 0
            return 0
        DynamicCache.get_max_length = get_max_length

    # Add get_seq_length method if missing
    if not hasattr(DynamicCache, 'get_seq_length'):
        def get_seq_length(self, layer_idx: int = 0):
            """Return sequence length of cached keys at layer."""
            if hasattr(self, 'key_cache') and len(self.key_cache) > layer_idx:
                return self.key_cache[layer_idx].shape[2]
            return 0
        DynamicCache.get_seq_length = get_seq_length

    DynamicCache.__init__ = patched_init
    DynamicCache._patched = True


# Apply patch at module load time
_patch_dynamic_cache()


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
        self._image_processor: Optional[ImageProcessor] = None

    @property
    def name(self) -> str:
        return "got"

    async def load(self) -> None:
        """Load the GOT-OCR model with 4-bit quantization."""
        if self._loaded:
            return

        logger.info(f"Loading GOT-OCR model: {MODEL_ID} (4-bit NF4 quantization)")

        try:
            # Configure 4-bit quantization - GOT works best with this config
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

            # Load model with 8-bit quantization
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
            self._loaded = True

            # Initialize image processor for preprocessing and splitting
            self._image_processor = ImageProcessor(self.name)

            logger.info("GOT-OCR model loaded successfully (4-bit)")

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

    async def _process_chunk(self, chunk: np.ndarray) -> tuple[str, str]:
        """Process a single preprocessed chunk and return (plain_text, formatted_text)."""
        # DEBUG: Log chunk processing
        logger.debug(f"[GOT-OCR] Processing chunk shape: {chunk.shape}, dtype: {chunk.dtype}")

        tmp_path = None
        try:
            # Save chunk to temp file for processing
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                Image.fromarray(chunk).save(tmp_path)

            # DEBUG: Log temp file creation
            logger.debug(f"[GOT-OCR] Created temp file: {tmp_path}")

            with torch.inference_mode():
                plain_text = self._model.chat(
                    self._tokenizer,
                    str(tmp_path),
                    ocr_type="ocr",
                )
                formatted_text = self._model.chat(
                    self._tokenizer,
                    str(tmp_path),
                    ocr_type="format",
                )

            # DEBUG: Log results
            logger.debug(f"[GOT-OCR] Extracted plain text length: {len(plain_text) if plain_text else 0}")
            logger.debug(f"[GOT-OCR] Extracted formatted text length: {len(formatted_text) if formatted_text else 0}")

            return (
                plain_text.strip() if plain_text else "",
                formatted_text.strip() if formatted_text else "",
            )
        except Exception as e:
            # DEBUG: Log processing errors
            logger.error(f"[GOT-OCR] Error processing chunk: {e}")
            raise
        finally:
            # Ensure cleanup of temp file
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
                logger.debug(f"[GOT-OCR] Cleaned up temp file: {tmp_path}")

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image using GOT-OCR.

        Uses preprocessing and content-aware splitting for large images.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text, formatted text, and output file path.
        """
        self._ensure_loaded()

        logger.info(f"Processing image with GOT-OCR: {image_path}")

        try:
            # Process image through preprocessor
            processed = self._image_processor.process(image_path)

            if not processed.was_split:
                # Single chunk - process directly
                plain_text, formatted_text = await self._process_chunk(processed.chunks[0])
            else:
                # Multiple chunks - process each and merge
                plain_texts = []
                formatted_texts = []

                for chunk in processed.chunks:
                    pt, ft = await self._process_chunk(chunk)
                    plain_texts.append(pt)
                    formatted_texts.append(ft)

                plain_text = "\n\n".join(plain_texts)
                formatted_text = "\n\n".join(formatted_texts)

            logger.info(
                f"Processing complete. Split: {processed.was_split}, "
                f"Preprocessing: {processed.preprocessing_applied}"
            )

            # Save formatted output to file
            output_file_path = self._save_output_file(image_path, formatted_text)

            return OCRResult(
                text=plain_text,
                formatted_text=formatted_text,
                output_file=output_file_path,
                confidence=0.0,
                metadata={
                    "engine": self.name,
                    "model_id": MODEL_ID,
                    "image_path": str(image_path),
                    "was_split": processed.was_split,
                    "split_method": processed.split_result.split_method if processed.split_result else None,
                    "grid_shape": processed.grid_shape,
                    "preprocessing_applied": processed.preprocessing_applied,
                },
            )

        except Exception as e:
            logger.error(f"GOT-OCR processing failed: {e}")
            raise RuntimeError(f"OCR processing failed: {e}")

    def _save_output_file(self, image_path: Path, formatted_text: str) -> str:
        """Save formatted OCR output to a text file."""
        # Create output directory next to uploads
        output_dir = image_path.parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename based on image name
        output_filename = f"{image_path.stem}_ocr.txt"
        output_path = output_dir / output_filename

        # Write formatted text to file
        output_path.write_text(formatted_text, encoding="utf-8")
        logger.info(f"Saved OCR output to: {output_path}")

        return str(output_path)

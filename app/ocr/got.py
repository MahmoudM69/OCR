import gc
import logging
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry

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

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image using GOT-OCR.

        Performs both plain OCR and formatted OCR extraction,
        saving the formatted output to a text file.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text, formatted text, and output file path.
        """
        self._ensure_loaded()

        logger.info(f"Processing image with GOT-OCR: {image_path}")

        try:
            with torch.inference_mode():
                # Get plain text OCR result
                plain_text = self._model.chat(
                    self._tokenizer,
                    str(image_path),
                    ocr_type="ocr",
                )

                # Get formatted OCR result (preserves structure/layout)
                formatted_text = self._model.chat(
                    self._tokenizer,
                    str(image_path),
                    ocr_type="format",
                )

            # Clean up results
            plain_text = plain_text.strip() if plain_text else ""
            formatted_text = formatted_text.strip() if formatted_text else ""

            # Save formatted output to file
            output_file_path = self._save_output_file(image_path, formatted_text)

            return OCRResult(
                text=plain_text,
                formatted_text=formatted_text,
                output_file=output_file_path,
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

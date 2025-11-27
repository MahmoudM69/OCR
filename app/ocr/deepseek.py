import gc
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from app.ocr.base import BaseOCREngine, OCRResult
from app.ocr.registry import OCREngineRegistry

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

# DeepSeek OCR prompt - simple OCR mode for text extraction
OCR_PROMPT = "<image>\nConvert this document to markdown. Extract all visible text accurately."


@OCREngineRegistry.register
class DeepSeekOCREngine(BaseOCREngine):
    """
    DeepSeek-OCR engine.

    A general-purpose OCR model with good multilingual support.
    """

    def __init__(self, model_path: Path):
        super().__init__(model_path)
        self._model = None
        self._tokenizer = None

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
            from transformers import AutoModel, AutoTokenizer

            # Use monkey-patch to bypass HuggingFace's broken import check
            # that doesn't respect conditional imports
            with patch(
                "transformers.dynamic_module_utils.get_imports",
                _patched_get_imports,
            ):
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                )

                # Load model with 4-bit quantization
                self._model = AutoModel.from_pretrained(
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

            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self._loaded = False
            logger.info("DeepSeek-OCR model unloaded")

        except Exception as e:
            logger.error(f"Error unloading DeepSeek-OCR model: {e}")

    def _process_image_file(self, image_path: str, output_dir: Path) -> str:
        """
        Process an image file using the model's built-in infer method.

        Args:
            image_path: Path to the image file.
            output_dir: Directory for model outputs.

        Returns:
            Extracted text from the image.
        """
        # Validate image path
        if not image_path:
            raise ValueError("Image path is empty")

        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Processing image file: {image_path}")

        # Create output directory for the model
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use the model's built-in infer method
        # The model prints text to low-level stdout (file descriptor 1)
        # We need to capture at the FD level, not just sys.stdout

        # Create a temporary file to capture stdout
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Flush Python's stdout buffer
            sys.stdout.flush()

            # Save the original stdout file descriptor
            stdout_fd = sys.stdout.fileno()
            saved_stdout_fd = os.dup(stdout_fd)

            # Open temp file and redirect stdout to it
            with open(tmp_path, 'w') as tmp_file:
                os.dup2(tmp_file.fileno(), stdout_fd)

                try:
                    # Call the model's infer method
                    self._model.infer(
                        self._tokenizer,
                        prompt=OCR_PROMPT,
                        image_file=str(image_file),
                        output_path=str(output_dir),
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=False,
                    )
                finally:
                    # Restore original stdout
                    sys.stdout.flush()
                    os.dup2(saved_stdout_fd, stdout_fd)
                    os.close(saved_stdout_fd)

            # Read captured output
            with open(tmp_path, 'r', encoding='utf-8') as f:
                captured_output = f.read()

            logger.info(f"Captured {len(captured_output)} chars from stdout")

            # Parse the captured output to extract the OCR text
            text = self._parse_infer_output(captured_output)

            if text:
                logger.info(f"Extracted text preview: {text[:200]}...")
            else:
                logger.warning(f"No text extracted. Raw output preview: {captured_output[:500]}")

            return text.strip() if text else ""

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _parse_infer_output(self, stdout_text: str) -> str:
        """
        Parse the stdout from infer() to extract the actual OCR text.

        The model prints debug info like:
        =====================
        BASE:  torch.Size([...])
        PATCHES:  torch.Size([...])
        =====================
        <actual text here>
        ===============save results:===============
        ...

        Args:
            stdout_text: Captured stdout from infer().

        Returns:
            Extracted OCR text.
        """
        lines = stdout_text.split("\n")
        text_lines = []
        marker_count = 0
        in_text_section = False

        for line in lines:
            # Count the separator markers
            if "=====================" in line and "save" not in line.lower():
                marker_count += 1
                # After the second marker, we're in the text section
                if marker_count >= 2:
                    in_text_section = True
                continue

            # Stop at save results section
            if "save results" in line.lower() or "===============" in line:
                break

            # Skip known debug patterns before text section
            if not in_text_section:
                continue

            # Skip debug lines even in text section
            if line.strip().startswith("BASE:") or line.strip().startswith("PATCHES:"):
                continue

            # Collect text lines (including empty lines for paragraph structure)
            if in_text_section:
                text_lines.append(line)

        return "\n".join(text_lines).strip()

    async def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from an image using DeepSeek-OCR.

        Uses the model's built-in image handling which supports large images
        natively without external splitting. Utilizes full GPU memory.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text.
        """
        self._ensure_loaded()

        logger.info(f"Processing image with DeepSeek-OCR: {image_path}")

        try:
            # Output directory for model's internal use
            output_dir = image_path.parent.parent / "outputs"

            # Use the model's built-in infer method directly
            # The model handles large images internally with its own cropping logic
            text = self._process_image_file(str(image_path), output_dir)

            logger.info(f"DeepSeek-OCR processing complete for {image_path}")

            # Save output to file
            output_file_path = self._save_output_file(image_path, text)

            return OCRResult(
                text=text,
                formatted_text=text,  # DeepSeek outputs markdown
                output_file=output_file_path,
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

    def _save_output_file(self, image_path: Path, text: str) -> str:
        """Save OCR output to a text file."""
        output_dir = image_path.parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{image_path.stem}_ocr.txt"
        output_path = output_dir / output_filename

        output_path.write_text(text, encoding="utf-8")
        logger.info(f"Saved OCR output to: {output_path}")

        return str(output_path)

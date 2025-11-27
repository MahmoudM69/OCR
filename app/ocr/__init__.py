# Import OCR engines to register them
from app.ocr.got import GOTOCREngine
from app.ocr.qari import QariOCREngine
from app.ocr.deepseek import DeepSeekOCREngine

# Import new preprocessing and splitting modules
from app.ocr.processor import ImageProcessor, create_processor
from app.ocr import preprocessing
from app.ocr import splitting

__all__ = [
    # OCR Engines
    "GOTOCREngine",
    "QariOCREngine",
    "DeepSeekOCREngine",
    # Image Processing
    "ImageProcessor",
    "create_processor",
    # Submodules
    "preprocessing",
    "splitting",
]

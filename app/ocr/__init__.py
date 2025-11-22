# Import OCR engines to register them
from app.ocr.got import GOTOCREngine
from app.ocr.qari import QariOCREngine
from app.ocr.deepseek import DeepSeekOCREngine

__all__ = [
    "GOTOCREngine",
    "QariOCREngine",
    "DeepSeekOCREngine",
]

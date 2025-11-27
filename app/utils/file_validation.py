"""
File validation utilities for secure file type verification.

This module provides multi-layer validation to prevent content-type spoofing:
1. Magic byte detection (file signature)
2. Library-specific validation (PIL for images, PyMuPDF for PDFs)
3. Structure validation

All validation is defensive - files are considered invalid unless proven otherwise.
"""

import io
import logging
from enum import Enum
from pathlib import Path
from typing import BinaryIO

from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class FileType(str, Enum):
    """Supported file types for OCR processing."""

    IMAGE = "image"
    PDF = "pdf"
    UNKNOWN = "unknown"


# Magic byte signatures for file type detection
MAGIC_BYTES = {
    # Image formats
    b"\xff\xd8\xff": FileType.IMAGE,  # JPEG
    b"\x89PNG\r\n\x1a\n": FileType.IMAGE,  # PNG
    b"GIF87a": FileType.IMAGE,  # GIF87a
    b"GIF89a": FileType.IMAGE,  # GIF89a
    b"BM": FileType.IMAGE,  # BMP
    b"II*\x00": FileType.IMAGE,  # TIFF (little-endian)
    b"MM\x00*": FileType.IMAGE,  # TIFF (big-endian)
    b"RIFF": FileType.IMAGE,  # WebP (needs further validation)
    # PDF format
    b"%PDF-": FileType.PDF,  # PDF
}


class ValidationError(Exception):
    """Raised when file validation fails."""

    pass


def detect_file_type_from_bytes(file_content: bytes) -> FileType:
    """
    Detect file type from magic bytes (file signature).

    Args:
        file_content: First few bytes of the file.

    Returns:
        Detected file type.
    """
    # Check each magic byte signature
    for signature, file_type in MAGIC_BYTES.items():
        if file_content.startswith(signature):
            # Special case: RIFF could be WebP or other formats
            if signature == b"RIFF":
                if len(file_content) >= 12 and file_content[8:12] == b"WEBP":
                    return FileType.IMAGE
                else:
                    continue  # Not WebP, might be something else
            return file_type

    return FileType.UNKNOWN


def validate_image_with_pil(file_obj: BinaryIO) -> bool:
    """
    Validate that a file is a valid image using PIL.

    Args:
        file_obj: File object to validate (will be seeked to start).

    Returns:
        True if valid image, False otherwise.
    """
    try:
        file_obj.seek(0)
        img = Image.open(file_obj)
        img.verify()  # Verify it's a valid image
        file_obj.seek(0)  # Reset for future reads
        return True
    except Exception as e:
        logger.debug(f"PIL validation failed: {e}")
        return False


def validate_pdf_with_pymupdf(file_obj: BinaryIO) -> bool:
    """
    Validate that a file is a valid PDF using PyMuPDF.

    Args:
        file_obj: File object to validate (will be seeked to start).

    Returns:
        True if valid PDF, False otherwise.
    """
    try:
        file_obj.seek(0)
        file_content = file_obj.read()
        file_obj.seek(0)  # Reset for future reads

        # Try to open as PDF
        doc = fitz.open(stream=file_content, filetype="pdf")

        # Validate it has at least 1 page
        if doc.page_count < 1:
            logger.warning("PDF has no pages")
            doc.close()
            return False

        # Validate first page can be accessed
        _ = doc[0]
        doc.close()
        return True

    except Exception as e:
        logger.debug(f"PyMuPDF validation failed: {e}")
        return False


def validate_file(
    file_obj: BinaryIO,
    expected_type: FileType,
    max_size_mb: int = 50,
) -> None:
    """
    Perform comprehensive file validation with multi-layer checks.

    Args:
        file_obj: File object to validate (should be at start).
        expected_type: Expected file type (IMAGE or PDF).
        max_size_mb: Maximum file size in MB.

    Raises:
        ValidationError: If file fails any validation check.
    """
    # Layer 1: Size check
    file_obj.seek(0, 2)  # Seek to end
    file_size = file_obj.tell()
    file_obj.seek(0)  # Reset

    if file_size == 0:
        raise ValidationError("File is empty")

    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise ValidationError(
            f"File too large: {file_size / 1024 / 1024:.1f}MB "
            f"(max {max_size_mb}MB)"
        )

    # Layer 2: Magic bytes detection
    header = file_obj.read(32)  # Read first 32 bytes
    file_obj.seek(0)  # Reset

    detected_type = detect_file_type_from_bytes(header)

    if detected_type == FileType.UNKNOWN:
        raise ValidationError("Unknown or unsupported file type")

    if detected_type != expected_type:
        raise ValidationError(
            f"File type mismatch: expected {expected_type}, "
            f"detected {detected_type}"
        )

    # Layer 3: Library-specific validation
    if expected_type == FileType.IMAGE:
        if not validate_image_with_pil(file_obj):
            raise ValidationError("File is not a valid image")

    elif expected_type == FileType.PDF:
        if not validate_pdf_with_pymupdf(file_obj):
            raise ValidationError("File is not a valid PDF")

    logger.info(
        f"File validated successfully: {expected_type}, "
        f"{file_size / 1024:.1f}KB"
    )


def validate_filename(filename: str) -> str:
    """
    Sanitize and validate filename to prevent path traversal attacks.

    Args:
        filename: Original filename from upload.

    Returns:
        Sanitized filename safe for filesystem operations.

    Raises:
        ValidationError: If filename is invalid or malicious.
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")

    # Remove any path components (prevent directory traversal)
    safe_filename = Path(filename).name

    if not safe_filename:
        raise ValidationError("Invalid filename")

    # Check for suspicious patterns
    if ".." in safe_filename or safe_filename.startswith("."):
        raise ValidationError("Invalid filename pattern")

    # Limit length
    if len(safe_filename) > 255:
        raise ValidationError("Filename too long (max 255 characters)")

    return safe_filename

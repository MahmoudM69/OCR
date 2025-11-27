"""
PDF processing service for converting PDFs to images.

This service handles:
- PDF to image conversion (page-by-page)
- Temporary file management for converted pages
- Page metadata extraction
"""

import io
import logging
from pathlib import Path
from typing import BinaryIO

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""

    pass


class PDFPage:
    """Represents a single page from a PDF with its image data."""

    def __init__(
        self,
        page_number: int,
        total_pages: int,
        image_path: Path,
        width: int,
        height: int,
    ):
        """
        Initialize a PDF page.

        Args:
            page_number: Page number (1-indexed).
            total_pages: Total number of pages in the PDF.
            image_path: Path to the saved page image.
            width: Page width in pixels.
            height: Page height in pixels.
        """
        self.page_number = page_number
        self.total_pages = total_pages
        self.image_path = image_path
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return (
            f"PDFPage(page={self.page_number}/{self.total_pages}, "
            f"path={self.image_path})"
        )


def pdf_to_images(
    pdf_file: BinaryIO,
    output_dir: Path,
    job_id: str,
    dpi: int = 300,
) -> list[PDFPage]:
    """
    Convert all pages of a PDF to images.

    Args:
        pdf_file: PDF file object (should be at start).
        output_dir: Directory to save page images.
        job_id: Unique job ID for naming files.
        dpi: Resolution for rendering PDF pages (default: 300 DPI).

    Returns:
        List of PDFPage objects with metadata and image paths.

    Raises:
        PDFProcessingError: If PDF conversion fails.
    """
    pages: list[PDFPage] = []
    doc = None

    try:
        # Read PDF content
        pdf_file.seek(0)
        pdf_content = pdf_file.read()
        pdf_file.seek(0)

        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")

        if doc.page_count == 0:
            raise PDFProcessingError("PDF has no pages")

        logger.info(
            f"Processing PDF with {doc.page_count} pages at {dpi} DPI"
        )

        # Process each page
        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Render page to pixmap at specified DPI
            # zoom = dpi / 72 (72 is default DPI)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Save page image
            page_filename = f"{job_id}_page_{page_num + 1:03d}.png"
            page_path = output_dir / page_filename

            img.save(page_path, "PNG")

            # Create page metadata
            pdf_page = PDFPage(
                page_number=page_num + 1,
                total_pages=doc.page_count,
                image_path=page_path,
                width=img.width,
                height=img.height,
            )

            pages.append(pdf_page)

            logger.debug(
                f"Converted page {page_num + 1}/{doc.page_count}: "
                f"{img.width}x{img.height}px"
            )

        doc.close()

        logger.info(
            f"Successfully converted {len(pages)} pages from PDF"
        )

        return pages

    except fitz.FitzError as e:
        # Clean up any pages that were already created
        _cleanup_partial_pages(pages)
        if doc:
            doc.close()
        raise PDFProcessingError(f"PyMuPDF error: {e}") from e
    except PDFProcessingError:
        # Clean up any pages that were already created
        _cleanup_partial_pages(pages)
        if doc:
            doc.close()
        raise
    except Exception as e:
        # Clean up any pages that were already created
        _cleanup_partial_pages(pages)
        if doc:
            doc.close()
        raise PDFProcessingError(
            f"Failed to convert PDF to images: {e}"
        ) from e


def _cleanup_partial_pages(pages: list[PDFPage]) -> None:
    """
    Clean up page image files that were created before an error.

    Args:
        pages: List of PDFPage objects with image paths to delete.
    """
    for page in pages:
        try:
            page.image_path.unlink(missing_ok=True)
            logger.debug(f"Cleaned up partial page: {page.image_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up page {page.image_path}: {e}")


def get_pdf_metadata(pdf_file: BinaryIO) -> dict:
    """
    Extract metadata from a PDF file.

    Args:
        pdf_file: PDF file object (should be at start).

    Returns:
        Dictionary with PDF metadata (title, author, pages, etc.).
    """
    try:
        pdf_file.seek(0)
        pdf_content = pdf_file.read()
        pdf_file.seek(0)

        doc = fitz.open(stream=pdf_content, filetype="pdf")

        metadata = {
            "page_count": doc.page_count,
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "format": doc.metadata.get("format", ""),
        }

        doc.close()
        return metadata

    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata: {e}")
        return {"page_count": 0}

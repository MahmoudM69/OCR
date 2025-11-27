"""
Result merger for combining OCR results from multiple chunks.

Handles RTL/LTR text ordering and deduplication of overlapping regions.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional
import re

from .base import SplitResult, ImageChunk


# Arabic diacritical marks (tashkeel) - used for normalization
ARABIC_DIACRITICS = (
    '\u064B'  # FATHATAN
    '\u064C'  # DAMMATAN
    '\u064D'  # KASRATAN
    '\u064E'  # FATHA
    '\u064F'  # DAMMA
    '\u0650'  # KASRA
    '\u0651'  # SHADDA
    '\u0652'  # SUKUN
    '\u0653'  # MADDAH ABOVE
    '\u0654'  # HAMZA ABOVE
    '\u0655'  # HAMZA BELOW
    '\u0656'  # SUBSCRIPT ALEF
    '\u0657'  # INVERTED DAMMA
    '\u0658'  # MARK NOON GHUNNA
    '\u0659'  # ZWARAKAY
    '\u065A'  # VOWEL SIGN SMALL V ABOVE
    '\u065B'  # VOWEL SIGN INVERTED SMALL V ABOVE
    '\u065C'  # VOWEL SIGN DOT BELOW
    '\u065D'  # REVERSED DAMMA
    '\u065E'  # FATHA WITH TWO DOTS
    '\u065F'  # WAVY HAMZA BELOW
    '\u0670'  # SUPERSCRIPT ALEF
)


@dataclass
class ChunkResult:
    """OCR result for a single chunk."""

    chunk: ImageChunk
    """The chunk this result belongs to."""

    text: str
    """Extracted text."""

    confidence: float = 0.0
    """Confidence score."""


@dataclass
class MergeConfig:
    """Configuration for result merging."""

    rtl: bool = False
    """Right-to-left text direction."""

    similarity_threshold: float = 0.8
    """Threshold for considering text as duplicate (0.0 to 1.0)."""

    min_overlap_chars: int = 10
    """Minimum characters to check for overlap."""

    section_separator: str = "\n\n"
    """Separator between merged sections."""

    preserve_chunk_markers: bool = False
    """Whether to include chunk markers in output."""


class ResultMerger:
    """
    Merges OCR results from multiple chunks into a single coherent text.

    Handles:
    - RTL (Arabic, Hebrew) and LTR text ordering
    - Deduplication of overlapping text
    - Proper reading order based on grid position
    """

    def __init__(self, config: Optional[MergeConfig] = None):
        """Initialize the merger."""
        self.config = config or MergeConfig()

    def merge(
        self,
        results: list[ChunkResult],
        split_result: SplitResult,
    ) -> str:
        """
        Merge chunk results into a single text.

        Args:
            results: List of OCR results for each chunk.
            split_result: The split result containing chunk metadata.

        Returns:
            Merged text with deduplication applied.
        """
        if not results:
            return ""

        if len(results) == 1:
            return results[0].text.strip()

        # Sort chunks by reading order
        sorted_results = self._sort_by_reading_order(results, split_result)

        # Merge with deduplication
        merged = self._merge_with_deduplication(sorted_results)

        return merged.strip()

    def _sort_by_reading_order(
        self,
        results: list[ChunkResult],
        split_result: SplitResult,
    ) -> list[ChunkResult]:
        """
        Sort results by proper reading order.

        Args:
            results: Unsorted chunk results.
            split_result: Split metadata.

        Returns:
            Results sorted by reading order.
        """
        if self.config.rtl:
            # RTL: Read right to left, top to bottom
            # Sort by row first, then by column descending
            return sorted(
                results,
                key=lambda r: (r.chunk.row, -r.chunk.col)
            )
        else:
            # LTR: Read left to right, top to bottom
            # Sort by row first, then by column ascending
            return sorted(
                results,
                key=lambda r: (r.chunk.row, r.chunk.col)
            )

    def _merge_with_deduplication(self, results: list[ChunkResult]) -> str:
        """
        Merge results while removing duplicated text from overlapping regions.

        Args:
            results: Sorted chunk results.

        Returns:
            Merged text with duplicates removed.
        """
        if not results:
            return ""

        merged_parts = []
        prev_text = ""

        for i, result in enumerate(results):
            current_text = result.text.strip()

            if not current_text:
                continue

            if not prev_text:
                merged_parts.append(current_text)
                prev_text = current_text
                continue

            # Check for overlap with previous chunk
            if self._chunks_are_adjacent(result.chunk, results[i - 1].chunk):
                # Remove duplicated content
                deduped = self._remove_overlap(prev_text, current_text)
                if deduped:
                    merged_parts.append(deduped)
                    prev_text = current_text
            else:
                # Non-adjacent chunks, add separator
                merged_parts.append(current_text)
                prev_text = current_text

        return self.config.section_separator.join(merged_parts)

    def _chunks_are_adjacent(self, chunk1: ImageChunk, chunk2: ImageChunk) -> bool:
        """Check if two chunks are adjacent in the grid."""
        row_diff = abs(chunk1.row - chunk2.row)
        col_diff = abs(chunk1.col - chunk2.col)

        # Adjacent if same row and columns differ by 1
        # Or same column and rows differ by 1
        return (row_diff == 0 and col_diff == 1) or (row_diff == 1 and col_diff == 0)

    def _normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text for comparison by removing diacritics.

        Arabic diacritical marks (tashkeel) like fatha, damma, kasra, etc.
        are often inconsistently recognized by OCR. Removing them allows
        better matching of overlapping text.

        Args:
            text: Arabic text possibly containing diacritics.

        Returns:
            Text with Arabic diacritics removed.
        """
        return ''.join(c for c in text if c not in ARABIC_DIACRITICS)

    def _remove_overlap(self, prev_text: str, current_text: str) -> str:
        """
        Remove overlapping text between consecutive chunks.

        For RTL (Arabic) text, uses normalized comparison that ignores
        diacritical marks for better matching.

        Args:
            prev_text: Text from previous chunk.
            current_text: Text from current chunk.

        Returns:
            Current text with overlapping portion removed.
        """
        if not prev_text or not current_text:
            return current_text

        min_chars = self.config.min_overlap_chars

        # Get the end of previous text and start of current text
        prev_end = prev_text[-500:] if len(prev_text) > 500 else prev_text
        current_start = current_text[:500] if len(current_text) > 500 else current_text

        # For RTL (Arabic) text, use normalized comparison
        if self.config.rtl:
            # Normalize Arabic text (remove diacritics) for comparison
            prev_norm = self._normalize_arabic(prev_end)
            curr_norm = self._normalize_arabic(current_start)

            # Find overlap in normalized text
            overlap_length = self._find_overlap_length(prev_norm, curr_norm)

            if overlap_length >= min_chars:
                # Map normalized overlap back to original text
                # Count characters in original text until we match normalized length
                char_count = 0
                original_pos = 0
                for i, c in enumerate(current_start):
                    if c not in ARABIC_DIACRITICS:
                        char_count += 1
                    if char_count >= overlap_length:
                        original_pos = i + 1
                        break
                return current_text[original_pos:].strip()

            # Try fuzzy matching with normalized text
            fuzzy_overlap = self._find_fuzzy_overlap(prev_norm, curr_norm)
            if fuzzy_overlap > 0:
                # Map back to original position
                char_count = 0
                original_pos = 0
                for i, c in enumerate(current_start):
                    if c not in ARABIC_DIACRITICS:
                        char_count += 1
                    if char_count >= fuzzy_overlap:
                        original_pos = i + 1
                        break
                return current_text[original_pos:].strip()
        else:
            # Standard LTR text processing
            overlap_length = self._find_overlap_length(prev_end, current_start)

            if overlap_length >= min_chars:
                return current_text[overlap_length:].strip()

            fuzzy_overlap = self._find_fuzzy_overlap(prev_end, current_start)
            if fuzzy_overlap > 0:
                return current_text[fuzzy_overlap:].strip()

        return current_text

    def _find_overlap_length(self, text1: str, text2: str) -> int:
        """
        Find exact overlap between end of text1 and start of text2.

        Args:
            text1: First text (use end portion).
            text2: Second text (use start portion).

        Returns:
            Length of overlapping portion.
        """
        max_overlap = min(len(text1), len(text2))

        for length in range(max_overlap, self.config.min_overlap_chars - 1, -1):
            suffix = text1[-length:]
            prefix = text2[:length]

            if suffix == prefix:
                return length

        return 0

    def _find_fuzzy_overlap(self, text1: str, text2: str) -> int:
        """
        Find fuzzy overlap accounting for OCR errors.

        Only matches when texts are nearly identical (>= 0.95 similarity)
        to avoid false positives from similar but distinct content.

        Args:
            text1: End of previous text.
            text2: Start of current text.

        Returns:
            Approximate position where unique content starts in text2.
        """
        # Use a very strict threshold to avoid false positives
        # Only match actual duplicates with minor OCR errors
        strict_threshold = 0.95
        min_chars = self.config.min_overlap_chars

        # Split into words for comparison
        words1 = text1.split()
        words2 = text2.split()

        if len(words1) < 3 or len(words2) < 3:
            return 0

        # Only try smaller windows to avoid matching similar but distinct content
        max_window = min(len(words1), len(words2), 10)

        for window in range(max_window, 2, -1):
            suffix_words = words1[-window:]
            prefix_words = words2[:window]

            suffix_str = " ".join(suffix_words)
            prefix_str = " ".join(prefix_words)

            similarity = SequenceMatcher(None, suffix_str, prefix_str).ratio()

            # Only accept very high similarity (near-exact matches)
            if similarity >= strict_threshold:
                # Verify this looks like actual overlap, not just similar content
                # Check if at least 80% of words are exact matches
                exact_matches = sum(1 for w1, w2 in zip(suffix_words, prefix_words) if w1 == w2)
                if exact_matches / window >= 0.8:
                    overlap_text = " ".join(words2[:window])
                    return len(overlap_text)

        return 0

    def merge_formatted(
        self,
        results: list[ChunkResult],
        split_result: SplitResult,
    ) -> str:
        """
        Merge results with chunk position markers for debugging.

        Args:
            results: List of chunk results.
            split_result: Split metadata.

        Returns:
            Merged text with position markers.
        """
        if not results:
            return ""

        sorted_results = self._sort_by_reading_order(results, split_result)

        parts = []
        for result in sorted_results:
            chunk = result.chunk
            marker = f"[Chunk {chunk.index}: Row {chunk.row}, Col {chunk.col}]"
            text = result.text.strip()
            if text:
                parts.append(f"{marker}\n{text}")

        return "\n\n".join(parts)


def create_merger(rtl: bool = False, **kwargs) -> ResultMerger:
    """
    Factory function to create a configured merger.

    Args:
        rtl: Whether text is right-to-left.
        **kwargs: Additional MergeConfig options.

    Returns:
        Configured ResultMerger.
    """
    config = MergeConfig(rtl=rtl, **kwargs)
    return ResultMerger(config)

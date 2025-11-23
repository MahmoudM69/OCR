"""
Content-aware image splitting module for OCR.

This module provides intelligent image splitting that avoids cutting through text,
using projection profiles and connected component analysis.
"""

from .base import BaseSplitter, SplitResult, ImageChunk, SplitConfig
from .analyzer import SplitAnalyzer
from .projection import ProjectionSplitter
from .components import ComponentSplitter
from .grid import GridSplitter
from .merger import ResultMerger, ChunkResult, MergeConfig, create_merger
from .splitter import SmartSplitter, create_splitter

__all__ = [
    "BaseSplitter",
    "SplitResult",
    "ImageChunk",
    "SplitConfig",
    "SplitAnalyzer",
    "ProjectionSplitter",
    "ComponentSplitter",
    "GridSplitter",
    "ResultMerger",
    "ChunkResult",
    "MergeConfig",
    "create_merger",
    "SmartSplitter",
    "create_splitter",
]

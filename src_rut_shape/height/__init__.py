"""
Height calculation module for rut shape analysis.

This module provides functionality for:
- Image loading and preprocessing
- Coordinate processing and transformation
- Rut depth calculation
- File management operations
- Specialized processors for height-specific operations

Author: Refactored from original height.py
"""

from .image_loader import ImageLoader
from .coordinate_processor import CoordinateProcessor
from .rut_calculator import RutCalculator
from .file_manager import HeightFileManager
from . import processors

__all__ = [
    'ImageLoader',
    'CoordinateProcessor', 
    'RutCalculator',
    'HeightFileManager',
    'processors'
]
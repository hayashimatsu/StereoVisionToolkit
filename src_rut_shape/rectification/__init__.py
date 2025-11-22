"""
Rectification module for stereo vision processing.

This module contains specialized classes for stereo image rectification,
including the core rectification engine, matrix calculations, and file management.
"""

from .engine import StereoRectificationEngine
from .matrix_calculator import MatrixCalculator
from .file_manager import RectificationFileManager

__all__ = [
    'StereoRectificationEngine',
    'MatrixCalculator', 
    'RectificationFileManager'
]
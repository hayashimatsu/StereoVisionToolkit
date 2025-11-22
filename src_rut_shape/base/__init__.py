"""
Base classes for stereo vision processing modules.

This package provides unified base classes for file management and processing
operations across rectification, disparity, and height processing modules.
"""

from .file_manager import BaseFileManager
from .processor import BaseProcessor

__all__ = ['BaseFileManager', 'BaseProcessor']
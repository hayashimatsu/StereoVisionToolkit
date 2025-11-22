"""
Specialized processors for height calculation module.

This package contains processors that are specifically designed for
height calculation operations, providing focused functionality for
rut shape processing and coordinate selection.
"""

from .rut_shape_processor import RutShapeProcessor
from .coordinate_selector import CoordinateSelector

__all__ = [
    'RutShapeProcessor',
    'CoordinateSelector'
]
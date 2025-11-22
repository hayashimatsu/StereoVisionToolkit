"""
Unified data structure for stereo rectification results

This module defines the RectificationResult dataclass that serves as the
unified data transfer object between calculation and visualization steps,
eliminating the need for intermediate JSON files.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class RectificationResult:
    """
    Unified data structure containing all stereo rectification results
    
    This class replaces the intermediate JSON file communication pattern
    with in-memory data transfer between components.
    """
    
    # Point data
    left_points_original: List[Dict[str, Any]]
    right_points_original: List[Dict[str, Any]]
    left_points_rectified: List[Dict[str, Any]]
    right_points_rectified: List[Dict[str, Any]]
    
    # Verification metrics
    mean_y_difference: float
    max_y_difference: float
    std_y_difference: float
    rms_y_difference: float
    mean_disparity: float
    min_disparity: float
    max_disparity: float
    num_points_verified: int
    
    # Rectification matrices
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    
    # Metadata
    phase: str
    improvements: List[str]
    
    # Optional debug information
    debug_info: Optional[Dict[str, Any]] = None
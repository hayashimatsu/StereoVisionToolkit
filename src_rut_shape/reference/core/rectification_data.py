"""
Unified data structure for stereo rectification results

This module defines the RectificationResult dataclass that serves as the
unified data transfer object between calculation and visualization steps.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class RectificationResult:
    """
    Unified data structure containing all stereo rectification results
    
    This class consolidates both numerical calculations and image processing
    results for in-memory data transfer between components.
    """
    
    # Original input images
    left_original_image: np.ndarray
    right_original_image: np.ndarray
    
    # Rectified output images
    rectified_left_image: np.ndarray
    rectified_right_image: np.ndarray
    
    # Point data
    left_points_original: List[Dict[str, Any]]
    right_points_original: List[Dict[str, Any]]
    left_points_rectified: List[Dict[str, Any]]
    right_points_rectified: List[Dict[str, Any]]
    
    # Rectification matrices
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    
    # ROI information
    roi1: tuple
    roi2: tuple
    
    # Verification metrics
    mean_y_difference: float
    max_y_difference: float
    std_y_difference: float
    rms_y_difference: float
    mean_disparity: float
    min_disparity: float
    max_disparity: float
    num_points_verified: int
    
    # Configuration parameters used
    alpha: float
    flags: int
    
    # Metadata
    phase: str
    improvements: List[str]
    
    # Optional debug information
    debug_info: Optional[Dict[str, Any]] = None
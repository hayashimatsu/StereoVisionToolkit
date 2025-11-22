"""
Coordinate transformation utilities for stereo vision processing.

This module provides common coordinate transformation functions used across
the rut shape analysis system, including:
- Image to world coordinate conversions
- Coordinate scaling and translation operations
- Optical axis calculations
- Matrix transformations

Author: Extracted from height module refactoring
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import logging


logger = logging.getLogger(__name__)


def scale_and_translate_coordinates(data: np.ndarray, target_width: float) -> Tuple[np.ndarray, dict]:
    """
    Scale x-coordinates to match target_width and translate to start at 0.
    
    Args:
        data: Coordinate data with shape (n, 2) where columns are [x, y]
        target_width: Target width in meters
    
    Returns:
        Tuple of (scaled_data, scaling_metadata)
    """
    # Extract coordinates
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    
    # Calculate scaling parameters
    current_width = np.max(x_coords) - np.min(x_coords)
    scale_factor = target_width / current_width
    
    # Apply scaling and translation
    x_scaled = x_coords * scale_factor
    x_translated = x_scaled - np.min(x_scaled)
    
    # Combine results
    scaled_data = np.column_stack((x_translated, y_coords))
    
    # Create metadata
    scaling_metadata = {
        "original_width": float(current_width),
        "target_width": float(target_width),
        "scale_factor": float(scale_factor),
        "translation_offset": float(np.min(x_scaled)),
        "original_x_range": [float(np.min(x_coords)), float(np.max(x_coords))],
        "scaled_x_range": [float(np.min(x_translated)), float(np.max(x_translated))],
        "version": "width_scaling_v1"
    }
    
    logger.info(f"Applied width scaling: original={current_width:.3f}m, "
                f"target={target_width:.3f}m, scale_factor={scale_factor:.3f}")
    
    return scaled_data, scaling_metadata


def calculate_xy_from_optical_axes(rut_rectified: np.ndarray, 
                                  optical_axes: List[np.ndarray], 
                                  disparity: np.ndarray, 
                                  baseline_translation: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate X and Y coordinates using optical axis method.
    
    Uses the formulation:
    X_n = Tx * || (x_refX_n, y_refX_n) - (x_rect_rut_n, y_rect_rut_n) || / d_n
    Y_n = Tx * || (x_refY_n, y_refY_n) - (x_rect_rut_n, y_rect_rut_n) || / d_n
    
    Args:
        rut_rectified: Rectified rut coordinates with shape (n, 2)
        optical_axes: List of [X_axis, Y_axis] optical reference points
        disparity: Disparity values with shape (n,)
        baseline_translation: Baseline translation parameter (Tx)
        
    Returns:
        Tuple of (X_coordinates, Y_coordinates)
    """
    X = None
    Y = None
    
    for i, rut_ref_opticalAxis in enumerate(optical_axes):
        ref_axes = np.asarray(rut_ref_opticalAxis, dtype=np.float64)
        if ref_axes.ndim > 1 and ref_axes.shape[1] > 2:
            ref_axes = ref_axes[:, :2]
        
        # Ensure all arrays have the same length
        n = min(len(rut_rectified), len(ref_axes), len(disparity))
        if n == 0:
            raise ValueError("No points available to compute coordinates")
        
        rr = rut_rectified[:n, :2]
        ra = ref_axes[:n, :2]
        d = disparity[:n]
        
        # Calculate Euclidean distance between corresponding points
        diff = rr - ra
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        
        # Calculate coordinates with divide-by-zero protection
        with np.errstate(divide='ignore', invalid='ignore'):
            coords = np.where(d != 0, baseline_translation * dist / d, 0.0)
            
            if i == 0:  # X coordinates
                X = coords.copy()
                # Apply sign correction for X coordinates
                idx_min = int(np.argmin(X))
                if idx_min > 0:
                    X[:idx_min] *= -1.0
            else:  # Y coordinates
                Y = coords.copy()
    
    return X, Y


def extract_xy_coordinates(world_coordinates: np.ndarray) -> np.ndarray:
    """
    Extract X,Y coordinates from 3D world coordinates.
    
    Args:
        world_coordinates: 3D coordinates with shape (n, 3) where columns are [x, y, z]
        
    Returns:
        2D coordinates with shape (n, 2) where columns are [x, y]
    """
    if world_coordinates.shape[1] < 2:
        raise ValueError("World coordinates must have at least 2 columns")
    
    return world_coordinates[:, :2]


def convert_image_to_world_coordinates(world_coord_array: np.ndarray,
                                     rectified_image: np.ndarray,
                                     disparity_image: np.ndarray,
                                     point_coordinates: List[List[float]]) -> np.ndarray:
    """
    Convert image coordinates to world coordinates using precomputed world coordinate array.
    
    Args:
        world_coord_array: Precomputed world coordinates with shape (height, width, 3)
        rectified_image: Rectified image for dimension reference
        disparity_image: Disparity image for validation
        point_coordinates: List of [x, y] pixel coordinates
        
    Returns:
        World coordinates for the specified points with shape (n, 3)
    """
    world_coords = []
    
    for point in point_coordinates:
        x, y = int(round(point[0])), int(round(point[1]))
        
        # Boundary check
        if (0 <= y < world_coord_array.shape[0] and 
            0 <= x < world_coord_array.shape[1]):
            world_coord = world_coord_array[y, x]
            world_coords.append(world_coord)
        else:
            logger.warning(f"Point ({x}, {y}) is outside world coordinate bounds")
            # Use NaN for out-of-bounds points
            world_coords.append([np.nan, np.nan, np.nan])
    
    return np.array(world_coords)


def apply_rotation_matrix(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply rotation matrix to 2D or 3D points.
    
    Args:
        points: Points to rotate with shape (n, 2) or (n, 3)
        rotation_matrix: Rotation matrix with shape (2, 2) or (3, 3)
        
    Returns:
        Rotated points with same shape as input
    """
    if points.shape[1] != rotation_matrix.shape[0]:
        raise ValueError(f"Point dimension {points.shape[1]} doesn't match "
                        f"rotation matrix dimension {rotation_matrix.shape[0]}")
    
    # Apply rotation: points_rotated = points @ rotation_matrix.T
    return points @ rotation_matrix.T


def calculate_transformation_parameters(Q_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Calculate transformation parameters from Q matrix.
    
    Args:
        Q_matrix: 4x4 Q matrix from stereo rectification
        
    Returns:
        Tuple of (baseline_translation, focal_length)
    """
    if Q_matrix.shape != (4, 4):
        raise ValueError("Q matrix must be 4x4")
    
    baseline_translation = 1 / Q_matrix[3, 2]  # Tx = 1 / Q[3,2]
    focal_length = Q_matrix[2, 3]              # fx = Q[2,3]
    
    return baseline_translation, focal_length


def normalize_coordinates(coordinates: np.ndarray, 
                         center: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize coordinates to zero mean.
    
    Args:
        coordinates: Input coordinates with shape (n, d)
        center: Optional center point. If None, uses mean of coordinates
        
    Returns:
        Tuple of (normalized_coordinates, center_used)
    """
    if center is None:
        center = np.mean(coordinates, axis=0)
    
    normalized = coordinates - center
    
    return normalized, center


def denormalize_coordinates(normalized_coordinates: np.ndarray, 
                           center: np.ndarray) -> np.ndarray:
    """
    Denormalize coordinates by adding back the center.
    
    Args:
        normalized_coordinates: Normalized coordinates with shape (n, d)
        center: Center point used for normalization
        
    Returns:
        Denormalized coordinates
    """
    return normalized_coordinates + center
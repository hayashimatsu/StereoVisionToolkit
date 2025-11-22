"""
Point processing utilities for stereo vision applications.

This module provides functions for point coordinate transformations, interpolation,
and optical center calculations that can be used across multiple stereo vision modules.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import json

from utils.logger_config import get_logger

logger = get_logger(__name__)


class PointProcessor:
    """Handles point coordinate transformations and interpolation for stereo vision."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @staticmethod
    def interpolate_points_between_coordinates(
        start_point: Tuple[float, float], 
        end_point: Tuple[float, float], 
        num_points: int = 10
    ) -> np.ndarray:
        """
        Generate interpolated points between two coordinates.
        
        Args:
            start_point: Starting coordinate (x, y)
            end_point: Ending coordinate (x, y)
            num_points: Minimum number of points to generate
            
        Returns:
            np.ndarray: Array of interpolated points with shape (n, 2)
        """
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Calculate number of points based on distance (ensure reasonable density)
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        actual_num_points = max(int(round(distance)), num_points)
        
        # Generate interpolated coordinates
        x_coords = np.linspace(x1, x2, actual_num_points)
        y_coords = np.linspace(y1, y2, actual_num_points)
        
        # Combine into array of [x, y] coordinates
        interpolated_points = np.column_stack((x_coords, y_coords))
        
        logger.info(f"Generated {actual_num_points} interpolated points between "
                   f"({x1:.1f}, {y1:.1f}) and ({x2:.1f}, {y2:.1f})")
        
        return interpolated_points
    
    @staticmethod
    def apply_resize_scaling_to_points(
        points: List[List[float]], 
        scale_factor: Tuple[float, float]
    ) -> List[List[float]]:
        """
        Apply resize scaling to a list of points.
        
        Args:
            points: List of [x, y] coordinate pairs
            scale_factor: (scale_x, scale_y) scaling factors
            
        Returns:
            List[List[float]]: Scaled points
        """
        scale_x, scale_y = scale_factor
        
        if scale_x == 1.0 and scale_y == 1.0:
            return points
            
        scaled_points = [[point[0] * scale_x, point[1] * scale_y] for point in points]
        
        logger.info(f"Applied resize scaling to {len(points)} points: "
                   f"scale=({scale_x:.3f}, {scale_y:.3f})")
        
        return scaled_points
    
    @staticmethod
    def round_points_to_integers(
        points: List[List[float]], 
        method: str = 'ceil'
    ) -> List[List[int]]:
        """
        Round point coordinates to integers.
        
        Args:
            points: List of [x, y] coordinate pairs
            method: Rounding method ('ceil', 'floor', 'round')
            
        Returns:
            List[List[int]]: Rounded integer points
        """
        if method == 'ceil':
            rounded_points = [[int(np.ceil(coord[0])), int(np.ceil(coord[1]))] 
                            for coord in points]
        elif method == 'floor':
            rounded_points = [[int(np.floor(coord[0])), int(np.floor(coord[1]))] 
                            for coord in points]
        elif method == 'round':
            rounded_points = [[int(round(coord[0])), int(round(coord[1]))] 
                            for coord in points]
        else:
            raise ValueError(f"Unknown rounding method: {method}")
            
        return rounded_points
    
    @staticmethod
    def validate_point_coordinates(
        points: Union[np.ndarray, List], 
        image_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Validate point coordinates.
        
        Args:
            points: Points to validate
            image_size: Optional (width, height) for boundary checking
            
        Returns:
            bool: True if all points are valid
            
        Raises:
            ValueError: If points are invalid
        """
        if isinstance(points, list):
            points = np.array(points)
            
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Points must have shape (n, 2), got {points.shape}")
            
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            raise ValueError("Points contain NaN or infinite values")
            
        if image_size is not None:
            width, height = image_size
            if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
                raise ValueError(f"X coordinates out of bounds [0, {width})")
            if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
                raise ValueError(f"Y coordinates out of bounds [0, {height})")
                
        return True
    
    @staticmethod
    def create_optical_axis_reference_points(
        rut_points: Tuple[np.ndarray, np.ndarray],
        optical_center: Tuple[float, float]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Create optical axis reference points for X and Y directions.
        
        Args:
            rut_points: Tuple of (left_point, right_point) arrays
            optical_center: (cx, cy) optical center coordinates
            
        Returns:
            Tuple containing:
                - (optical_axis_ref_y_left, optical_axis_ref_y_right): Y-axis reference points
                - (optical_axis_ref_x_left, optical_axis_ref_x_right): X-axis reference points
        """
        rut_left, rut_right = rut_points
        cx, cy = optical_center
        
        # Create Y-axis reference points (same X as rut points, Y from optical center)
        optical_axis_ref_y_left = np.array([[rut_left[0][0], cy]])
        optical_axis_ref_y_right = np.array([[rut_right[0][0], cy]])
        
        # Create X-axis reference points (X from optical center, same Y as rut points)
        optical_axis_ref_x_left = np.array([[cx, rut_left[0][1]]])
        optical_axis_ref_x_right = np.array([[cx, rut_right[0][1]]])
        
        return ((optical_axis_ref_y_left, optical_axis_ref_y_right),
                (optical_axis_ref_x_left, optical_axis_ref_x_right))


class RutPointLoader:
    """Handles loading and parsing of rut point data from JSON files."""
    
    @staticmethod
    def load_rut_points_from_json(
        parameter_path: Path, 
        pair_name: str
    ) -> dict:
        """
        Load rut points from JSON file.
        
        Args:
            parameter_path: Path to parameter directory
            pair_name: Name of the image pair
            
        Returns:
            dict: Dictionary mapping labels to (x, y) coordinates
            
        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If required rut points are missing
        """
        expected_file = parameter_path / f"left_{pair_name}.json"
        
        if not expected_file.exists():
            raise FileNotFoundError(f"Point JSON not found. Expected at: {expected_file}")
            
        with open(expected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        shapes = data.get('shapes', [])
        label_to_point = {}
        
        for item in shapes:
            label = item.get('label')
            pts = item.get('points')
            
            if (label in ('rut_1', 'rut_2') and 
                isinstance(pts, list) and len(pts) > 0 and 
                isinstance(pts[0], list) and len(pts[0]) >= 2):
                
                x, y = pts[0][0], pts[0][1]
                label_to_point[label] = (float(x), float(y))
                
        if 'rut_1' not in label_to_point or 'rut_2' not in label_to_point:
            raise ValueError(f"JSON missing required rut_1/rut_2 points: {expected_file}")
            
        return label_to_point
    
    @staticmethod
    def extract_start_end_coordinates(
        label_to_point: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right points from label dictionary.
        
        Args:
            label_to_point: Dictionary mapping labels to coordinates
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_point, right_point) arrays with shape (1, 2)
        """
        left_point = np.array([label_to_point['rut_1']], dtype=np.float64)
        right_point = np.array([label_to_point['rut_2']], dtype=np.float64)
        
        return left_point, right_point
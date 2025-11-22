"""
Rut depth calculation algorithms module.

This module provides different methods for calculating rut depth:
- Original method: Simple peak-to-valley distance calculation
- Improved method: Advanced perpendicular distance with peak validation

Input formats:
- Point data: numpy arrays with shape (n, 2) where columns are [x, y]
- Units: x in meters, y in millimeters (for improved method)

Output formats:
- Depth values: float in millimeters
- Detailed results: Dictionary with calculation metadata
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


class RutCalculator:
    """
    Handles rut depth calculation using different algorithms.
    
    Supports two calculation methods:
    - Original: Traditional peak-to-valley distance
    - Improved: Perpendicular distance with advanced peak detection
    """
    
    def __init__(self):
        """Initialize RutCalculator."""
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this class."""
        logger = logging.getLogger(f"{__name__}.RutCalculator")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def calculate_rut_depth_original(self, points: np.ndarray) -> float:
        """
        Calculate rut depth using original method.
        
        Finds the minimum point and maximum points on left and right sides,
        then calculates the perpendicular distance from minimum to the line
        connecting the two maximum points.
        
        Args:
            points: Rut coordinate data with shape (n, 2), units in meters
            
        Returns:
            Rut depth in meters
        """
        all_min = np.argmin(points[:, 1])  # Index of minimum y value
        
        # Find left maximum (before minimum)
        l_max = 0
        if all_min != 0:
            l_max = np.argmax(points[:all_min, 1])
        
        # Find right maximum (after minimum)  
        r_max = all_min + np.argmax(points[all_min:, 1])
        
        p1 = points[l_max]   # Left peak point
        p2 = points[r_max]   # Right peak point
        p3 = points[all_min] # Minimum point
        
        # Calculate perpendicular distance from p3 to line p1-p2
        u = np.array([p2[0] - p1[0], p2[1] - p1[1]])  # Direction vector
        v = np.array([p3[0] - p1[0], p3[1] - p1[1]])  # Vector to minimum point
        
        # Distance = |cross product| / |direction vector|
        dist = float(abs(np.cross(u, v) / np.linalg.norm(u)))
        
        return dist    
    def calculate_rut_depth_improved(self, points: np.ndarray, 
                                   min_distance_ratio: float = 0.1,
                                   fallback_to_local_peaks: bool = True,
                                   output_folder: Optional[str] = None,
                                   pair_name: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate rut depth using improved method with advanced peak detection.
        
        This method provides more robust peak detection and perpendicular distance
        calculation with detailed result metadata.
        
        Args:
            points: Rut coordinate data with shape (n, 2)
                   Units: x in meters, y in millimeters
            min_distance_ratio: Minimum distance ratio between peaks (0.0-1.0)
            fallback_to_local_peaks: Whether to fallback to local peaks if global peaks fail
            output_folder: Optional folder path for saving detailed results
            pair_name: Optional pair name for file naming
            
        Returns:
            Tuple of (depth_in_millimeters, detailed_results_dict)
        """
        
        def find_global_peaks(points: np.ndarray) -> Tuple[int, int]:
            """Find the highest and second highest points globally."""
            y_values = points[:, 1]
            sorted_indices = np.argsort(y_values)[::-1]  # Sort in descending order
            return sorted_indices[0], sorted_indices[1]
        
        def find_local_peaks(points: np.ndarray, min_idx: int) -> Tuple[int, int]:
            """Find highest points on left and right sides of minimum point."""
            l_max = 0
            if min_idx > 0:
                l_max = np.argmax(points[:min_idx, 1])
            
            r_max = min_idx + np.argmax(points[min_idx:, 1])
            return l_max, r_max
        
        def validate_peak_positions(peak1_idx: int, peak2_idx: int, min_idx: int, 
                                  points: np.ndarray) -> bool:
            """Validate that peaks are on opposite sides of minimum and sufficiently separated."""
            # Check if peaks are on opposite sides of minimum
            if not ((peak1_idx < min_idx < peak2_idx) or (peak2_idx < min_idx < peak1_idx)):
                return False
            
            # Check minimum distance ratio
            total_length = len(points) - 1
            min_distance = int(total_length * min_distance_ratio)
            
            if abs(peak1_idx - peak2_idx) < min_distance:
                return False
                
            return True
        
        # Find minimum point
        min_idx = np.argmin(points[:, 1])
        min_point = points[min_idx]
        
        # Try global peaks first
        peak1_idx, peak2_idx = find_global_peaks(points)
        
        # Validate peak positions
        if not validate_peak_positions(peak1_idx, peak2_idx, min_idx, points):
            if fallback_to_local_peaks:
                # Fallback to local peaks
                peak1_idx, peak2_idx = find_local_peaks(points, min_idx)
                self.logger.info("Using local peaks as fallback")
            else:
                raise ValueError("Invalid peak positions found and fallback disabled")
        
        # Ensure peak1 is left, peak2 is right
        if peak1_idx > peak2_idx:
            peak1_idx, peak2_idx = peak2_idx, peak1_idx
        
        p1 = points[peak1_idx]  # Left peak
        p2 = points[peak2_idx]  # Right peak  
        p3 = min_point          # Minimum point
        
        # Convert coordinates to consistent units for calculation (all in mm)
        # X: meters to mm, Y: already in mm
        p1_mm = np.array([p1[0] * 1000, p1[1]])
        p2_mm = np.array([p2[0] * 1000, p2[1]])
        p3_mm = np.array([p3[0] * 1000, p3[1]])
        
        # Calculate baseline equation: y = mx + b (in mm units)
        if p2_mm[0] != p1_mm[0]:
            m_baseline = (p2_mm[1] - p1_mm[1]) / (p2_mm[0] - p1_mm[0])
            b_baseline = p1_mm[1] - m_baseline * p1_mm[0]
            
            # Perpendicular line equation through minimum point
            m_perp = -1 / m_baseline if m_baseline != 0 else float('inf')
            b_perp = p3_mm[1] - m_perp * p3_mm[0]
            
            # Find intersection point (in mm)
            x_intersect = (b_perp - b_baseline) / (m_baseline - m_perp)
            y_intersect = m_baseline * x_intersect + b_baseline
            intersect_point_mm = np.array([x_intersect, y_intersect])
            
            # Check if intersection is between peaks
            x_min = min(p1_mm[0], p2_mm[0])
            x_max = max(p1_mm[0], p2_mm[0])
            is_between_peaks = x_min <= x_intersect <= x_max
            
            # Calculate depth in mm
            depth_mm = float(np.linalg.norm(p3_mm - intersect_point_mm))
            
            # Convert intersection back to original units for output (x in m, y in mm)
            intersect_point = np.array([x_intersect / 1000, y_intersect])
            
        else:
            # Vertical baseline case
            depth_mm = 0.0
            intersect_point = p1
            m_baseline = float('inf')
            b_baseline = p1_mm[0]
            m_perp = 0.0
            b_perp = p3_mm[1]
            is_between_peaks = True
        
        # Create detailed results
        result = {
            "depth_mm": depth_mm,
            "depth_m": depth_mm / 1000,  # Convert to meters for compatibility
            "min_point": {"x": float(p3[0]), "y": float(p3[1]), "index": int(min_idx)},
            "peak1": {"x": float(p1[0]), "y": float(p1[1]), "index": int(peak1_idx)},
            "peak2": {"x": float(p2[0]), "y": float(p2[1]), "index": int(peak2_idx)},
            "baseline_start": {"x": float(p1[0]), "y": float(p1[1])},
            "baseline_end": {"x": float(p2[0]), "y": float(p2[1])},
            "intersection_point": {"x": float(intersect_point[0]), "y": float(intersect_point[1])},
            "baseline_equation": {"slope": float(m_baseline), "intercept": float(b_baseline)},
            "perpendicular_equation": {"slope": float(m_perp), "intercept": float(b_perp)},
            "intersection_between_peaks": bool(is_between_peaks),
            "calculation_method": "improved_perpendicular_distance",
            "input_data_points": len(points),
            "units": {"input_x": "meters", "input_y": "millimeters", "output": "millimeters"}
        }
        
        # Save detailed results to JSON if output folder is provided
        if output_folder and pair_name:
            self._save_detailed_results(result, output_folder, pair_name)
        
        return depth_mm, result

    def _save_detailed_results(self, result: Dict[str, Any], output_folder: str, pair_name: str):
        """
        Save detailed calculation results to JSON file.
        
        Args:
            result: Detailed calculation results dictionary
            output_folder: Output folder path
            pair_name: Pair name for file naming
        """
        output_path = Path(output_folder) / f'rut_calculation_detailed_{pair_name}.json'
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved detailed calculation results to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save detailed results: {e}")


# Standalone functions for backward compatibility
def calculate_rut_value(points: np.ndarray) -> float:
    """
    Calculate rut depth using original method (backward compatibility).
    
    Args:
        points: Rut coordinate data (unit: meters)
        
    Returns:
        Rut depth (unit: meters)
    """
    calculator = RutCalculator()
    return calculator.calculate_rut_depth_original(points)


def calculate_rut_value_improved(points: np.ndarray, 
                               min_distance_ratio: float = 0.1,
                               fallback_to_local_peaks: bool = True,
                               output_folder: Optional[str] = None,
                               pair_name: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate rut depth using improved method (backward compatibility).
    
    Args:
        points: Rut coordinate data (unit: x in meters, y in millimeters)
        min_distance_ratio: Minimum distance ratio between peaks
        fallback_to_local_peaks: Whether to fallback to local peaks if global peaks fail
        output_folder: Output folder path for saving results
        pair_name: Pair name for file naming
        
    Returns:
        Tuple of (rut_depth_in_millimeters, detailed_results)
    """
    calculator = RutCalculator()
    return calculator.calculate_rut_depth_improved(
        points, min_distance_ratio, fallback_to_local_peaks, output_folder, pair_name)
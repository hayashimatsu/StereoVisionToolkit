# """
# Rut depth calculation algorithms and utilities.

# This module provides various algorithms for calculating rut depth from coordinate data:
# - Original peak-to-valley method
# - Improved perpendicular distance method
# - Peak detection utilities
# - Validation functions

# Author: Extracted from height module refactoring
# """

# import numpy as np
# from typing import Tuple, Dict, Any, Optional, List
# import logging


# logger = logging.getLogger(__name__)


# def calculate_rut_depth_original(points: np.ndarray) -> float:
#     """
#     Calculate rut depth using original method.
    
#     Finds the minimum point and maximum points on left and right sides,
#     then calculates the perpendicular distance from minimum to the line
#     connecting the two maximum points.
    
#     Args:
#         points: Rut coordinate data with shape (n, 2), units in meters
        
#     Returns:
#         Rut depth in meters
#     """
#     all_min = np.argmin(points[:, 1])  # Index of minimum y value
    
#     # Find left maximum (before minimum)
#     l_max = 0
#     if all_min != 0:
#         l_max = np.argmax(points[:all_min, 1])
    
#     # Find right maximum (after minimum)  
#     r_max = all_min + np.argmax(points[all_min:, 1])
    
#     p1 = points[l_max]   # Left peak point
#     p2 = points[r_max]   # Right peak point
#     p3 = points[all_min] # Minimum point
    
#     # Calculate perpendicular distance from p3 to line p1-p2
#     u = np.array([p2[0] - p1[0], p2[1] - p1[1]])  # Direction vector
#     v = np.array([p3[0] - p1[0], p3[1] - p1[1]])  # Vector to minimum point
    
#     # Distance = |cross product| / |direction vector|
#     dist = float(abs(np.cross(u, v) / np.linalg.norm(u)))
    
#     return dist


# def find_global_peaks(points: np.ndarray) -> Tuple[int, int]:
#     """
#     Find the highest and second highest points globally.
    
#     Args:
#         points: Coordinate data with shape (n, 2)
        
#     Returns:
#         Tuple of (peak1_index, peak2_index)
#     """
#     y_values = points[:, 1]
#     sorted_indices = np.argsort(y_values)[::-1]  # Sort in descending order
#     return sorted_indices[0], sorted_indices[1]


# def find_local_peaks(points: np.ndarray, min_idx: int) -> Tuple[int, int]:
#     """
#     Find highest points on left and right sides of minimum point.
    
#     Args:
#         points: Coordinate data with shape (n, 2)
#         min_idx: Index of minimum point
        
#     Returns:
#         Tuple of (left_peak_index, right_peak_index)
#     """
#     l_max = 0
#     if min_idx > 0:
#         l_max = np.argmax(points[:min_idx, 1])
    
#     r_max = min_idx + np.argmax(points[min_idx:, 1])
#     return l_max, r_max


# def validate_peak_positions(peak1_idx: int, peak2_idx: int, min_idx: int, 
#                           points: np.ndarray, min_distance_ratio: float = 0.1) -> bool:
#     """
#     Validate that peaks are on opposite sides of minimum and sufficiently separated.
    
#     Args:
#         peak1_idx: Index of first peak
#         peak2_idx: Index of second peak
#         min_idx: Index of minimum point
#         points: Coordinate data
#         min_distance_ratio: Minimum distance ratio between peaks (0.0-1.0)
        
#     Returns:
#         True if peaks are valid, False otherwise
#     """
#     # Check if peaks are on opposite sides of minimum
#     if not ((peak1_idx < min_idx < peak2_idx) or (peak2_idx < min_idx < peak1_idx)):
#         return False
    
#     # Check minimum distance ratio
#     total_length = len(points) - 1
#     min_distance = int(total_length * min_distance_ratio)
    
#     if abs(peak1_idx - peak2_idx) < min_distance:
#         return False
        
#     return True


# # def calculate_perpendicular_distance(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
# #     """
# #     Calculate perpendicular distance from point p3 to line defined by p1-p2.
    
# #     Args:
# #         p1: First point on line [x, y]
# #         p2: Second point on line [x, y]
# #         p3: Point to calculate distance from [x, y]
        
# #     Returns:
# #         Perpendicular distance
# #     """
# #     # Convert to consistent units (all in mm)
# #     p1_mm = np.array([p1[0] * 1000, p1[1]]) if p1[0] < 10 else np.array([p1[0], p1[1]])
# #     p2_mm = np.array([p2[0] * 1000, p2[1]]) if p2[0] < 10 else np.array([p2[0], p2[1]])
# #     p3_mm = np.array([p3[0] * 1000, p3[1]]) if p3[0] < 10 else np.array([p3[0], p3[1]])
    
# #     # Calculate baseline equation: y = mx + b
# #     if p2_mm[0] != p1_mm[0]:
# #         m_baseline = (p2_mm[1] - p1_mm[1]) / (p2_mm[0] - p1_mm[0])
# #         b_baseline = p1_mm[1] - m_baseline * p1_mm[0]
        
# #         # Perpendicular line equation through minimum point
# #         m_perp = -1 / m_baseline if m_baseline != 0 else float('inf')
# #         b_perp = p3_mm[1] - m_perp * p3_mm[0]
        
# #         # Find intersection point
# #         x_intersect = (b_perp - b_baseline) / (m_baseline - m_perp)
# #         y_intersect = m_baseline * x_intersect + b_baseline
# #         intersect_point_mm = np.array([x_intersect, y_intersect])
        
# #         # Calculate distance
# #         distance = float(np.linalg.norm(p3_mm - intersect_point_mm))
# #     else:
# #         # Vertical baseline case
# #         distance = 0.0
    
# #     return distance


# def calculate_baseline_equation(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
#     """
#     Calculate baseline equation parameters y = mx + b.
    
#     Args:
#         p1: First point [x, y]
#         p2: Second point [x, y]
        
#     Returns:
#         Tuple of (slope, intercept)
#     """
#     if p2[0] == p1[0]:
#         return float('inf'), p1[0]  # Vertical line
    
#     slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
#     intercept = p1[1] - slope * p1[0]
    
#     return slope, intercept


# def find_intersection_point(baseline_slope: float, baseline_intercept: float,
#                            perp_slope: float, perp_intercept: float) -> np.ndarray:
#     """
#     Find intersection point of two lines.
    
#     Args:
#         baseline_slope: Slope of baseline
#         baseline_intercept: Intercept of baseline
#         perp_slope: Slope of perpendicular line
#         perp_intercept: Intercept of perpendicular line
        
#     Returns:
#         Intersection point [x, y]
#     """
#     if baseline_slope == perp_slope:
#         raise ValueError("Lines are parallel, no intersection")
    
#     x_intersect = (perp_intercept - baseline_intercept) / (baseline_slope - perp_slope)
#     y_intersect = baseline_slope * x_intersect + baseline_intercept
    
#     return np.array([x_intersect, y_intersect])


# def validate_rut_data(points: np.ndarray) -> bool:
#     """
#     Validate rut coordinate data for calculation.
    
#     Args:
#         points: Coordinate data with shape (n, 2)
        
#     Returns:
#         True if data is valid, False otherwise
#     """
#     if points.shape[0] < 3:
#         logger.error("Insufficient points for rut calculation (need at least 3)")
#         return False
    
#     if points.shape[1] != 2:
#         logger.error(f"Points must have 2 columns, got {points.shape[1]}")
#         return False
    
#     if np.any(np.isnan(points)) or np.any(np.isinf(points)):
#         logger.error("Points contain NaN or infinite values")
#         return False
    
#     return True


# def smooth_rut_profile(points: np.ndarray, window_size: int = 5) -> np.ndarray:
#     """
#     Apply simple moving average smoothing to rut profile.
    
#     Args:
#         points: Coordinate data with shape (n, 2)
#         window_size: Size of smoothing window
        
#     Returns:
#         Smoothed coordinate data
#     """
#     if window_size <= 1 or window_size >= len(points):
#         return points.copy()
    
#     smoothed = points.copy()
#     half_window = window_size // 2
    
#     for i in range(half_window, len(points) - half_window):
#         smoothed[i, 1] = np.mean(points[i-half_window:i+half_window+1, 1])
    
#     return smoothed


# def detect_outliers(points: np.ndarray, threshold: float = 3.0) -> np.ndarray:
#     """
#     Detect outliers in rut profile using z-score method.
    
#     Args:
#         points: Coordinate data with shape (n, 2)
#         threshold: Z-score threshold for outlier detection
        
#     Returns:
#         Boolean array indicating outliers
#     """
#     y_values = points[:, 1]
#     z_scores = np.abs((y_values - np.mean(y_values)) / np.std(y_values))
#     return z_scores > threshold


# def calculate_rut_statistics(points: np.ndarray) -> Dict[str, float]:
#     """
#     Calculate basic statistics for rut profile.
    
#     Args:
#         points: Coordinate data with shape (n, 2)
        
#     Returns:
#         Dictionary with statistical measures
#     """
#     y_values = points[:, 1]
    
#     return {
#         'min_depth': float(np.min(y_values)),
#         'max_height': float(np.max(y_values)),
#         'mean_height': float(np.mean(y_values)),
#         'std_height': float(np.std(y_values)),
#         'range': float(np.max(y_values) - np.min(y_values)),
#         'profile_length': float(np.max(points[:, 0]) - np.min(points[:, 0])),
#         'num_points': len(points)
#     }
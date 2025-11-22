# """
# Disparity visualization utilities for stereo vision applications.

# This module provides specialized visualization tools for disparity maps,
# including color mapping, depth visualization, and comparison tools that
# can be used across multiple stereo vision modules.
# """

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import logging
# from typing import Dict, Any, List, Tuple, Optional, Union
# from pathlib import Path

# logger = logging.getLogger(__name__)


# class DisparityVisualizer:
#     """Specialized visualization tools for disparity maps."""
    
#     @staticmethod
#     def create_disparity_colormap(
#         disparity: np.ndarray,
#         colormap: Union[int, str] = cv2.COLORMAP_JET,
#         min_disparity: Optional[float] = None,
#         max_disparity: Optional[float] = None,
#         invalid_color: Tuple[int, int, int] = (0, 0, 0)
#     ) -> np.ndarray:
#         """
#         Create color-mapped disparity visualization.
        
#         Args:
#             disparity: Disparity map (normalized to pixels)
#             colormap: OpenCV colormap or matplotlib colormap name
#             min_disparity: Minimum disparity for color mapping (auto if None)
#             max_disparity: Maximum disparity for color mapping (auto if None)
#             invalid_color: Color for invalid pixels (BGR format)
            
#         Returns:
#             np.ndarray: Color-mapped disparity image (BGR format)
#         """
#         # Handle invalid disparities
#         valid_mask = disparity > 0
        
#         if not np.any(valid_mask):
#             # No valid disparities, return black image
#             return np.zeros((*disparity.shape, 3), dtype=np.uint8)
        
#         # Determine disparity range
#         if min_disparity is None or max_disparity is None:
#             valid_disparity = disparity[valid_mask]
#             if min_disparity is None:
#                 min_disparity = float(valid_disparity.min())
#             if max_disparity is None:
#                 max_disparity = float(valid_disparity.max())
        
#         # Normalize disparity to 0-255 range
#         disparity_normalized = np.zeros_like(disparity, dtype=np.uint8)
        
#         if max_disparity > min_disparity:
#             # Clip and normalize
#             disparity_clipped = np.clip(disparity, min_disparity, max_disparity)
#             disparity_normalized[valid_mask] = (
#                 255 * (disparity_clipped[valid_mask] - min_disparity) / 
#                 (max_disparity - min_disparity)
#             ).astype(np.uint8)
        
#         # Apply colormap
#         if isinstance(colormap, str):
#             # Use matplotlib colormap
#             cmap = plt.get_cmap(colormap)
#             colored = (cmap(disparity_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)
#             # Convert RGB to BGR
#             colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
#         else:
#             # Use OpenCV colormap
#             colored = cv2.applyColorMap(disparity_normalized, colormap)
        
#         # Set invalid pixels to specified color
#         colored[~valid_mask] = invalid_color
        
#         return colored
    
#     @staticmethod
#     def create_depth_visualization(
#         disparity: np.ndarray,
#         focal_length: float,
#         baseline: float,
#         max_depth: float = 10.0,
#         colormap: str = 'plasma_r'
#     ) -> np.ndarray:
#         """
#         Create depth visualization from disparity map.
        
#         Args:
#             disparity: Disparity map in pixels
#             focal_length: Focal length in pixels
#             baseline: Baseline in meters
#             max_depth: Maximum depth for visualization (meters)
#             colormap: Matplotlib colormap name
            
#         Returns:
#             np.ndarray: Depth visualization image (BGR format)
#         """
#         # Convert disparity to depth
#         valid_mask = disparity > 0
#         depth = np.zeros_like(disparity, dtype=np.float32)
        
#         if np.any(valid_mask):
#             depth[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
        
#         # Clip depth to reasonable range
#         depth = np.clip(depth, 0, max_depth)
        
#         # Create depth visualization
#         depth_vis = DisparityVisualizer.create_disparity_colormap(
#             depth, colormap, 0, max_depth
#         )
        
#         return depth_vis
    
#     @staticmethod
#     def create_disparity_comparison(
#         disparity1: np.ndarray,
#         disparity2: np.ndarray,
#         labels: Tuple[str, str] = ("Disparity 1", "Disparity 2"),
#         colormap: int = cv2.COLORMAP_JET
#     ) -> np.ndarray:
#         """
#         Create side-by-side comparison of two disparity maps.
        
#         Args:
#             disparity1: First disparity map
#             disparity2: Second disparity map
#             labels: Labels for the two disparities
#             colormap: OpenCV colormap
            
#         Returns:
#             np.ndarray: Side-by-side comparison image
#         """
#         # Ensure same size
#         if disparity1.shape != disparity2.shape:
#             raise ValueError(f"Disparity shapes don't match: {disparity1.shape} vs {disparity2.shape}")
        
#         # Find common range for consistent coloring
#         valid1 = disparity1 > 0
#         valid2 = disparity2 > 0
        
#         if np.any(valid1) and np.any(valid2):
#             min_disp = min(disparity1[valid1].min(), disparity2[valid2].min())
#             max_disp = max(disparity1[valid1].max(), disparity2[valid2].max())
#         elif np.any(valid1):
#             min_disp, max_disp = disparity1[valid1].min(), disparity1[valid1].max()
#         elif np.any(valid2):
#             min_disp, max_disp = disparity2[valid2].min(), disparity2[valid2].max()
#         else:
#             min_disp, max_disp = 0, 1
        
#         # Create colored versions
#         colored1 = DisparityVisualizer.create_disparity_colormap(
#             disparity1, colormap, min_disp, max_disp
#         )
#         colored2 = DisparityVisualizer.create_disparity_colormap(
#             disparity2, colormap, min_disp, max_disp
#         )
        
#         # Add labels
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.7
#         color = (255, 255, 255)
#         thickness = 2
        
#         cv2.putText(colored1, labels[0], (10, 30), font, font_scale, color, thickness)
#         cv2.putText(colored2, labels[1], (10, 30), font, font_scale, color, thickness)
        
#         # Combine side by side
#         comparison = np.hstack([colored1, colored2])
        
#         return comparison
    
#     @staticmethod
#     def create_disparity_difference(
#         disparity1: np.ndarray,
#         disparity2: np.ndarray,
#         max_diff: Optional[float] = None
#     ) -> np.ndarray:
#         """
#         Create visualization of disparity difference.
        
#         Args:
#             disparity1: First disparity map
#             disparity2: Second disparity map
#             max_diff: Maximum difference for color scaling (auto if None)
            
#         Returns:
#             np.ndarray: Difference visualization image
#         """
#         if disparity1.shape != disparity2.shape:
#             raise ValueError(f"Disparity shapes don't match: {disparity1.shape} vs {disparity2.shape}")
        
#         # Calculate difference only where both are valid
#         valid_mask = (disparity1 > 0) & (disparity2 > 0)
#         difference = np.zeros_like(disparity1)
        
#         if np.any(valid_mask):
#             difference[valid_mask] = np.abs(disparity1[valid_mask] - disparity2[valid_mask])
        
#         # Determine max difference for scaling
#         if max_diff is None and np.any(valid_mask):
#             max_diff = difference[valid_mask].max()
#         elif max_diff is None:
#             max_diff = 1.0
        
#         # Create visualization
#         diff_vis = DisparityVisualizer.create_disparity_colormap(
#             difference, cv2.COLORMAP_HOT, 0, max_diff
#         )
        
#         return diff_vis
    
#     @staticmethod
#     def add_disparity_markers(
#         disparity_image: np.ndarray,
#         points: List[Tuple[int, int]],
#         colors: Optional[List[Tuple[int, int, int]]] = None,
#         marker_size: int = 20,
#         thickness: int = 3
#     ) -> np.ndarray:
#         """
#         Add markers to disparity visualization.
        
#         Args:
#             disparity_image: Disparity visualization image
#             points: List of (x, y) coordinates to mark
#             colors: List of BGR colors for each point (auto if None)
#             marker_size: Size of markers
#             thickness: Line thickness
            
#         Returns:
#             np.ndarray: Image with markers added
#         """
#         marked_image = disparity_image.copy()
        
#         # Default colors if not provided
#         if colors is None:
#             default_colors = [
#                 (0, 0, 255),    # Red
#                 (0, 255, 0),    # Green
#                 (255, 0, 0),    # Blue
#                 (0, 255, 255),  # Yellow
#                 (255, 0, 255),  # Magenta
#                 (255, 255, 0),  # Cyan
#             ]
#             colors = [default_colors[i % len(default_colors)] for i in range(len(points))]
        
#         # Add markers
#         for i, (x, y) in enumerate(points):
#             color = colors[i] if i < len(colors) else (255, 255, 255)
#             cv2.drawMarker(
#                 marked_image, (x, y), color,
#                 markerType=cv2.MARKER_CROSS,
#                 markerSize=marker_size,
#                 thickness=thickness
#             )
        
#         return marked_image
    
#     @staticmethod
#     def create_disparity_profile(
#         disparity: np.ndarray,
#         line_coords: Tuple[Tuple[int, int], Tuple[int, int]],
#         title: str = "Disparity Profile"
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Extract and visualize disparity profile along a line.
        
#         Args:
#             disparity: Disparity map
#             line_coords: ((x1, y1), (x2, y2)) line endpoints
#             title: Title for the profile plot
            
#         Returns:
#             Tuple[np.ndarray, np.ndarray]: (distances, disparity_values)
#         """
#         (x1, y1), (x2, y2) = line_coords
        
#         # Generate line coordinates
#         num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
#         x_coords = np.linspace(x1, x2, num_points).astype(int)
#         y_coords = np.linspace(y1, y2, num_points).astype(int)
        
#         # Extract disparity values along the line
#         disparity_values = []
#         distances = []
        
#         for i, (x, y) in enumerate(zip(x_coords, y_coords)):
#             if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
#                 disp_val = disparity[y, x]
#                 disparity_values.append(disp_val)
#                 distances.append(i)
        
#         return np.array(distances), np.array(disparity_values)
    
#     @staticmethod
#     def create_disparity_histogram(
#         disparity: np.ndarray,
#         bins: int = 50,
#         title: str = "Disparity Histogram"
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Create histogram of disparity values.
        
#         Args:
#             disparity: Disparity map
#             bins: Number of histogram bins
#             title: Title for the histogram
            
#         Returns:
#             Tuple[np.ndarray, np.ndarray]: (bin_centers, counts)
#         """
#         # Extract valid disparity values
#         valid_disparity = disparity[disparity > 0]
        
#         if len(valid_disparity) == 0:
#             return np.array([]), np.array([])
        
#         # Create histogram
#         counts, bin_edges = np.histogram(valid_disparity, bins=bins)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
#         return bin_centers, counts
    
#     @staticmethod
#     def save_disparity_visualization(
#         disparity: np.ndarray,
#         output_path: Path,
#         filename: str,
#         visualization_type: str = 'colormap',
#         **kwargs
#     ) -> bool:
#         """
#         Save disparity visualization to file.
        
#         Args:
#             disparity: Disparity map
#             output_path: Output directory
#             filename: Output filename
#             visualization_type: Type of visualization ('colormap', 'depth', 'grayscale')
#             **kwargs: Additional arguments for visualization functions
            
#         Returns:
#             bool: True if successful
#         """
#         try:
#             output_path.mkdir(parents=True, exist_ok=True)
            
#             if visualization_type == 'colormap':
#                 vis_image = DisparityVisualizer.create_disparity_colormap(disparity, **kwargs)
#             elif visualization_type == 'depth':
#                 vis_image = DisparityVisualizer.create_depth_visualization(disparity, **kwargs)
#             elif visualization_type == 'grayscale':
#                 # Simple grayscale normalization
#                 valid_mask = disparity > 0
#                 if np.any(valid_mask):
#                     disp_norm = cv2.normalize(
#                         disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
#                     )
#                     vis_image = cv2.cvtColor(disp_norm, cv2.COLOR_GRAY2BGR)
#                 else:
#                     vis_image = np.zeros((*disparity.shape, 3), dtype=np.uint8)
#             else:
#                 raise ValueError(f"Unknown visualization type: {visualization_type}")
            
#             # Save image
#             full_path = output_path / filename
#             cv2.imwrite(str(full_path), vis_image)
            
#             logger.info(f"Saved disparity visualization: {filename}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to save disparity visualization: {e}")
#             return False
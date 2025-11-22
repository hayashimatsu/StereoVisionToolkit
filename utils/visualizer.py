"""
Visualization utilities for stereo vision applications.

This module provides common visualization functions including point marking,
image overlays, and drawing utilities that can be used across multiple
computer vision modules.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class Colors:
    """Standard colors for visualization."""
    
    # BGR format for OpenCV
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    
    # Common marker colors
    ENDPOINT_LEFT = RED
    ENDPOINT_RIGHT = GREEN
    INTERPOLATED_POINTS = YELLOW
    OPTICAL_CENTER = BLUE
    REFERENCE_POINTS = CYAN


class MarkerStyles:
    """Standard marker styles and sizes."""
    
    # Marker types
    CIRCLE = 'circle'
    CROSS = 'cross'
    SQUARE = 'square'
    TRIANGLE = 'triangle'
    
    # Sizes
    SMALL_MARKER = 3
    MEDIUM_MARKER = 10
    LARGE_MARKER = 20
    EXTRA_LARGE_MARKER = 50
    
    # Thickness
    THIN_LINE = 1
    MEDIUM_LINE = 3
    THICK_LINE = 5


class PointVisualizer:
    """Handles point marking and visualization on images."""
    
    @staticmethod
    def draw_points(
        image: np.ndarray,
        points: List[List[Union[int, float]]],
        color: Tuple[int, int, int] = Colors.YELLOW,
        marker_size: int = MarkerStyles.SMALL_MARKER,
        marker_type: str = MarkerStyles.CIRCLE,
        thickness: int = -1
    ) -> np.ndarray:
        """
        Draw points on image.
        
        Args:
            image: Input image
            points: List of [x, y] coordinates
            color: BGR color tuple
            marker_size: Size of markers
            marker_type: Type of marker ('circle', 'cross', 'square')
            thickness: Line thickness (-1 for filled)
            
        Returns:
            np.ndarray: Image with drawn points
        """
        img_copy = image.copy()
        
        for point in points:
            x, y = int(round(point[0])), int(round(point[1]))
            
            if marker_type == MarkerStyles.CIRCLE:
                cv2.circle(img_copy, (x, y), marker_size, color, thickness)
            elif marker_type == MarkerStyles.CROSS:
                cv2.drawMarker(img_copy, (x, y), color, 
                              markerType=cv2.MARKER_CROSS, 
                              markerSize=marker_size, 
                              thickness=max(1, thickness))
            elif marker_type == MarkerStyles.SQUARE:
                half_size = marker_size // 2
                cv2.rectangle(img_copy, 
                            (x - half_size, y - half_size),
                            (x + half_size, y + half_size),
                            color, thickness)
            else:
                # Default to circle
                cv2.circle(img_copy, (x, y), marker_size, color, thickness)
                
        return img_copy
    
    @staticmethod
    def draw_endpoints(
        image: np.ndarray,
        points: List[List[Union[int, float]]],
        left_color: Tuple[int, int, int] = Colors.ENDPOINT_LEFT,
        right_color: Tuple[int, int, int] = Colors.ENDPOINT_RIGHT,
        marker_size: int = MarkerStyles.LARGE_MARKER,
        thickness: int = MarkerStyles.THICK_LINE
    ) -> np.ndarray:
        """
        Draw endpoint markers with different colors for left and right.
        
        Args:
            image: Input image
            points: List of points (first and last will be marked as endpoints)
            left_color: Color for left endpoint
            right_color: Color for right endpoint
            marker_size: Size of endpoint markers
            thickness: Line thickness
            
        Returns:
            np.ndarray: Image with endpoint markers
        """
        if len(points) < 2:
            return image.copy()
            
        img_copy = image.copy()
        
        # Draw left endpoint
        left_point = points[0]
        lx, ly = int(round(left_point[0])), int(round(left_point[1]))
        cv2.drawMarker(img_copy, (lx, ly), left_color,
                      markerType=cv2.MARKER_CROSS,
                      markerSize=marker_size,
                      thickness=thickness)
        
        # Draw right endpoint
        right_point = points[-1]
        rx, ry = int(round(right_point[0])), int(round(right_point[1]))
        cv2.drawMarker(img_copy, (rx, ry), right_color,
                      markerType=cv2.MARKER_CROSS,
                      markerSize=marker_size,
                      thickness=thickness)
        
        return img_copy
    
    @staticmethod
    def draw_interpolated_line_with_points(
        image: np.ndarray,
        points: List[List[Union[int, float]]],
        line_color: Tuple[int, int, int] = Colors.MAGENTA,
        point_color: Tuple[int, int, int] = Colors.YELLOW,
        endpoint_left_color: Tuple[int, int, int] = Colors.ENDPOINT_LEFT,
        endpoint_right_color: Tuple[int, int, int] = Colors.ENDPOINT_RIGHT,
        line_thickness: int = MarkerStyles.MEDIUM_LINE,
        point_size: int = MarkerStyles.SMALL_MARKER,
        endpoint_size: int = MarkerStyles.LARGE_MARKER
    ) -> np.ndarray:
        """
        Draw interpolated line with all points and highlighted endpoints.
        
        Args:
            image: Input image
            points: List of interpolated points
            line_color: Color for connecting line
            point_color: Color for interpolated points
            endpoint_left_color: Color for left endpoint
            endpoint_right_color: Color for right endpoint
            line_thickness: Thickness of connecting line
            point_size: Size of interpolated point markers
            endpoint_size: Size of endpoint markers
            
        Returns:
            np.ndarray: Image with complete visualization
        """
        if len(points) < 2:
            return image.copy()
            
        img_copy = image.copy()
        
        # Convert points to integer coordinates
        int_points = [(int(round(p[0])), int(round(p[1]))) for p in points]
        
        # Draw connecting line
        if len(int_points) > 1:
            cv2.polylines(img_copy, [np.array(int_points)], False, line_color, line_thickness)
        
        # Draw all interpolated points
        img_copy = PointVisualizer.draw_points(
            img_copy, points, point_color, point_size, MarkerStyles.CIRCLE, -1
        )
        
        # Draw endpoints with special markers
        img_copy = PointVisualizer.draw_endpoints(
            img_copy, points, endpoint_left_color, endpoint_right_color, 
            endpoint_size, MarkerStyles.THICK_LINE
        )
        
        return img_copy


class ImageOverlayCreator:
    """Creates various types of image overlays for visualization."""
    
    @staticmethod
    def create_stereo_overlay(
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_weight: float = 0.75,
        right_weight: float = 0.5,
        gamma: float = 0.0
    ) -> np.ndarray:
        """
        Create overlay of stereo image pair.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            left_weight: Weight for left image
            right_weight: Weight for right image
            gamma: Gamma correction
            
        Returns:
            np.ndarray: Overlaid image
        """
        if left_image.shape != right_image.shape:
            raise ValueError(f"Image shapes don't match: "
                           f"left={left_image.shape}, right={right_image.shape}")
        
        # Convert to RGB for blending
        if len(left_image.shape) == 3:
            left_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        else:
            left_rgb = left_image
            right_rgb = right_image
        
        # Create weighted overlay
        overlay = cv2.addWeighted(left_rgb, left_weight, right_rgb, right_weight, gamma)
        
        # Convert back to BGR if needed
        if len(left_image.shape) == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
        return overlay
    
    @staticmethod
    def create_marked_overlay(
        left_image: np.ndarray,
        right_image: np.ndarray,
        points: List[List[Union[int, float]]],
        overlay_weights: Tuple[float, float] = (0.75, 0.5),
        **marker_kwargs
    ) -> np.ndarray:
        """
        Create stereo overlay with marked points.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            points: Points to mark on overlay
            overlay_weights: (left_weight, right_weight) for overlay
            **marker_kwargs: Additional arguments for point marking
            
        Returns:
            np.ndarray: Marked overlay image
        """
        # Create base overlay
        overlay = ImageOverlayCreator.create_stereo_overlay(
            left_image, right_image, overlay_weights[0], overlay_weights[1]
        )
        
        # Add point markers
        marked_overlay = PointVisualizer.draw_interpolated_line_with_points(
            overlay, points, **marker_kwargs
        )
        
        return marked_overlay
    
    @staticmethod
    def create_marked_single_image(
        image: np.ndarray,
        points: List[List[Union[int, float]]],
        **marker_kwargs
    ) -> np.ndarray:
        """
        Create single image with marked points.
        
        Args:
            image: Input image
            points: Points to mark
            **marker_kwargs: Additional arguments for point marking
            
        Returns:
            np.ndarray: Marked image
        """
        return PointVisualizer.draw_interpolated_line_with_points(
            image, points, **marker_kwargs
        )


class OpticalCenterVisualizer:
    """Specialized visualizer for optical center and reference points."""
    
    @staticmethod
    def draw_optical_center(
        image: np.ndarray,
        optical_center: Tuple[float, float],
        color: Tuple[int, int, int] = Colors.OPTICAL_CENTER,
        marker_size: int = MarkerStyles.EXTRA_LARGE_MARKER,
        thickness: int = MarkerStyles.THICK_LINE,
        label: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw optical center marker on image.
        
        Args:
            image: Input image
            optical_center: (cx, cy) optical center coordinates
            color: Marker color
            marker_size: Size of marker
            thickness: Line thickness
            label: Optional text label
            
        Returns:
            np.ndarray: Image with optical center marker
        """
        img_copy = image.copy()
        
        cx, cy = int(round(optical_center[0])), int(round(optical_center[1]))
        
        # Draw cross marker
        cv2.drawMarker(img_copy, (cx, cy), color,
                      markerType=cv2.MARKER_CROSS,
                      markerSize=marker_size,
                      thickness=thickness)
        
        # Add text label if provided
        if label:
            cv2.putText(img_copy, label, (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return img_copy
    
    @staticmethod
    def draw_multiple_optical_centers(
        image: np.ndarray,
        centers: List[Tuple[Tuple[float, float], str, Tuple[int, int, int]]],
        marker_size: int = MarkerStyles.EXTRA_LARGE_MARKER,
        thickness: int = MarkerStyles.THICK_LINE
    ) -> np.ndarray:
        """
        Draw multiple optical centers with different colors and labels.
        
        Args:
            image: Input image
            centers: List of ((cx, cy), label, color) tuples
            marker_size: Size of markers
            thickness: Line thickness
            
        Returns:
            np.ndarray: Image with multiple optical center markers
        """
        img_copy = image.copy()
        
        for (cx, cy), label, color in centers:
            img_copy = OpticalCenterVisualizer.draw_optical_center(
                img_copy, (cx, cy), color, marker_size, thickness, label
            )
        
        return img_copy


class VisualizationSaver:
    """Handles saving of visualization results."""
    
    @staticmethod
    def save_visualization(
        image: np.ndarray,
        output_path: Path,
        filename: str,
        quality: int = 95
    ) -> bool:
        """
        Save visualization image to file.
        
        Args:
            image: Image to save
            output_path: Output directory path
            filename: Output filename
            quality: JPEG quality (if applicable)
            
        Returns:
            bool: True if successful
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            full_path = output_path / filename
            
            # Set compression parameters based on file extension
            if full_path.suffix.lower() == '.jpg' or full_path.suffix.lower() == '.jpeg':
                cv2.imwrite(str(full_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(full_path), image)
                
            logger.info(f"Saved visualization: {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save visualization {full_path}: {e}")
            return False
    
    @staticmethod
    def save_multiple_visualizations(
        images_and_names: List[Tuple[np.ndarray, str]],
        output_path: Path,
        quality: int = 95
    ) -> Dict[str, bool]:
        """
        Save multiple visualization images.
        
        Args:
            images_and_names: List of (image, filename) tuples
            output_path: Output directory path
            quality: JPEG quality
            
        Returns:
            Dict[str, bool]: Results for each file
        """
        results = {}
        
        for image, filename in images_and_names:
            success = VisualizationSaver.save_visualization(
                image, output_path, filename, quality
            )
            results[filename] = success
            
        return results


# Legacy compatibility functions
def create_rectified_overlay(left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
    """Legacy function for backward compatibility."""
    return ImageOverlayCreator.create_stereo_overlay(left_image, right_image)


def mark_points_on_image(
    image: np.ndarray, 
    points: List[List[Union[int, float]]]
) -> np.ndarray:
    """Legacy function for backward compatibility."""
    return PointVisualizer.draw_interpolated_line_with_points(image, points)
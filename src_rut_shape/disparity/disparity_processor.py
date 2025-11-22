"""
Disparity map processing utilities.

This module handles post-processing of disparity maps including normalization,
filtering, point extraction, and quality assessment.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from utils.logger_config import get_logger

logger = get_logger(__name__)


class DisparityProcessor:
    """Handles post-processing and analysis of disparity maps."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def normalize_disparity(
        self, 
        disparity: np.ndarray, 
        scale_factor: float = 16.0
    ) -> np.ndarray:
        """
        Normalize disparity map from fixed-point to floating-point format.
        
        Args:
            disparity: Raw disparity map from SGBM (16-bit fixed point)
            scale_factor: Scale factor to convert to pixels (default: 16.0)
            
        Returns:
            np.ndarray: Normalized disparity map in pixels
        """
        if disparity is None:
            raise ValueError("Disparity map cannot be None")
        
        # Convert from fixed-point to floating-point
        disparity_normalized = disparity.astype(np.float32) / scale_factor
        
        # Log statistics
        valid_pixels = disparity_normalized > 0
        if np.any(valid_pixels):
            valid_disparity = disparity_normalized[valid_pixels]
            self.logger.info(f"Disparity normalized: "
                           f"range=[{valid_disparity.min():.2f}, {valid_disparity.max():.2f}], "
                           f"mean={valid_disparity.mean():.2f}")
        
        return disparity_normalized
    
    def create_disparity_colormap(
        self, 
        disparity: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create color-mapped disparity image for visualization.
        
        Args:
            disparity: Normalized disparity map
            colormap: OpenCV colormap type
            
        Returns:
            np.ndarray: Color-mapped disparity image (BGR format)
        """
        # Normalize to 0-255 range
        disp_norm = cv2.normalize(
            disparity, None, 
            alpha=0, beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        
        # Apply colormap
        disp_color = cv2.applyColorMap(disp_norm, colormap)
        
        return disp_color
    
    def extract_rut_disparity_values(
        self, 
        disparity: np.ndarray,
        rut_points_file: Path
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Extract disparity values at rut interpolated points.
        
        Args:
            disparity: Normalized disparity map
            rut_points_file: Path to JSON file containing rut points
            
        Returns:
            Tuple containing:
                - List of disparity values at interpolated points
                - Metadata about the extraction
                
        Raises:
            FileNotFoundError: If rut points file doesn't exist
            ValueError: If rut points data is invalid
        """
        if not rut_points_file.exists():
            raise FileNotFoundError(f"Rut points file not found: {rut_points_file}")
        
        # Load rut points data
        try:
            with open(rut_points_file, 'r', encoding='utf-8') as f:
                rut_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in rut points file: {e}")
        
        # Extract interpolated points
        interpolated_points = rut_data.get('interpolated_points', [])
        if not interpolated_points:
            raise ValueError("No interpolated points found in rut data")
        
        # Extract disparity values at each point
        disparity_values = []
        valid_points = 0
        
        for point in interpolated_points:
            x, y = int(round(point[0])), int(round(point[1]))
            
            # Check bounds
            if (0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]):
                disp_value = disparity[y, x]
                disparity_values.append(float(disp_value))
                
                if disp_value > 0:
                    valid_points += 1
            else:
                # Point outside image bounds
                disparity_values.append(0.0)
        
        # Create metadata
        metadata = {
            'total_points': len(interpolated_points),
            'valid_disparity_points': valid_points,
            'validity_ratio': valid_points / len(interpolated_points) if interpolated_points else 0,
            'disparity_range': {
                'min': float(min(disparity_values)) if disparity_values else 0,
                'max': float(max(disparity_values)) if disparity_values else 0,
                'mean': float(np.mean(disparity_values)) if disparity_values else 0
            },
            'source_file': str(rut_points_file)
        }
        
        self.logger.info(f"Extracted disparity values: "
                        f"{valid_points}/{len(interpolated_points)} valid points, "
                        f"range=[{metadata['disparity_range']['min']:.2f}, "
                        f"{metadata['disparity_range']['max']:.2f}]")
        
        return disparity_values, metadata
    
    def create_marked_disparity_image(
        self,
        disparity: np.ndarray,
        rut_points_file: Path,
        colormap: int = cv2.COLORMAP_JET
    ) -> Optional[np.ndarray]:
        """
        Create disparity image with rut endpoint markers.
        
        Args:
            disparity: Normalized disparity map
            rut_points_file: Path to JSON file containing rut points
            colormap: OpenCV colormap for disparity visualization
            
        Returns:
            np.ndarray or None: Marked disparity image, or None if marking failed
        """
        try:
            # Create color disparity image
            disp_color = self.create_disparity_colormap(disparity, colormap)
            
            # Load rut points data
            with open(rut_points_file, 'r', encoding='utf-8') as f:
                rut_data = json.load(f)
            
            # Extract endpoints for marking
            endpoints = rut_data.get('endpoints', {})
            left_point = endpoints.get('left', [None, None])
            right_point = endpoints.get('right', [None, None])
            
            if None in left_point or None in right_point:
                self.logger.warning("Invalid endpoint coordinates in rut data")
                return disp_color
            
            # Draw markers for endpoints
            left_x, left_y = int(round(left_point[0])), int(round(left_point[1]))
            right_x, right_y = int(round(right_point[0])), int(round(right_point[1]))
            
            # Left endpoint marker (magenta)
            cv2.drawMarker(
                disp_color, (left_x, left_y), 
                (255, 0, 255),  # Magenta in BGR
                markerType=cv2.MARKER_TILTED_CROSS, 
                markerSize=50, 
                thickness=5
            )
            
            # Right endpoint marker (cyan)
            cv2.drawMarker(
                disp_color, (right_x, right_y), 
                (255, 255, 0),  # Cyan in BGR
                markerType=cv2.MARKER_TILTED_CROSS, 
                markerSize=50, 
                thickness=5
            )
            
            self.logger.info(f"Added endpoint markers at ({left_x}, {left_y}) and ({right_x}, {right_y})")
            return disp_color
            
        except Exception as e:
            self.logger.error(f"Failed to create marked disparity image: {e}")
            return None
    
    def assess_disparity_quality(
        self, 
        disparity: np.ndarray
    ) -> Dict[str, Any]:
        """
        Assess the quality of a disparity map.
        
        Args:
            disparity: Normalized disparity map
            
        Returns:
            Dict[str, Any]: Quality assessment metrics
        """
        # Calculate basic statistics
        valid_mask = disparity > 0
        total_pixels = disparity.size
        valid_pixels = np.sum(valid_mask)
        
        quality_metrics = {
            'total_pixels': total_pixels,
            'valid_pixels': int(valid_pixels),
            'validity_ratio': float(valid_pixels / total_pixels),
            'coverage_percentage': float(100 * valid_pixels / total_pixels)
        }
        
        if valid_pixels > 0:
            valid_disparity = disparity[valid_mask]
            
            quality_metrics.update({
                'disparity_range': {
                    'min': float(valid_disparity.min()),
                    'max': float(valid_disparity.max()),
                    'mean': float(valid_disparity.mean()),
                    'std': float(valid_disparity.std())
                },
                'dynamic_range': float(valid_disparity.max() - valid_disparity.min())
            })
            
            # Assess quality categories
            if quality_metrics['validity_ratio'] > 0.8:
                quality_level = 'excellent'
            elif quality_metrics['validity_ratio'] > 0.6:
                quality_level = 'good'
            elif quality_metrics['validity_ratio'] > 0.4:
                quality_level = 'fair'
            else:
                quality_level = 'poor'
                
            quality_metrics['quality_level'] = quality_level
        else:
            quality_metrics.update({
                'disparity_range': None,
                'dynamic_range': 0.0,
                'quality_level': 'failed'
            })
        
        self.logger.info(f"Disparity quality assessment: "
                        f"{quality_metrics['quality_level']} "
                        f"({quality_metrics['coverage_percentage']:.1f}% coverage)")
        
        return quality_metrics
    
    def filter_disparity_outliers(
        self, 
        disparity: np.ndarray,
        method: str = 'median',
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Apply filtering to remove disparity outliers.
        
        Args:
            disparity: Input disparity map
            method: Filtering method ('median', 'bilateral')
            kernel_size: Filter kernel size
            
        Returns:
            np.ndarray: Filtered disparity map
        """
        if method == 'median':
            # Apply median filter to reduce noise
            filtered = cv2.medianBlur(disparity.astype(np.uint8), kernel_size)
            filtered = filtered.astype(np.float32)
            
        elif method == 'bilateral':
            # Apply bilateral filter to preserve edges
            filtered = cv2.bilateralFilter(
                disparity.astype(np.uint8), 
                kernel_size, 
                sigmaColor=75, 
                sigmaSpace=75
            ).astype(np.float32)
            
        else:
            raise ValueError(f"Unknown filtering method: {method}")
        
        # Preserve original invalid regions
        valid_mask = disparity > 0
        filtered[~valid_mask] = 0
        
        self.logger.info(f"Applied {method} filtering with kernel size {kernel_size}")
        return filtered
    
    def create_disparity_metadata(
        self,
        disparity: np.ndarray,
        parameters: Dict[str, Any],
        quality_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for disparity results.
        
        Args:
            disparity: Disparity map
            parameters: SGBM parameters used
            quality_metrics: Optional quality assessment results
            
        Returns:
            Dict[str, Any]: Comprehensive metadata
        """
        metadata = {
            'disparity_info': {
                'shape': list(disparity.shape),
                'dtype': str(disparity.dtype),
                'processing_version': 'disparity_processor_v1.0'
            },
            'sgbm_parameters': parameters,
            'quality_metrics': quality_metrics or self.assess_disparity_quality(disparity)
        }
        
        return metadata
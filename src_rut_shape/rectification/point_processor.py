"""
Rectification-specific point processing module.

This module handles point processing operations specific to stereo rectification,
including rut point processing, optical axis calculations, and coordinate transformations.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path

from utils.point_processor import PointProcessor, RutPointLoader
from utils.logger_config import get_logger

logger = get_logger(__name__)


class RectificationPointProcessor:
    """Handles point processing operations specific to rectification."""
    
    def __init__(self, config_manager):
        """
        Initialize rectification point processor.
        
        Args:
            config_manager: ConfigManager instance for accessing configuration
        """
        self.config_manager = config_manager
        self.logger = get_logger(__name__)
    
    def process_rut_points(
        self,
        pair_name: str,
        parameter_path: Path,
        rectification_engine,
        resize_scale: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Process rut points through rectification and interpolation.
        
        Args:
            pair_name: Name of the image pair
            parameter_path: Path to parameter directory
            rectification_engine: Rectification engine instance
            resize_scale: Resize scaling factors
            
        Returns:
            Dict[str, Any]: Processed rut points data
        """
        self.logger.info(f"Processing rut points for {pair_name}")
        
        # Load rut points from JSON
        rut_points_dict = RutPointLoader.load_rut_points_from_json(parameter_path, pair_name)
        rut_left, rut_right = RutPointLoader.extract_start_end_coordinates(rut_points_dict)
        
        # Get camera parameters for optical axis calculations
        K_left = rectification_engine.K_left
        optical_center = (K_left[0, 2], K_left[1, 2])
        
        # Create optical axis reference points
        optical_ref_points = PointProcessor.create_optical_axis_reference_points(
            (rut_left, rut_right), optical_center
        )
        
        # Compute rectified points
        rectified_interpolated_points = self._compute_rectified_points(
            rut_left, rut_right, rectification_engine, resize_scale
        )
        
        # Compute optical axis reference points
        ref_y_points = self._compute_rectified_points(
            optical_ref_points[0][0], optical_ref_points[0][1], 
            rectification_engine, resize_scale,
            num=len(rectified_interpolated_points)
        )
        
        ref_x_points = self._compute_rectified_points(
            optical_ref_points[1][0], optical_ref_points[1][1], 
            rectification_engine, resize_scale,
            num=len(rectified_interpolated_points)
        )
        
        result = {
            'rectified_interpolated_points': rectified_interpolated_points,
            'optical_ref_x_points': ref_x_points,
            'optical_ref_y_points': ref_y_points,
            'original_rut_points': {'rut_left': rut_left, 'rut_right': rut_right}
        }
        
        self.logger.info(f"Processed {len(rectified_interpolated_points)} rectified points")
        return result
    
    def _compute_rectified_points(
        self,
        left_point: np.ndarray,
        right_point: np.ndarray,
        rectification_engine,
        resize_scale: Tuple[float, float],
        num: int = 10
    ) -> List[List[int]]:
        """
        Compute rectified coordinates for interpolated points.
        
        Args:
            left_point: Left point coordinates
            right_point: Right point coordinates
            rectification_engine: Rectification engine instance
            resize_scale: Resize scaling factors
            num: Number of interpolated points
            
        Returns:
            List[List[int]]: Rectified interpolated points
        """
        # Prepare rectification parameters
        rectification_params = {
            'camera_matrix': rectification_engine.K_left,
            'distortion_coeffs': rectification_engine.d_left,
            'rectification_matrix': rectification_engine.R1,
            'projection_matrix': (
                rectification_engine.P1_adjusted 
                if rectification_engine.P1_adjusted is not None 
                else rectification_engine.P1
            )
        }
        
        # Use the enhanced point processor from utils
        return PointProcessor.compute_rectified_interpolated_points(
            left_point, right_point, rectification_params, resize_scale, num
        )
    
    def create_comprehensive_points_data(
        self,
        pair_name: str,
        parameter_path: Path,
        rectification_engine,
        resize_scale: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Create comprehensive points data including all reference points and metadata.
        
        Args:
            pair_name: Name of the image pair
            parameter_path: Path to parameter directory
            rectification_engine: Rectification engine instance
            resize_scale: Resize scaling factors
            
        Returns:
            Dict[str, Any]: Comprehensive points data with metadata
        """
        # Process basic rut points
        points_data = self.process_rut_points(
            pair_name, parameter_path, rectification_engine, resize_scale
        )
        
        # Add metadata
        points_data['metadata'] = {
            'pair_name': pair_name,
            'num_interpolated_points': len(points_data['rectified_interpolated_points']),
            'resize_scale_applied': resize_scale,
            'optical_center': (rectification_engine.K_left[0, 2], rectification_engine.K_left[1, 2]),
            'processing_method': 'rectification_point_processor_v1.0'
        }
        
        # Add endpoint information
        if points_data['rectified_interpolated_points']:
            points_data['metadata']['endpoints'] = {
                'left': points_data['rectified_interpolated_points'][0],
                'right': points_data['rectified_interpolated_points'][-1]
            }
        
        return points_data
    
    def validate_points_data(self, points_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processed points data.
        
        Args:
            points_data: Processed points data to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check required keys
        required_keys = [
            'rectified_interpolated_points',
            'optical_ref_x_points', 
            'optical_ref_y_points',
            'original_rut_points'
        ]
        
        for key in required_keys:
            if key not in points_data:
                validation_results['errors'].append(f"Missing required key: {key}")
                validation_results['valid'] = False
        
        if not validation_results['valid']:
            return validation_results
        
        # Validate point counts
        main_points = points_data['rectified_interpolated_points']
        ref_x_points = points_data['optical_ref_x_points']
        ref_y_points = points_data['optical_ref_y_points']
        
        if len(main_points) != len(ref_x_points) or len(main_points) != len(ref_y_points):
            validation_results['warnings'].append(
                f"Point count mismatch: main={len(main_points)}, "
                f"ref_x={len(ref_x_points)}, ref_y={len(ref_y_points)}"
            )
        
        # Validate point coordinates
        try:
            for point_type, points in [
                ('main', main_points),
                ('ref_x', ref_x_points),
                ('ref_y', ref_y_points)
            ]:
                if points:
                    PointProcessor.validate_point_coordinates(np.array(points))
        except ValueError as e:
            validation_results['errors'].append(f"Invalid {point_type} coordinates: {e}")
            validation_results['valid'] = False
        
        # Log validation results
        if validation_results['warnings']:
            self.logger.warning(f"Points validation warnings: {validation_results['warnings']}")
        
        if validation_results['errors']:
            self.logger.error(f"Points validation errors: {validation_results['errors']}")
        
        return validation_results
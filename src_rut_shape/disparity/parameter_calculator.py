"""
Parameter calculation utilities for disparity processing.

This module handles automatic calculation and optimization of disparity-related
parameters based on camera calibration and target depth requirements.
"""

import numpy as np
import math
import logging
from typing import Dict, Any, Tuple, Optional, List

from utils.logger_config import get_logger

logger = get_logger(__name__)


class DisparityParameterCalculator:
    """Calculates optimal parameters for disparity computation."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def calculate_sgbm_parameters(
        self,
        camera_matrices: Tuple[np.ndarray, np.ndarray],
        translation_vector: np.ndarray,
        target_depth: float,
        min_depth: float,
        max_depth: float,
        resize_scale: float = 1.0,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal SGBM parameters based on camera setup and requirements.
        
        Args:
            camera_matrices: (K_left, K_right) camera intrinsic matrices
            translation_vector: Translation vector between cameras
            target_depth: Target depth in meters
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters
            resize_scale: Scale factor applied to images
            config_overrides: Optional parameter overrides from configuration
            
        Returns:
            Dict[str, Any]: Calculated SGBM parameters
        """
        K_left, K_right = camera_matrices
        
        # Calculate average focal length in pixels
        focal_length_pixels = self._calculate_average_focal_length(K_left, K_right)
        
        # Calculate baseline in meters
        baseline_meters = self._calculate_baseline(translation_vector)
        
        # Calculate disparity range
        disparity_range = self._calculate_disparity_range(
            focal_length_pixels, baseline_meters, min_depth, max_depth, target_depth
        )
        
        # # Apply resize scaling
        # if resize_scale != 1.0:
        #     disparity_range = self._apply_resize_scaling(disparity_range, resize_scale)
        
        # Calculate SGBM parameters
        sgbm_params = self._calculate_sgbm_params(disparity_range, config_overrides, resize_scale)
        
        # Add metadata
        sgbm_params.update({
            'focal_length_pixels': focal_length_pixels,
            'baseline_meters': baseline_meters,
            'resize_scale': resize_scale,
            'depth_range': {
                'min_depth': min_depth,
                'max_depth': max_depth,
                'target_depth': target_depth
            }
        })
        
        self.logger.info(f"Calculated SGBM parameters: "
                        f"numDisp={sgbm_params['num_disparities']}, "
                        f"blockSize={sgbm_params['block_size']}, "
                        f"minDisp={sgbm_params['min_disparity']}")
        
        return sgbm_params
    
    def _calculate_average_focal_length(
        self, 
        K_left: np.ndarray, 
        K_right: np.ndarray
    ) -> float:
        """
        Calculate average focal length from camera matrices.
        
        Args:
            K_left: Left camera intrinsic matrix
            K_right: Right camera intrinsic matrix
            
        Returns:
            float: Average focal length in pixels
        """
        # Extract focal lengths
        fx_left, fy_left = K_left[0, 0], K_left[1, 1]
        fx_right, fy_right = K_right[0, 0], K_right[1, 1]
        
        # Calculate average
        avg_focal_length = (fx_left + fy_left + fx_right + fy_right) / 4
        
        self.logger.debug(f"Average focal length: {avg_focal_length:.2f} pixels")
        return avg_focal_length
    
    def _calculate_baseline(self, translation_vector: np.ndarray) -> float:
        """
        Calculate stereo baseline from translation vector.
        
        Args:
            translation_vector: Translation vector between cameras
            
        Returns:
            float: Baseline in meters
        """
        # For horizontal stereo setup, baseline is primarily the X component
        baseline = abs(translation_vector[0])
        
        self.logger.debug(f"Stereo baseline: {baseline:.4f} meters")
        return baseline
    
    def _calculate_disparity_range(
        self,
        focal_length: float,
        baseline: float,
        min_depth: float,
        max_depth: float,
        target_depth: float
    ) -> Dict[str, float]:
        """
        Calculate disparity range based on depth requirements.
        
        Args:
            focal_length: Focal length in pixels
            baseline: Baseline in meters
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters
            target_depth: Target depth in meters
            
        Returns:
            Dict[str, float]: Disparity range information
        """
        # Convert depths to disparities: disparity = (focal_length * baseline) / depth
        disp_at_min_depth = focal_length * baseline / min_depth
        disp_at_max_depth = focal_length * baseline / max_depth
        disp_at_target_depth = focal_length * baseline / target_depth
        
        # Calculate range around target depth
        range_val = min(
            disp_at_target_depth - disp_at_max_depth,
            disp_at_min_depth - disp_at_target_depth
        )
        
        disparity_range = {
            'min_disparity': max(0, disp_at_target_depth - range_val),
            'max_disparity': disp_at_target_depth + range_val,
            'target_disparity': disp_at_target_depth,
            'range_around_target': range_val
        }
        
        self.logger.debug(f"Disparity range: {disparity_range}")
        return disparity_range
    
    def _apply_resize_scaling(
        self, 
        disparity_range: Dict[str, float], 
        resize_scale: float
    ) -> Dict[str, float]:
        """
        Apply resize scaling to disparity values.
        
        Args:
            disparity_range: Original disparity range
            resize_scale: Resize scale factor
            
        Returns:
            Dict[str, float]: Scaled disparity range
        """
        if resize_scale == 1.0:
            return disparity_range
        
        scaled_range = {}
        for key, value in disparity_range.items():
            scaled_range[key] = value * resize_scale
        
        self.logger.debug(f"Applied resize scaling {resize_scale}: {scaled_range}")
        return scaled_range
    
    def _calculate_sgbm_params(
        self, 
        disparity_range: Dict[str, float],
        config_overrides: Optional[Dict[str, Any]] = None,
        resize_scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate SGBM parameters from disparity range and configuration.
        
        Args:
            disparity_range: Calculated disparity range
            config_overrides: Optional parameter overrides
            
        Returns:
            Dict[str, Any]: SGBM parameters
        """
        config_overrides = config_overrides or {}
        
        # Calculate num_disparities (must be divisible by 16)
        if 'NUM_DISPARITIES' in config_overrides and config_overrides['NUM_DISPARITIES']:
            # Use configured value with scaling
            num_disparities = int((config_overrides['NUM_DISPARITIES']*resize_scale // 16) * 16)
        else:
            # Calculate from disparity range
            disparity_span = disparity_range['max_disparity'] - disparity_range['min_disparity']
            num_disparities = int(math.ceil(disparity_span / 16)) * 16
        
        # Ensure minimum reasonable value
        num_disparities = max(num_disparities, 32)
        
        # Calculate min_disparity
        min_disparity = config_overrides.get('MIN_DISPARITY', 0)
        
        # Calculate block_size (must be odd)
        if 'BLOCKSIZE' in config_overrides and config_overrides['BLOCKSIZE'] is not None:
            block_size = config_overrides['BLOCKSIZE'] * 2 + 1
        else:
            block_size = 5  # Default value
        
        # Ensure block_size is odd and reasonable
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, min(block_size, 21))
        
        # SGBM mode
        use_fast_mode = config_overrides.get('SGBM_MODE_FAST', 'True') == 'True'
        
        return {
            'num_disparities': num_disparities,
            'min_disparity': min_disparity,
            'block_size': block_size,
            'use_fast_mode': use_fast_mode
        }
    
    def calculate_depth_from_disparity(
        self,
        disparity: np.ndarray,
        focal_length: float,
        baseline: float
    ) -> np.ndarray:
        """
        Convert disparity map to depth map.
        
        Args:
            disparity: Disparity map (in pixels)
            focal_length: Focal length in pixels
            baseline: Baseline in meters
            
        Returns:
            np.ndarray: Depth map in meters
        """
        # Avoid division by zero
        valid_disparity = disparity > 0
        depth = np.zeros_like(disparity, dtype=np.float32)
        
        # Calculate depth: depth = (focal_length * baseline) / disparity
        depth[valid_disparity] = (focal_length * baseline) / disparity[valid_disparity]
        
        return depth
    
    def calculate_disparity_visualization_range(
        self,
        focal_length: float,
        baseline: float,
        target_depth: float,
        min_depth: float,
        max_depth: float
    ) -> Dict[str, float]:
        """
        Calculate optimal disparity range for visualization.
        Matches the original disparity.py implementation exactly.
        
        Args:
            focal_length: Focal length in pixels
            baseline: Baseline in meters
            target_depth: Target depth in meters
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters
            
        Returns:
            Dict[str, float]: Visualization range parameters
        """
        # Calculate disparities using the same formula as original disparity.py
        # Note: depths are converted to mm in original (multiply by 1000)
        disp_farthest = math.floor(focal_length * baseline / (max_depth * 1000))
        disp_closest = math.ceil(focal_length * baseline / (min_depth * 1000) / 10) * 10
        disp_target = (math.ceil(focal_length * baseline / (target_depth * 1000) / 16)) * 16
        
        # Calculate symmetric range around target (same as original)
        range_val = min(disp_target - disp_farthest, disp_closest - disp_target)
        disp_farthest_final = disp_target - range_val
        disp_closest_final = disp_target + range_val
        
        return {
            'target_disparity': disp_target,
            'min_display': disp_farthest_final,
            'max_display': disp_closest_final,
            'range_symmetric': range_val
        }
    
    def validate_parameters(
        self, 
        parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate calculated parameters for reasonableness.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True
        
        # Check num_disparities
        num_disp = parameters.get('num_disparities', 0)
        if num_disp <= 0 or num_disp % 16 != 0:
            warnings.append(f"Invalid num_disparities: {num_disp} (must be positive and divisible by 16)")
            is_valid = False
        
        if num_disp > 1000:
            warnings.append(f"Very large num_disparities: {num_disp} (may be slow)")
        
        # Check block_size
        block_size = parameters.get('block_size', 0)
        if block_size <= 0 or block_size % 2 == 0:
            warnings.append(f"Invalid block_size: {block_size} (must be positive and odd)")
            is_valid = False
        
        if block_size > 15:
            warnings.append(f"Large block_size: {block_size} (may reduce accuracy)")
        
        # Check min_disparity
        min_disp = parameters.get('min_disparity', 0)
        if min_disp < 0:
            warnings.append(f"Invalid min_disparity: {min_disp} (must be non-negative)")
            is_valid = False
        
        return is_valid, warnings
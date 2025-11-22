"""
Stereo rectification engine for computer vision applications.

This module contains the core stereo rectification algorithms and processing
logic, including optimal principal point calculation and rectification map generation.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from utils.stereo_math import StereoMath, GeometryValidator
from utils.image_processing import ImageProcessor

logger = logging.getLogger(__name__)


class StereoRectificationEngine:
    """Core engine for stereo image rectification processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize rectification engine.
        
        Args:
            config: Configuration dictionary with rectification parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Rectification parameters
        self.K_left: Optional[np.ndarray] = None
        self.K_right: Optional[np.ndarray] = None
        self.d_left: Optional[np.ndarray] = None
        self.d_right: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None
        
        # Computed rectification matrices
        self.R1: Optional[np.ndarray] = None
        self.R2: Optional[np.ndarray] = None
        self.P1: Optional[np.ndarray] = None
        self.P2: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        
        # Adjusted matrices for optimal principal points
        self.P1_adjusted: Optional[np.ndarray] = None
        self.P2_adjusted: Optional[np.ndarray] = None
        
        # Rectification configuration
        self.image_size: Optional[Tuple[int, int]] = None
        self.rectified_size: Optional[Tuple[int, int]] = None
        self.optimal_principal_offsets: Optional[Tuple[float, float]] = None
    
    def load_calibration_parameters(
        self,
        K_left: np.ndarray,
        d_left: np.ndarray,
        K_right: np.ndarray,
        d_right: np.ndarray,
        R: np.ndarray,
        T: np.ndarray
    ) -> None:
        """
        Load stereo calibration parameters.
        
        Args:
            K_left: Left camera intrinsic matrix (3x3)
            d_left: Left camera distortion coefficients
            K_right: Right camera intrinsic matrix (3x3)
            d_right: Right camera distortion coefficients
            R: Rotation matrix between cameras (3x3)
            T: Translation vector between cameras (3x1 or 3,)
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate all parameters
        StereoMath.validate_camera_matrix(K_left, "K_left")
        StereoMath.validate_camera_matrix(K_right, "K_right")
        StereoMath.validate_distortion_coefficients(d_left, "d_left")
        StereoMath.validate_distortion_coefficients(d_right, "d_right")
        StereoMath.validate_rotation_matrix(R, "R")
        StereoMath.validate_translation_vector(T, "T")
        
        # Store parameters
        self.K_left = K_left.copy()
        self.K_right = K_right.copy()
        self.d_left = d_left.copy()
        self.d_right = d_right.copy()
        self.R = R.copy()
        self.T = T.copy()
        
        self.logger.info("Calibration parameters loaded and validated")
    
    def validate_stereo_setup(self, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Validate the complete stereo setup.
        
        Args:
            image_size: (width, height) of input images
            
        Returns:
            Dict[str, Any]: Validation results
        """
        if any(param is None for param in [self.K_left, self.K_right, self.R, self.T]):
            raise ValueError("Calibration parameters not loaded")
            
        return GeometryValidator.validate_stereo_geometry(
            self.K_left, self.K_right, self.R, self.T, image_size
        )
    
    def compute_rectification_matrices(
        self, 
        image_size: Tuple[int, int],
        alpha: float = 0.0
    ) -> None:
        """
        Compute standard rectification matrices using OpenCV.
        
        Args:
            image_size: (width, height) of input images
            alpha: Free scaling parameter (0=only valid pixels, 1=all pixels)
        """
        if any(param is None for param in [self.K_left, self.K_right, self.R, self.T]):
            raise ValueError("Calibration parameters not loaded")
            
        self.image_size = image_size
        
        # Compute rectification matrices
        (self.R1, self.R2, self.P1, self.P2, self.Q, 
         roi_left, roi_right) = cv2.stereoRectify(
            self.K_left, self.d_left,
            self.K_right, self.d_right,
            image_size, self.R, self.T,
            alpha=alpha
        )
        
        self.logger.info(f"Computed rectification matrices for size {image_size}, alpha={alpha}")
        
        # Log matrix information
        self.logger.debug(f"P1 focal lengths: fx={self.P1[0,0]:.2f}, fy={self.P1[1,1]:.2f}")
        self.logger.debug(f"P1 principal point: cx={self.P1[0,2]:.2f}, cy={self.P1[1,2]:.2f}")
        
        baseline = StereoMath.calculate_baseline_from_projection_matrices(self.P1, self.P2)
        self.logger.debug(f"Computed baseline: {baseline:.4f}")
    
    def calculate_optimal_rectification_size(self) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """
        Calculate optimal rectified image size and principal points.
        
        Returns:
            Tuple containing:
                - (width, height): Optimal rectified image size
                - (cx, cy): Optimal principal points
        """
        if self.image_size is None or self.R1 is None or self.P1 is None:
            raise ValueError("Rectification matrices not computed")
            
        width, height = self.image_size
        
        # Calculate corner projections to determine bounding box
        corners = np.array([
            [0, 0], [width-1, 0], 
            [0, height-1], [width-1, height-1]
        ], dtype=np.float32)
        
        x_coords, y_coords = [], []
        
        for corner in corners:
            # Transform corner through rectification pipeline
            corner_homo = np.array([corner[0], corner[1], 1.0])
            normalized = np.linalg.inv(self.K_left) @ corner_homo
            rotated = self.R1 @ normalized
            projected = self.P1 @ np.array([rotated[0], rotated[1], rotated[2], 1.0])
            
            if projected[2] != 0:
                x_coords.append(projected[0] / projected[2])
                y_coords.append(projected[1] / projected[2])
        
        # Calculate bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Optimal size
        optimal_width = int(np.ceil(x_max - x_min))
        optimal_height = int(np.ceil(y_max - y_min))
        
        # Optimal principal points (center of projected corners)
        optimal_cx = np.mean(x_coords)
        optimal_cy = np.mean(y_coords)
        
        self.logger.info(f"Calculated optimal size: {optimal_width}x{optimal_height}")
        self.logger.info(f"Calculated optimal principal points: cx={optimal_cx:.2f}, cy={optimal_cy:.2f}")
        
        return (optimal_width, optimal_height), (x_min, y_min)
    
    def adjust_projection_matrices_for_optimal_size(
        self,
        optimal_size: Tuple[int, int],
        optimal_principal_offsets: Tuple[float, float]
    ) -> None:
        """
        Adjust projection matrices for optimal rectified image size.
        
        Args:
            optimal_size: (width, height) of optimal rectified image
            optimal_principal_points: (cx, cy) optimal principal points
        """
        if self.P1 is None or self.P2 is None:
            raise ValueError("Projection matrices not computed")
            
        # Calculate offset from current to optimal principal points
        current_cx, current_cy = self.P1[0, 2], self.P1[1, 2]
        offset_x, offset_y = optimal_principal_offsets # x_min, y_min
        
        # offset_x = current_cx - optimal_cx
        # offset_y = current_cy - optimal_cy
        
        # Adjust projection matrices
        self.P1_adjusted = self.P1.copy()
        self.P2_adjusted = self.P2.copy()
        
        self.P1_adjusted[0, 2] -= offset_x  # Adjust cx
        self.P1_adjusted[1, 2] -= offset_y  # Adjust cy
        self.P2_adjusted[0, 2] -= offset_x  # Adjust cx
        self.P2_adjusted[1, 2] -= offset_y  # Adjust cy
        
        # Adjust baseline term in P2
        self.P2_adjusted[0, 3] -= offset_x * self.P2_adjusted[0, 0] / self.P1_adjusted[0, 0]
        
        self.logger.info(f"Adjusted projection matrices with offset: ({offset_x:.2f}, {offset_y:.2f})")
    
    def setup_rectification(
        self, 
        image_size: Tuple[int, int],
        size_mode: str = 'default',
        alpha: float = 0.0
    ) -> None:
        """
        Complete rectification setup with specified parameters.
        
        Args:
            image_size: (width, height) of input images
            size_mode: 'default' (original size) or 'auto' (optimal size)
            alpha: Free scaling parameter for rectification
        """
        # Validate stereo setup
        validation_results = self.validate_stereo_setup(image_size)
        if not validation_results['valid']:
            raise ValueError(f"Invalid stereo setup: {validation_results['errors']}")
            
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                self.logger.warning(warning)
        
        # Compute base rectification matrices
        self.compute_rectification_matrices(image_size, alpha)
        
        # Determine rectified image size and principal points
        if size_mode == 'auto':
            optimal_size, optimal_principal_offsets = self.calculate_optimal_rectification_size()
            self.rectified_size = optimal_size
            self.optimal_principal_offsets = optimal_principal_offsets
            
            # Adjust projection matrices for optimal size
            self.adjust_projection_matrices_for_optimal_size(optimal_size, optimal_principal_offsets)
            
        else:  # default mode
            self.rectified_size = image_size
            
            # Still calculate optimal principal points for consistency
            _, optimal_principal_offsets = self.calculate_optimal_rectification_size()
            self.optimal_principal_offsets = optimal_principal_offsets
            
            # Use original projection matrices
            self.P1_adjusted = self.P1.copy()
            self.P2_adjusted = self.P2.copy()
        
        self.logger.info(f"Rectification setup complete: size_mode={size_mode}, "
                        f"rectified_size={self.rectified_size}")
    
    def generate_rectification_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate rectification maps for image remapping.
        
        Returns:
            Tuple containing (map1_left, map2_left, map1_right, map2_right)
        """
        if (self.rectified_size is None or self.R1 is None or self.R2 is None or
            self.P1_adjusted is None or self.P2_adjusted is None):
            raise ValueError("Rectification not properly set up")
        
        # Generate rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.K_left, self.d_left, self.R1, self.P1_adjusted,
            self.rectified_size, cv2.CV_32FC1
        )
        
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.K_right, self.d_right, self.R2, self.P2_adjusted,
            self.rectified_size, cv2.CV_32FC1
        )
        
        self.logger.info(f"Generated rectification maps for size {self.rectified_size}")
        
        return map1_left, map2_left, map1_right, map2_right
    
    def rectify_image_pair(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify a stereo image pair.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rectified_left, rectified_right)
        """
        # Validate input images
        ImageProcessor.validate_image_pair(left_image, right_image)
        
        # Generate rectification maps
        map1_left, map2_left, map1_right, map2_right = self.generate_rectification_maps()
        
        # Apply rectification
        rectified_left = cv2.remap(left_image, map1_left, map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_image, map1_right, map2_right, cv2.INTER_LINEAR)
        
        self.logger.info(f"Rectified image pair to size {self.rectified_size}")
        
        return rectified_left, rectified_right
    
    def get_rectification_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about current rectification setup.
        
        Returns:
            Dict[str, Any]: Rectification information
        """
        if self.P1_adjusted is None or self.P2_adjusted is None:
            return {'setup_complete': False}
        
        baseline = StereoMath.calculate_baseline_from_projection_matrices(
            self.P1_adjusted, self.P2_adjusted
        )
        
        info = {
            'setup_complete': True,
            'original_size': self.image_size,
            'rectified_size': self.rectified_size,
            'optimal_principal_offsets': self.optimal_principal_offsets,
            'baseline': baseline,
            'focal_lengths': {
                'fx': float(self.P1_adjusted[0, 0]),
                'fy': float(self.P1_adjusted[1, 1])
            },
            'principal_points': {
                'cx': float(self.P1_adjusted[0, 2]),
                'cy': float(self.P1_adjusted[1, 2])
            }
        }
        
        if self.image_size and self.rectified_size:
            info['size_change_ratio'] = {
                'width': self.rectified_size[0] / self.image_size[0],
                'height': self.rectified_size[1] / self.image_size[1]
            }
        
        return info
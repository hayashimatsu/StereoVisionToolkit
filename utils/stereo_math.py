"""
Stereo vision mathematical utilities.

This module provides mathematical functions and validations for stereo vision
applications, including matrix operations, geometric calculations, and validations.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class StereoMath:
    """Mathematical utilities for stereo vision calculations."""
    
    @staticmethod
    def validate_camera_matrix(K: np.ndarray, matrix_name: str = "K") -> bool:
        """
        Validate camera intrinsic matrix.
        
        Args:
            K: 3x3 camera matrix
            matrix_name: Name for error messages
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If matrix is invalid
        """
        if K.shape != (3, 3):
            raise ValueError(f"{matrix_name} must be 3x3, got {K.shape}")
            
        # Check focal lengths are positive
        fx, fy = K[0, 0], K[1, 1]
        if fx <= 0 or fy <= 0:
            raise ValueError(f"{matrix_name} focal lengths must be positive: fx={fx}, fy={fy}")
            
        # Check bottom row
        if not np.allclose(K[2, :], [0, 0, 1]):
            raise ValueError(f"{matrix_name} bottom row must be [0, 0, 1]")
            
        return True
    
    @staticmethod
    def validate_distortion_coefficients(d: np.ndarray, coeff_name: str = "d") -> bool:
        """
        Validate distortion coefficients.
        
        Args:
            d: Distortion coefficients array
            coeff_name: Name for error messages
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If coefficients are invalid
        """
        if d.ndim != 1:
            raise ValueError(f"{coeff_name} must be 1D array, got shape {d.shape}")
            
        if len(d) not in [4, 5, 8, 12, 14]:
            logger.warning(f"{coeff_name} has unusual length {len(d)}, "
                          f"expected 4, 5, 8, 12, or 14")
            
        if np.any(np.isnan(d)) or np.any(np.isinf(d)):
            raise ValueError(f"{coeff_name} contains NaN or infinite values")
            
        return True
    
    @staticmethod
    def validate_rotation_matrix(R: np.ndarray, matrix_name: str = "R") -> bool:
        """
        Validate rotation matrix.
        
        Args:
            R: 3x3 rotation matrix
            matrix_name: Name for error messages
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If matrix is invalid
        """
        if R.shape != (3, 3):
            raise ValueError(f"{matrix_name} must be 3x3, got {R.shape}")
            
        # Check orthogonality: R @ R.T should be identity
        identity_check = R @ R.T
        if not np.allclose(identity_check, np.eye(3), atol=1e-6):
            raise ValueError(f"{matrix_name} is not orthogonal")
            
        # Check determinant is 1 (proper rotation, not reflection)
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-6):
            raise ValueError(f"{matrix_name} determinant must be 1, got {det}")
            
        return True
    
    @staticmethod
    def validate_translation_vector(T: np.ndarray, vector_name: str = "T") -> bool:
        """
        Validate translation vector.
        
        Args:
            T: Translation vector
            vector_name: Name for error messages
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If vector is invalid
        """
        if T.shape not in [(3,), (3, 1)]:
            raise ValueError(f"{vector_name} must have shape (3,) or (3,1), got {T.shape}")
            
        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            raise ValueError(f"{vector_name} contains NaN or infinite values")
            
        # Check if baseline is reasonable (not too small)
        baseline = np.linalg.norm(T)
        if baseline < 1e-6:
            logger.warning(f"{vector_name} baseline is very small: {baseline}")
            
        return True
    
    @staticmethod
    def validate_projection_matrix(P: np.ndarray, matrix_name: str = "P") -> bool:
        """
        Validate projection matrix.
        
        Args:
            P: 3x4 projection matrix
            matrix_name: Name for error messages
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If matrix is invalid
        """
        if P.shape != (3, 4):
            raise ValueError(f"{matrix_name} must be 3x4, got {P.shape}")
            
        # Extract focal lengths and check they're positive
        fx, fy = P[0, 0], P[1, 1]
        if fx <= 0 or fy <= 0:
            raise ValueError(f"{matrix_name} focal lengths must be positive: fx={fx}, fy={fy}")
            
        return True
    
    @staticmethod
    def calculate_baseline_from_projection_matrices(
        P1: np.ndarray, 
        P2: np.ndarray
    ) -> float:
        """
        Calculate baseline from projection matrices.
        
        Args:
            P1: Left camera projection matrix
            P2: Right camera projection matrix
            
        Returns:
            float: Baseline distance
        """
        # For horizontal stereo setup, baseline is primarily in x-direction
        # Extract the effective baseline from P2 matrix
        baseline = abs(P2[0, 3] / P2[0, 0])
        
        if baseline < 1e-6:
            logger.warning(f"Very small baseline detected: {baseline}")
            
        return baseline
    
    @staticmethod
    def check_focal_length_consistency(
        P1: np.ndarray, 
        P2: np.ndarray, 
        tolerance: float = 1e-3
    ) -> bool:
        """
        Check if focal lengths are consistent between projection matrices.
        
        Args:
            P1: Left camera projection matrix
            P2: Right camera projection matrix
            tolerance: Tolerance for focal length differences
            
        Returns:
            bool: True if focal lengths are consistent
        """
        fx1, fy1 = P1[0, 0], P1[1, 1]
        fx2, fy2 = P2[0, 0], P2[1, 1]
        
        fx_diff = abs(fx1 - fx2)
        fy_diff = abs(fy1 - fy2)
        
        if fx_diff > tolerance or fy_diff > tolerance:
            logger.warning(f"Focal length inconsistency: "
                          f"fx_diff={fx_diff:.6f}, fy_diff={fy_diff:.6f}")
            return False
            
        # Check if fx and fy are equal within each matrix (rectified assumption)
        if abs(fx1 - fy1) > tolerance:
            logger.warning(f"P1 focal lengths differ: fx={fx1:.2f}, fy={fy1:.2f}")
            
        if abs(fx2 - fy2) > tolerance:
            logger.warning(f"P2 focal lengths differ: fx={fx2:.2f}, fy={fy2:.2f}")
            
        return True
    
    @staticmethod
    def validate_q_matrix_parameters(
        fx: float, 
        fy: float, 
        cx: float, 
        cy: float, 
        baseline: float
    ) -> bool:
        """
        Validate Q matrix calculation parameters.
        
        Args:
            fx: Focal length in x direction
            fy: Focal length in y direction  
            cx: Principal point x coordinate
            cy: Principal point y coordinate
            baseline: Stereo baseline
            
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Focal lengths must be positive: fx={fx}, fy={fy}")
            
        if abs(baseline) < 1e-6:
            raise ValueError(f"Baseline too small: {baseline}")
            
        if abs(fx - fy) > 1e-3:
            logger.warning(f"Focal lengths differ significantly: fx={fx:.2f}, fy={fy:.2f}")
            
        return True
    
    @staticmethod
    def calculate_reprojection_error(
        points_3d: np.ndarray,
        points_2d: np.ndarray, 
        projection_matrix: np.ndarray
    ) -> float:
        """
        Calculate reprojection error for 3D to 2D point correspondences.
        
        Args:
            points_3d: 3D points in homogeneous coordinates (Nx4)
            points_2d: Corresponding 2D points (Nx2)
            projection_matrix: 3x4 projection matrix
            
        Returns:
            float: RMS reprojection error
        """
        if points_3d.shape[0] != points_2d.shape[0]:
            raise ValueError("Number of 3D and 2D points must match")
            
        # Project 3D points to 2D
        projected_homogeneous = projection_matrix @ points_3d.T
        projected_2d = (projected_homogeneous[:2] / projected_homogeneous[2]).T
        
        # Calculate Euclidean distances
        errors = np.linalg.norm(projected_2d - points_2d, axis=1)
        
        # Return RMS error
        rms_error = np.sqrt(np.mean(errors**2))
        
        return rms_error


class GeometryValidator:
    """Validates geometric consistency in stereo vision setups."""
    
    @staticmethod
    def validate_stereo_geometry(
        K_left: np.ndarray,
        K_right: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of stereo geometry.
        
        Args:
            K_left: Left camera matrix
            K_right: Right camera matrix
            R: Rotation matrix between cameras
            T: Translation vector between cameras
            image_size: (width, height) of images
            
        Returns:
            Dict[str, Any]: Validation results and metrics
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Validate individual matrices
            StereoMath.validate_camera_matrix(K_left, "K_left")
            StereoMath.validate_camera_matrix(K_right, "K_right")
            StereoMath.validate_rotation_matrix(R, "R")
            StereoMath.validate_translation_vector(T, "T")
            
            # Calculate metrics
            baseline = np.linalg.norm(T)
            results['metrics']['baseline'] = baseline
            
            # Check focal length consistency
            fx_left, fy_left = K_left[0, 0], K_left[1, 1]
            fx_right, fy_right = K_right[0, 0], K_right[1, 1]
            
            focal_diff = max(abs(fx_left - fx_right), abs(fy_left - fy_right))
            results['metrics']['focal_length_difference'] = focal_diff
            
            if focal_diff > 10:  # pixels
                results['warnings'].append(f"Large focal length difference: {focal_diff:.2f} pixels")
                
            # Check principal point positions
            cx_left, cy_left = K_left[0, 2], K_left[1, 2]
            cx_right, cy_right = K_right[0, 2], K_right[1, 2]
            
            width, height = image_size
            
            if not (0 <= cx_left <= width and 0 <= cx_right <= width):
                results['warnings'].append("Principal points outside image width bounds")
                
            if not (0 <= cy_left <= height and 0 <= cy_right <= height):
                results['warnings'].append("Principal points outside image height bounds")
                
            # Check baseline to focal length ratio
            avg_focal = (fx_left + fx_right) / 2
            baseline_focal_ratio = baseline / avg_focal
            results['metrics']['baseline_focal_ratio'] = baseline_focal_ratio
            
            if baseline_focal_ratio < 0.1:
                results['warnings'].append(f"Small baseline/focal ratio: {baseline_focal_ratio:.3f}")
                
        except ValueError as e:
            results['valid'] = False
            results['errors'].append(str(e))
            
        return results
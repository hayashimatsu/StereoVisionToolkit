"""
Matrix calculation utilities for stereo rectification.

This module handles Q matrix calculations, projection matrix adjustments,
and matrix consistency validations for stereo vision applications.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

from utils.stereo_math import StereoMath

logger = logging.getLogger(__name__)


class MatrixCalculator:
    """Handles matrix calculations for stereo rectification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_q_matrix_from_projection_matrices(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        validate_params: bool = True
    ) -> np.ndarray:
        """
        Calculate Q matrix from rectified projection matrices.
        
        This method derives the Q matrix parameters directly from the final
        rectified projection matrices, ensuring geometric consistency.
        
        Args:
            P1: Left camera projection matrix (3x4)
            P2: Right camera projection matrix (3x4)
            validate_params: Whether to validate input parameters
            
        Returns:
            np.ndarray: 4x4 Q matrix for disparity-to-depth conversion
            
        Raises:
            ValueError: If projection matrices are invalid
        """
        if validate_params:
            StereoMath.validate_projection_matrix(P1, "P1")
            StereoMath.validate_projection_matrix(P2, "P2")
        
        # Extract parameters from P1 matrix
        fx = P1[0, 0]  # focal length x
        fy = P1[1, 1]  # focal length y (should equal fx for rectified)
        cx = P1[0, 2]  # principal point x
        cy = P1[1, 2]  # principal point y
        
        # Calculate baseline from P2 matrix
        # For horizontal stereo setup: baseline = -P2[0,3] / P2[0,0]
        baseline_x = P2[0, 3] / P2[0, 0]  # This is typically negative
        
        if validate_params:
            StereoMath.validate_q_matrix_parameters(fx, fy, cx, cy, abs(baseline_x))
        
        # Construct Q matrix according to OpenCV convention
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0] = 1.0
        Q[1, 1] = 1.0
        Q[0, 3] = -cx                    # -cx
        Q[1, 3] = -cy                    # -cy
        Q[2, 3] = fx                     # focal length
        Q[3, 2] = -1.0 / baseline_x      # -1/baseline (note: baseline_x is negative)
        Q[3, 3] = 0.0                    # (cx_left - cx_right) / baseline (should be 0 for rectified)
        
        # Log calculation details
        self.logger.info(f"Q matrix calculated from projection matrices:")
        self.logger.info(f"  Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
        self.logger.info(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
        self.logger.info(f"  Baseline: {abs(baseline_x):.4f}")
        self.logger.info(f"  Q matrix elements: Q[0,3]={Q[0,3]:.2f}, Q[1,3]={Q[1,3]:.2f}")
        self.logger.info(f"  Q matrix elements: Q[2,3]={Q[2,3]:.2f}, Q[3,2]={Q[3,2]:.6f}")
        
        return Q
    
    def recalculate_q_matrix_with_resize(
        self,
        original_P1: np.ndarray,
        original_P2: np.ndarray,
        resize_scale: Tuple[float, float]
    ) -> np.ndarray:
        """
        Recalculate Q matrix accounting for image resize operations.
        
        Args:
            original_P1: Original left projection matrix
            original_P2: Original right projection matrix
            resize_scale: (scale_x, scale_y) resize scaling factors
            
        Returns:
            np.ndarray: Q matrix adjusted for resize
        """
        scale_x, scale_y = resize_scale
        
        if scale_x == 1.0 and scale_y == 1.0:
            # No resize applied
            return self.calculate_q_matrix_from_projection_matrices(original_P1, original_P2)
        
        # Adjust projection matrices for resize
        P1_resized = original_P1.copy()
        P2_resized = original_P2.copy()
        
        # Scale principal points and baseline
        P1_resized[0, 2] *= scale_x  # cx
        P1_resized[1, 2] *= scale_y  # cy
        P2_resized[0, 2] *= scale_x  # cx
        P2_resized[1, 2] *= scale_y  # cy
        P2_resized[0, 3] *= scale_x  # baseline term
        
        self.logger.info(f"Recalculated Q matrix with resize scale: ({scale_x:.3f}, {scale_y:.3f})")
        
        return self.calculate_q_matrix_from_projection_matrices(P1_resized, P2_resized)
    
    def adjust_projection_matrices_for_resize(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        resize_scale: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust projection matrices for image resize operations.
        
        When images are resized, the projection matrices need to be adjusted
        to maintain correct geometric relationships.
        
        Args:
            P1: Left camera projection matrix
            P2: Right camera projection matrix
            resize_scale: (scale_x, scale_y) resize scaling factors
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Adjusted (P1, P2) matrices
        """
        scale_x, scale_y = resize_scale
        
        if scale_x == 1.0 and scale_y == 1.0:
            return P1.copy(), P2.copy()
        
        P1_adjusted = P1.copy()
        P2_adjusted = P2.copy()
        
        # Adjust principal points
        P1_adjusted[0, 2] *= scale_x  # cx
        P1_adjusted[1, 2] *= scale_y  # cy
        P2_adjusted[0, 2] *= scale_x  # cx
        P2_adjusted[1, 2] *= scale_y  # cy
        
        # Adjust baseline term in P2 matrix
        P2_adjusted[0, 3] *= scale_x  # fx * Tx
        
        self.logger.info(f"Adjusted projection matrices for resize: ({scale_x:.3f}, {scale_y:.3f})")
        self.logger.info(f"New P1 principal point: cx={P1_adjusted[0,2]:.2f}, cy={P1_adjusted[1,2]:.2f}")
        
        return P1_adjusted, P2_adjusted
    
    def validate_matrix_consistency(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        Q: np.ndarray,
        tolerance: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Validate consistency between projection matrices and Q matrix.
        
        Args:
            P1: Left projection matrix
            P2: Right projection matrix
            Q: Q matrix
            tolerance: Tolerance for consistency checks
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results = {
            'consistent': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Extract parameters from matrices
            fx_p1, fy_p1 = P1[0, 0], P1[1, 1]
            cx_p1, cy_p1 = P1[0, 2], P1[1, 2]
            
            fx_p2, fy_p2 = P2[0, 0], P2[1, 1]
            cx_p2, cy_p2 = P2[0, 2], P2[1, 2]
            
            baseline_from_p = abs(P2[0, 3] / P2[0, 0])
            
            # Extract parameters from Q matrix
            cx_q = -Q[0, 3]
            cy_q = -Q[1, 3]
            fx_q = Q[2, 3]
            baseline_from_q = abs(1.0 / Q[3, 2])
            
            # Check focal length consistency
            fx_diff = abs(fx_p1 - fx_q)
            if fx_diff > tolerance:
                results['warnings'].append(f"Focal length inconsistency: P1={fx_p1:.2f}, Q={fx_q:.2f}")
            
            # Check principal point consistency
            cx_diff = abs(cx_p1 - cx_q)
            cy_diff = abs(cy_p1 - cy_q)
            
            if cx_diff > tolerance:
                results['warnings'].append(f"Principal point X inconsistency: P1={cx_p1:.2f}, Q={cx_q:.2f}")
            
            if cy_diff > tolerance:
                results['warnings'].append(f"Principal point Y inconsistency: P1={cy_p1:.2f}, Q={cy_q:.2f}")
            
            # Check baseline consistency
            baseline_diff = abs(baseline_from_p - baseline_from_q)
            baseline_ratio = baseline_diff / baseline_from_p if baseline_from_p > 0 else float('inf')
            
            if baseline_ratio > 0.01:  # 1% tolerance for baseline
                results['warnings'].append(f"Baseline inconsistency: P={baseline_from_p:.4f}, Q={baseline_from_q:.4f}")
            
            # Check P1 and P2 consistency
            if not StereoMath.check_focal_length_consistency(P1, P2, tolerance):
                results['warnings'].append("P1 and P2 focal lengths are inconsistent")
            
            # Store metrics
            results['metrics'] = {
                'focal_length_diff': fx_diff,
                'principal_point_diff': (cx_diff, cy_diff),
                'baseline_diff': baseline_diff,
                'baseline_ratio': baseline_ratio
            }
            
            if results['warnings']:
                results['consistent'] = False
                
        except Exception as e:
            results['consistent'] = False
            results['errors'].append(f"Validation failed: {str(e)}")
        
        return results
    
    def create_matrix_metadata(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        Q: np.ndarray,
        original_size: Tuple[int, int],
        rectified_size: Tuple[int, int],
        resize_scale: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for rectification matrices.
        
        Args:
            P1: Left projection matrix
            P2: Right projection matrix
            Q: Q matrix
            original_size: Original image size
            rectified_size: Rectified image size
            resize_scale: Optional resize scaling factors
            
        Returns:
            Dict[str, Any]: Matrix metadata
        """
        baseline = StereoMath.calculate_baseline_from_projection_matrices(P1, P2)
        
        metadata = {
            'matrix_version': 'rectification_v1.0',
            'original_size': list(original_size),
            'rectified_size': list(rectified_size),
            'resize_applied': (resize_scale is not None and 
                              (resize_scale[0] != 1.0 or resize_scale[1] != 1.0)),
            'resize_scale': list(resize_scale) if resize_scale else None,
            'projection_matrices': {
                'P1': P1.tolist(),
                'P2': P2.tolist(),
                'focal_lengths': {
                    'fx': float(P1[0, 0]),
                    'fy': float(P1[1, 1])
                },
                'principal_points': {
                    'cx': float(P1[0, 2]),
                    'cy': float(P1[1, 2])
                },
                'baseline': float(baseline)
            },
            'q_matrix': {
                'Q': Q.tolist(),
                'parameters': {
                    'cx': float(-Q[0, 3]),
                    'cy': float(-Q[1, 3]),
                    'fx': float(Q[2, 3]),
                    'baseline_inv': float(Q[3, 2])
                }
            }
        }
        
        # Add consistency validation
        consistency = self.validate_matrix_consistency(P1, P2, Q)
        metadata['consistency_check'] = consistency
        
        return metadata
    
    def compare_q_matrices(
        self,
        Q1: np.ndarray,
        Q2: np.ndarray,
        labels: Tuple[str, str] = ("Q1", "Q2")
    ) -> Dict[str, Any]:
        """
        Compare two Q matrices and analyze differences.
        
        Args:
            Q1: First Q matrix
            Q2: Second Q matrix
            labels: Labels for the matrices
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        label1, label2 = labels
        
        # Extract parameters from both matrices
        params1 = {
            'cx': -Q1[0, 3],
            'cy': -Q1[1, 3],
            'fx': Q1[2, 3],
            'baseline_inv': Q1[3, 2]
        }
        
        params2 = {
            'cx': -Q2[0, 3],
            'cy': -Q2[1, 3],
            'fx': Q2[2, 3],
            'baseline_inv': Q2[3, 2]
        }
        
        # Calculate differences
        differences = {}
        for key in params1:
            diff = abs(params1[key] - params2[key])
            rel_diff = diff / abs(params1[key]) if abs(params1[key]) > 1e-10 else float('inf')
            differences[key] = {
                'absolute': diff,
                'relative': rel_diff,
                f'{label1}_value': params1[key],
                f'{label2}_value': params2[key]
            }
        
        # Overall matrix difference (Frobenius norm)
        matrix_diff = np.linalg.norm(Q1 - Q2, 'fro')
        
        comparison = {
            'matrix_labels': labels,
            'parameter_differences': differences,
            'matrix_frobenius_norm_diff': matrix_diff,
            'parameters': {
                label1: params1,
                label2: params2
            }
        }
        
        self.logger.info(f"Q matrix comparison ({label1} vs {label2}):")
        for param, diff_info in differences.items():
            self.logger.info(f"  {param}: {diff_info['absolute']:.6f} "
                           f"(relative: {diff_info['relative']:.2%})")
        
        return comparison
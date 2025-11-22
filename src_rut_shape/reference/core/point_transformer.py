"""Point transformation utilities for stereo rectification"""

import numpy as np
import cv2
import warnings
from typing import Dict, List, Any


class PointTransformer:
    """Handles point transformation and rectification operations"""
    
    @staticmethod
    def rectify_points(points: np.ndarray, K: np.ndarray, D: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Complete point rectification including undistortion, rotation, and projection"""
        if points.size == 0:
            return points
            
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        K_cv = K.astype(np.float32)
        D_cv = D.flatten().astype(np.float32)
        
        # Handle None matrices
        R_cv = R.astype(np.float32) if R is not None else None
        P_cv = P.astype(np.float32) if P is not None else None
        
        # Complete rectification: undistortion + rotation + new projection
        rectified_points = cv2.undistortPoints(points_cv, K_cv, D_cv, R=R_cv, P=P_cv)
        return rectified_points.reshape(-1, 2).astype(np.float64)
    
    @staticmethod
    def undistort_points(points: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Undistort points before rectification"""
        warnings.warn("This method is deprecated. Use rectify_points for complete rectification.", DeprecationWarning)
        if points.size == 0:
            return points
            
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        K_cv = K.astype(np.float32)
        D_cv = D.flatten().astype(np.float32)
        
        points_undistorted = cv2.undistortPoints(points_cv, K_cv, D_cv, None, K_cv)
        return points_undistorted.reshape(-1, 2).astype(np.float64)
    
    @staticmethod
    def transform_points(points: np.ndarray, K: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Transform points using homography"""
        warnings.warn("This method is deprecated. Use rectify_points for complete rectification.", DeprecationWarning)
        if points.size == 0:
            return points
            
        # Extract new intrinsic matrix from projection matrix
        K_new = P[:, :3]
        K_inv = np.linalg.inv(K)
        
        # Compute homography matrix
        H = K_new @ R @ K_inv
        
        # Convert to homogeneous coordinates
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Apply transformation
        projected_h = (H @ points_h.T).T
        
        # Convert back to Cartesian coordinates
        w = projected_h[:, 2]
        valid_mask = np.abs(w) > 1e-10
        
        rectified_points = np.zeros_like(points, dtype=np.float64)
        rectified_points[valid_mask] = projected_h[valid_mask, :2] / w[valid_mask, np.newaxis]
        
        # Handle degenerate cases
        if not np.all(valid_mask):
            warnings.warn(f"Degenerate projection detected for {np.sum(~valid_mask)} points.")
            rectified_points[~valid_mask] = points[~valid_mask]

        return rectified_points
    
    @staticmethod
    def rectify_point_sets(left_points: Dict[str, List[List[float]]], 
                          right_points: Dict[str, List[List[float]]],
                          K1: np.ndarray, K2: np.ndarray,
                          D1: np.ndarray, D2: np.ndarray,
                          R1: np.ndarray, R2: np.ndarray,
                          P1: np.ndarray, P2: np.ndarray) -> tuple:
        """Rectify both left and right point sets"""
        cameras = [
            (left_points, K1, D1, R1, P1),
            (right_points, K2, D2, R2, P2)
        ]
        
        rectified_results = []
        
        for points_dict, K, D, R, P in cameras:
            rectified_dict = {}
            for label, points in points_dict.items():
                point_array = np.array(points, dtype=np.float64)
                rectified_points = PointTransformer.rectify_points(point_array, K, D, R, P)
                rectified_dict[label] = rectified_points.tolist()
            rectified_results.append(rectified_dict)
        
        return rectified_results[0], rectified_results[1]
    
    @staticmethod
    def verify_rectification(rectified_left_points: Dict[str, List[List[float]]],
                           rectified_right_points: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        """Verify rectification quality by checking y-coordinate alignment"""
        if rectified_left_points is None or rectified_right_points is None:
            raise ValueError("Points have not been rectified.")
        
        common_labels = set(rectified_left_points.keys()).intersection(
            set(rectified_right_points.keys())
        )
        
        if not common_labels:
            raise ValueError("No corresponding points found between left and right images.")
        
        y_diffs = []
        disparities = []
        
        for label in common_labels:
            left_points = np.array(rectified_left_points[label])
            right_points = np.array(rectified_right_points[label])
            
            min_points = min(len(left_points), len(right_points))
            
            for i in range(min_points):
                y_diff = left_points[i][1] - right_points[i][1]
                disparity = left_points[i][0] - right_points[i][0]
                
                y_diffs.append(y_diff)
                disparities.append(disparity)
        
        y_diffs = np.array(y_diffs)
        disparities = np.array(disparities)
        
        return {
            "mean_y_difference": float(np.mean(np.abs(y_diffs))),
            "max_y_difference": float(np.max(np.abs(y_diffs))),
            "std_y_difference": float(np.std(y_diffs)),
            "rms_y_difference": float(np.sqrt(np.mean(y_diffs**2))),
            "mean_disparity": float(np.mean(disparities)),
            "min_disparity": float(np.min(disparities)),
            "max_disparity": float(np.max(disparities)),
            "num_points_verified": len(y_diffs)
        }
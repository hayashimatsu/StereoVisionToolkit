"""Matrix calculation utilities for stereo rectification"""

import numpy as np
import cv2
import warnings
from typing import Tuple, Dict, Any


class MatrixCalculator:
    """Handles all matrix calculations for stereo rectification"""
    
    @staticmethod
    def check_numerical_stability(K1: np.ndarray, K2: np.ndarray, 
                                D1: np.ndarray, D2: np.ndarray) -> None:
        """Check numerical stability of camera matrices"""
        cond_K1 = np.linalg.cond(K1)
        cond_K2 = np.linalg.cond(K2)
        
        if cond_K1 > 1e10:
            warnings.warn(f"Left camera matrix is ill-conditioned (condition number: {cond_K1:.2e})")
        if cond_K2 > 1e10:
            warnings.warn(f"Right camera matrix is ill-conditioned (condition number: {cond_K2:.2e})")
            
        if np.any(np.abs(D1) > 1.0):
            warnings.warn("Left camera distortion parameters seem unusually large")
        if np.any(np.abs(D2) > 1.0):
            warnings.warn("Right camera distortion parameters seem unusually large")
    
    @staticmethod
    def compute_rectification_matrices(R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Calculate stereo rectification matrices R1 and R2"""
        # Convert rotation matrix to vector if needed
        if R.shape == (3, 3):
            r_vec, _ = cv2.Rodrigues(R.astype(np.float32))
            r_vec = r_vec.astype(np.float64)
        else:
            r_vec = R
        
        # Calculate average rotation
        r_vec = r_vec * (-0.5)
        r_r, _ = cv2.Rodrigues(r_vec.astype(np.float32))
        r_r = r_r.astype(np.float64)
        
        # Calculate rotated translation vector
        t = r_r @ T.reshape(3, 1)
        
        # Determine principal axis
        idx = 0 if abs(t[0, 0]) > abs(t[1, 0]) else 1
        c = t[idx, 0]
        nt = np.linalg.norm(t)
        
        # Determine reference vector
        uu = np.zeros((3, 1), dtype=np.float64)
        uu[idx, 0] = 1.0 if c > 0 else -1.0
        
        # Calculate rotation axis
        ww = np.cross(t.reshape(-1), uu.reshape(-1)).reshape(3, 1)
        nw = np.linalg.norm(ww)
        
        # Calculate and apply rotation angle
        if nw > 0.0:
            ww = ww * (np.arccos(abs(c) / nt) / nw)
        
        # Convert rotation vector to rotation matrix
        wR, _ = cv2.Rodrigues(ww.astype(np.float32))
        wR = wR.astype(np.float64)
        
        # Calculate final R1 and R2
        R1 = wR @ r_r.T
        R2 = wR @ r_r
        
        # Debug information
        debug_info = {
            'r_vec': r_vec, 'r_r': r_r, 't': t, 'idx': idx,
            'c': c, 'nt': nt, 'uu': uu, 'ww': ww, 'nw': nw, 'wR': wR
        }
        
        return R1, R2, debug_info
    
    @staticmethod
    def compute_projection_matrices(K1: np.ndarray, K2: np.ndarray, D1: np.ndarray, D2: np.ndarray,
                                  R1: np.ndarray, R2: np.ndarray, img_size: np.ndarray,
                                  t: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute projection matrices P1 and P2"""
        width, height = img_size.flatten()
        
        # Calculate new focal length
        fc_new = (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]) / 4.0
        
        # Define corners
        corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float64)
        
        # Undistort corners
        corners_undist_l = MatrixCalculator._undistort_points(corners, K1, D1)
        corners_undist_r = MatrixCalculator._undistort_points(corners, K2, D2)
        
        # Transform corners
        corners_norm_l = cv2.undistortPoints(
            corners_undist_l.reshape(-1, 1, 2).astype(np.float32),
            K1.astype(np.float32), np.zeros(5, dtype=np.float32),
            None, R1.astype(np.float32)
        )
        
        corners_norm_r = cv2.undistortPoints(
            corners_undist_r.reshape(-1, 1, 2).astype(np.float32),
            K2.astype(np.float32), np.zeros(5, dtype=np.float32),
            None, R2.astype(np.float32)
        )
        
        # Calculate new principal points
        cc_new_l = np.array([
            width/2 - np.mean(corners_norm_l[:, 0, 0]) * fc_new,
            height/2 - np.mean(corners_norm_l[:, 0, 1]) * fc_new
        ], dtype=np.float64)
        
        cc_new_r = np.array([
            width/2 - np.mean(corners_norm_r[:, 0, 0]) * fc_new,
            height/2 - np.mean(corners_norm_r[:, 0, 1]) * fc_new
        ], dtype=np.float64)
        
        # Align principal points
        if idx == 0:
            cc_new_l[1] = cc_new_r[1] = (cc_new_l[1] + cc_new_r[1]) * 0.5
        else:
            cc_new_l[0] = cc_new_r[0] = (cc_new_l[0] + cc_new_r[0]) * 0.5
        
        # Build projection matrices
        P1 = np.zeros((3, 4), dtype=np.float64)
        P1[0, 0] = P1[1, 1] = fc_new
        P1[0, 2] = cc_new_l[0]
        P1[1, 2] = cc_new_l[1]
        P1[2, 2] = 1.0
        
        P2 = P1.copy()
        P2[0, 2] = cc_new_r[0]
        P2[1, 2] = cc_new_r[1]
        P2[idx, 3] = t[idx, 0] * fc_new
        
        return P1, P2
    
    @staticmethod
    def compute_inverse_camera_matrix(K: np.ndarray) -> np.ndarray:
        """Compute inverse of camera matrix"""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        
        return np.array([
            [1.0/fx, 0, -cx/fx],
            [0, 1.0/fy, -cy/fy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @staticmethod
    def _undistort_points(points: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Undistort points using OpenCV"""
        if points.size == 0:
            return points
            
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        K_cv = K.astype(np.float32)
        D_cv = D.flatten().astype(np.float32)
        
        points_undistorted = cv2.undistortPoints(points_cv, K_cv, D_cv, None, K_cv)
        return points_undistorted.reshape(-1, 2).astype(np.float64)
"""Main stereo rectifier class"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from ..utils.config_loader import ConfigLoader
from ..utils.data_loader import DataLoader
from .matrix_calculator import MatrixCalculator
from .point_transformer import PointTransformer
from ..visualization.plotter import RectificationPlotter
from .rectification_result import RectificationResult
from .rectification_data import RectificationResult as UnifiedResult
import cv2


class StereoRectifier:
    """Main class for stereo rectification with clean architecture"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file"""
        self.config = ConfigLoader.load_config(config_path)
        ConfigLoader.resolve_config_paths(self.config)
        
        self.output_folder = os.path.join("output", self.config["output"]["folder_name"])
        os.makedirs(self.output_folder, exist_ok=True)
        
        self._load_camera_parameters()
        self._load_test_points()
        self._initialize_results()
        
        # Check numerical stability
        MatrixCalculator.check_numerical_stability(self.K1, self.K2, self.D1, self.D2)
    
    def _load_camera_parameters(self) -> None:
        """Load all camera parameters from CSV files"""
        params = self.config["parameters"]
        
        self.K1 = DataLoader.load_csv_matrix(params["K1"])
        self.K2 = DataLoader.load_csv_matrix(params["K2"])
        self.D1 = DataLoader.load_csv_matrix(params["D1"])
        self.D2 = DataLoader.load_csv_matrix(params["D2"])
        self.R = DataLoader.load_csv_matrix(params["R"])
        self.T = DataLoader.load_csv_matrix(params["T"])
        self.Q = DataLoader.load_csv_matrix(params["Q"])
        self.sensor_size = DataLoader.load_csv_matrix(params["SensorSize"])
        self.img_size = DataLoader.load_csv_matrix(params["imgSize"])
    
    def _load_test_points(self) -> None:
        """Load test points from CSV files"""
        self.left_points = DataLoader.load_test_points(self.config['test_points']['left'])
        self.right_points = DataLoader.load_test_points(self.config['test_points']['right'])
    
    def _initialize_results(self) -> None:
        """Initialize result variables"""
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.K1_inv = None
        self.K2_inv = None
        self.rectified_left_points = None
        self.rectified_right_points = None
        self.debug_info = None
    
    def compute_rectification_matrices(self) -> None:
        """Compute all rectification matrices"""
        # Compute rectification matrices
        self.R1, self.R2, self.debug_info = MatrixCalculator.compute_rectification_matrices(
            self.R, self.T
        )
        
        # Compute projection matrices
        self.P1, self.P2 = MatrixCalculator.compute_projection_matrices(
            self.K1, self.K2, self.D1, self.D2, self.R1, self.R2,
            self.img_size, self.debug_info['t'], self.debug_info['idx']
        )
        
        # Compute inverse camera matrices
        self.K1_inv = MatrixCalculator.compute_inverse_camera_matrix(self.K1)
        self.K2_inv = MatrixCalculator.compute_inverse_camera_matrix(self.K2)
    
    # def rectify_points(self) -> None:
    #     """Rectify test points using computed matrices"""
    #     self.rectified_left_points, self.rectified_right_points = PointTransformer.rectify_point_sets(
    #         self.left_points, self.right_points,
    #         self.K1, self.K2, self.D1, self.D2,
    #         self.R1, self.R2, self.P1, self.P2
    #     )
    
    def process(self, left_image: np.ndarray, right_image: np.ndarray, 
                points: Optional[Dict] = None, alpha: float = -1.0, flags: int = 0,
                rectify_options: Optional[Dict] = None) -> UnifiedResult:
        """Complete end-to-end stereo rectification processing"""
        # Store original images
        left_original = left_image.copy()
        right_original = right_image.copy()
        
        # Get image size
        image_size = (left_image.shape[1], left_image.shape[0])
        
        # Compute rectification matrices using cv2.stereoRectify
        self.R1, self.R2, self.P1, self.P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2,
            image_size, self.R, self.T,
            alpha=alpha, flags=flags,
            newImageSize=image_size
        )
        
        # Handle rectify options
        if rectify_options is None:
            rectify_options = {}
        
        # Determine output image size and adjusted projection matrices
        new_image_size = rectify_options.get("new_image_size")
        
        if new_image_size == "auto":
            output_size, P1_adj, P2_adj = self._calculate_optimal_rectification_maps(image_size)
        elif new_image_size == "default":
            output_size = image_size
            P1_adj, P2_adj = self.P1, self.P2
        else:
            output_size = new_image_size or image_size
            P1_adj, P2_adj = self.P1, self.P2
        
        # Rectify images using cv2.remap with options
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, P1_adj, output_size, cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, P2_adj, output_size, cv2.CV_32FC1)
        
        interpolation = rectify_options.get("interpolation", cv2.INTER_LINEAR)
        border_mode = rectify_options.get("border_mode", cv2.BORDER_CONSTANT)
        border_value = rectify_options.get("border_value", 0)
        
        rectified_left = cv2.remap(left_image, map1_left, map2_left, interpolation, border_mode, border_value)
        rectified_right = cv2.remap(right_image, map1_right, map2_right, interpolation, border_mode, border_value)
        
        # Process points if provided
        if points is None:
            points = {'left': self.left_points, 'right': self.right_points}
        
        # Rectify points using the computed matrices
        self.rectified_left_points, self.rectified_right_points = PointTransformer.rectify_point_sets(
            points['left'], points['right'],
            self.K1, self.K2, self.D1, self.D2,
            self.R1, self.R2, P1_adj, P2_adj
        )
        
        # Verify rectification
        verification_results = PointTransformer.verify_rectification(
            self.rectified_left_points, self.rectified_right_points
        )
        
        # Prepare point data
        left_points_data, right_points_data = self._prepare_point_data()
        
        return UnifiedResult(
            left_original_image=left_original,
            right_original_image=right_original,
            rectified_left_image=rectified_left,
            rectified_right_image=rectified_right,
            left_points_original=left_points_data,
            right_points_original=right_points_data,
            left_points_rectified=left_points_data,
            right_points_rectified=right_points_data,
            R1=self.R1,
            R2=self.R2,
            P1=self.P1,
            P2=self.P2,
            Q=Q,
            roi1=roi1,
            roi2=roi2,
            mean_y_difference=verification_results['mean_y_difference'],
            max_y_difference=verification_results['max_y_difference'],
            std_y_difference=verification_results['std_y_difference'],
            rms_y_difference=verification_results['rms_y_difference'],
            mean_disparity=verification_results['mean_disparity'],
            min_disparity=verification_results['min_disparity'],
            max_disparity=verification_results['max_disparity'],
            num_points_verified=verification_results['num_points_verified'],
            alpha=alpha,
            flags=flags,
            phase="Unified Processing - Core + Visual Integration",
            improvements=[
                "Integrated image rectification with cv2.stereoRectify",
                "Added cv2.remap for image transformation",
                "Unified data flow with RectificationResult",
                "Eliminated intermediate file dependencies"
            ]
        )
    
    def verify_rectification(self) -> RectificationResult:
        """Verify rectification quality and return legacy result object"""
        verification_results = PointTransformer.verify_rectification(
            self.rectified_left_points, self.rectified_right_points
        )
        
        # Prepare point data
        left_points_data, right_points_data = self._prepare_point_data()
        
        return RectificationResult(
            left_points_original=[],
            right_points_original=[],
            left_points_rectified=left_points_data,
            right_points_rectified=right_points_data,
            mean_y_difference=verification_results['mean_y_difference'],
            max_y_difference=verification_results['max_y_difference'],
            std_y_difference=verification_results['std_y_difference'],
            rms_y_difference=verification_results['rms_y_difference'],
            mean_disparity=verification_results['mean_disparity'],
            min_disparity=verification_results['min_disparity'],
            max_disparity=verification_results['max_disparity'],
            num_points_verified=verification_results['num_points_verified'],
            R1=self.R1,
            R2=self.R2,
            P1=self.P1,
            P2=self.P2,
            phase="Phase 1 - Improved with undistortion and double precision",
            improvements=[
                "Added undistortion processing before rectification",
                "Fixed coordinate transformation sequence",
                "Implemented double precision arithmetic",
                "Added numerical stability checks",
                "Enhanced projection matrix computation"
            ],
            debug_info=self.debug_info
        )
    
    def plot_results(self, output_path: Optional[str] = None) -> None:
        """Plot rectification results"""
        RectificationPlotter.plot_rectification_results(
            self.left_points, self.right_points,
            self.rectified_left_points, self.rectified_right_points,
            output_path
        )
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        """Save all results to files"""
        verification_results = self.verify_rectification()
        
        # Prepare results data
        results = self._prepare_results_data(verification_results)
        
        # Save JSON results
        output_path_json = os.path.join(output_path, self.config['case'] + ".json")
        with open(output_path_json, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save visualization
        plot_path_png = os.path.join(output_path, self.config['case'] + ".png")
        self.plot_results(plot_path_png)
        
        # Save comparison report
        self._save_comparison_report(verification_results, output_path)
    
    def _prepare_point_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare point data for RectificationResult"""
        common_labels = set(self.left_points.keys()).intersection(set(self.right_points.keys()))
        
        left_points_data = []
        right_points_data = []
        
        for label in common_labels:
            left_original = np.array(self.left_points[label])
            right_original = np.array(self.right_points[label])
            left_rectified = np.array(self.rectified_left_points.get(label, []))
            right_rectified = np.array(self.rectified_right_points.get(label, []))
            
            if left_rectified.size == 0 or right_rectified.size == 0:
                continue
                
            min_points = min(left_original.shape[0], right_original.shape[0],
                           left_rectified.shape[0], right_rectified.shape[0])
            
            for i in range(min_points):
                left_points_data.append({
                    "label": label, "id": i,
                    "original_x": float(left_original[i, 0]),
                    "original_y": float(left_original[i, 1]),
                    "rectified_x": float(left_rectified[i, 0]),
                    "rectified_y": float(left_rectified[i, 1])
                })
                
                right_points_data.append({
                    "label": label, "id": i,
                    "original_x": float(right_original[i, 0]),
                    "original_y": float(right_original[i, 1]),
                    "rectified_x": float(right_rectified[i, 0]),
                    "rectified_y": float(right_rectified[i, 1])
                })
        
        return left_points_data, right_points_data
    
    def _calculate_optimal_rectification_maps(self, image_size: tuple) -> tuple:
        """Calculate optimal output size and adjusted projection matrices for extreme distortions"""
        width, height = image_size
        
        # Step A: Calculate bounding box of rectified image
        corners = np.array([
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ], dtype=np.float32)
        
        # Transform corners through complete rectification pipeline
        x_coords, y_coords = [], []
        
        for corner in corners:
            # Convert to homogeneous coordinates and apply inverse camera matrix
            corner_homo = np.array([corner[0], corner[1], 1.0])
            normalized = np.linalg.inv(self.K1) @ corner_homo
            
            # Apply rectification rotation
            rotated = self.R1 @ normalized
            
            # Apply new projection matrix
            projected = self.P1 @ np.array([rotated[0], rotated[1], rotated[2], 1.0])
            
            # Convert back to 2D coordinates
            if projected[2] != 0:
                x = projected[0] / projected[2]
                y = projected[1] / projected[2]
                x_coords.append(x)
                y_coords.append(y)
        
        # Calculate bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        new_width = int(np.ceil(x_max - x_min))
        new_height = int(np.ceil(y_max - y_min))
        
        # Step B: Adjust projection matrices to eliminate negative coordinates
        P1_adj = self.P1.copy()
        P2_adj = self.P2.copy()
        
        # Apply translation to move content to positive coordinate space
        P1_adj[0, 2] += -x_min  # Adjust principal point x
        P1_adj[1, 2] += -y_min  # Adjust principal point y
        P2_adj[0, 2] += -x_min
        P2_adj[1, 2] += -y_min
        
        return (new_width, new_height), P1_adj, P2_adj
    
    def _prepare_results_data(self, verification_results: RectificationResult) -> Dict[str, Any]:
        """Prepare results data for JSON export"""
        left_points_data, right_points_data = self._prepare_point_data()
        
        return {
            "phase": "Unified Processing - Core + Visual Integration",
            "improvements": [
                "Integrated image rectification with cv2.stereoRectify",
                "Added cv2.remap for image transformation", 
                "Unified data flow with RectificationResult",
                "Eliminated intermediate file dependencies",
                "Fixed point rectification algorithm"
            ],
            "left_points": left_points_data,
            "right_points": right_points_data,
            "verification": {
                "mean_y_difference": verification_results.mean_y_difference,
                "max_y_difference": verification_results.max_y_difference,
                "std_y_difference": verification_results.std_y_difference,
                "rms_y_difference": verification_results.rms_y_difference,
                "mean_disparity": verification_results.mean_disparity,
                "min_disparity": verification_results.min_disparity,
                "max_disparity": verification_results.max_disparity,
                "num_points_verified": verification_results.num_points_verified
            },
            "matrices": {
                "R1": self.R1.tolist() if self.R1 is not None else None,
                "R2": self.R2.tolist() if self.R2 is not None else None,
                "P1": self.P1.tolist() if self.P1 is not None else None,
                "P2": self.P2.tolist() if self.P2 is not None else None
            }
        }
    
    def _save_comparison_report(self, verification_results: RectificationResult, output_path: Optional[str] = None) -> None:
        """Save improvement comparison report"""
        comparison_path = os.path.join(output_path, "processing_report.txt")
        
        with open(comparison_path, 'w') as f:
            f.write("=== UNIFIED SYSTEM IMPROVEMENTS REPORT ===\n\n")
            f.write("Key Changes Made:\n")
            f.write("1. Integrated image rectification with cv2.stereoRectify\n")
            f.write("2. Added cv2.remap for image transformation\n")
            f.write("3. Unified data flow with RectificationResult\n")
            f.write("4. Eliminated intermediate file dependencies\n")
            f.write("5. Fixed point rectification algorithm\n")
            f.write("6. Removed double undistortion processing\n\n")
            
            f.write("System Benefits:\n")
            f.write("- Complete end-to-end processing\n")
            f.write("- Algorithmically correct point rectification\n")
            f.write("- In-memory data flow\n")
            f.write("- Configuration-driven workflow\n\n")
            
            f.write("Current Performance:\n")
            f.write(f"Mean Y difference: {verification_results.mean_y_difference:.3f} pixels\n")
            f.write(f"Max Y difference: {verification_results.max_y_difference:.3f} pixels\n")
            f.write(f"RMS Y difference: {verification_results.rms_y_difference:.3f} pixels\n")
            f.write(f"Points verified: {verification_results.num_points_verified}\n")
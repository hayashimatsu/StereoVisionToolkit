"""
Coordinate processing and transformation module for height calculation.

This module handles:
- Converting image coordinates to world coordinates
- Processing different coordinate calculation methods (Case 1 & Case 2)
- Coordinate transformation and scaling operations
- Integration with RutShapeProcessor for advanced processing

Input formats:
- World coordinates: numpy arrays with shape (height, width, 3) for XYZ data
- Image coordinates: Lists of [x, y] pixel coordinates
- Disparity data: numpy arrays with disparity values
- Q matrices: 4x4 transformation matrices

Output formats:
- World coordinates: numpy arrays with shape (n, 3) for XYZ points
- 2D coordinates: numpy arrays with shape (n, 2) for XY points
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from .processors import RutShapeProcessor


class CoordinateProcessor:
    """
    Handles coordinate processing and transformation for height calculation.
    
    Supports two main processing methods:
    - Method 1 (cv2ReprojectImageTo3D): Uses OpenCV's reprojection with world coordinates
    - Method 2 (epipolarCorrection): Uses epipolar geometry with optical axis correction
    
    Attributes:
        config: Configuration object with processing parameters
        temp_folder (Path): Path to temporary files
        rut_processor (RutShapeProcessor): Advanced rut shape processing instance
        logger: Logger instance for this class
    """
    
    def __init__(self, config, temp_folder: Path, rut_processor: RutShapeProcessor):
        """
        Initialize CoordinateProcessor with configuration and dependencies.
        
        Args:
            config: Configuration object with processing parameters
            temp_folder: Path to temporary files folder
            rut_processor: RutShapeProcessor instance for advanced processing
        """
        self.config = config
        self.temp_folder = temp_folder
        self.rut_processor = rut_processor
        self.logger = self._setup_logger()
        
        # Folder name constants (moved from constants.py for consistency)
        self.rectify_folder_name = "target_pictures_set_rectified"
        self.disparity_folder_name = "target_pictures_set_disparity"
        
        # Processing method flags
        self.m1_cv2ReprojectImageTo3D = config.m1_cv2ReprojectImageTo3D == "True"
        self.m2_epipolarCorrection = config.m2_epipolarCorrection == "True"
        
        # Validate method configuration
        self._validate_processing_methods()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this class."""
        logger = logging.getLogger(f"{__name__}.CoordinateProcessor")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _validate_processing_methods(self):
        """
        Validate processing method configuration.
        
        Ensures that exactly one processing method is enabled.
        If both have the same value, defaults to Method 1.
        """
        if self.m1_cv2ReprojectImageTo3D == self.m2_epipolarCorrection:
            self.logger.warning("Both m1_cv2ReprojectImageTo3D and m2_epipolarCorrection have "
                              "the same value. Setting m1_cv2ReprojectImageTo3D=True as default.")
            self.m1_cv2ReprojectImageTo3D = True
            self.m2_epipolarCorrection = False

    def process_coordinates(self, world_coord_file: Path, set_name: str, pair_name: str,
                          rectified_left_pic: np.ndarray, disparity_pic: np.ndarray,
                          rut_rectified: List[List[float]]) -> np.ndarray:
        """
        Process coordinates using the configured method.
        
        Args:
            world_coord_file: Path to world coordinate .npy file
            set_name: Name of the image set
            pair_name: Name of the image pair
            rectified_left_pic: Rectified left image
            disparity_pic: Disparity image
            rut_rectified: List of rectified rut coordinates
            
        Returns:
            Processed coordinate data as numpy array with shape (n, 2)
            
        Raises:
            ValueError: If processing method configuration is invalid
        """
        if self.m1_cv2ReprojectImageTo3D:
            return self._process_method1(world_coord_file, rectified_left_pic, 
                                       disparity_pic, rut_rectified)
        elif self.m2_epipolarCorrection:
            return self._process_method2(set_name, pair_name, rut_rectified)
        else:
            raise ValueError("No valid processing method configured")

    def _process_method1(self, world_coord_file: Path, rectified_left_pic: np.ndarray,
                        disparity_pic: np.ndarray, rut_rectified: List[List[float]]) -> np.ndarray:
        """
        Process coordinates using Method 1 (cv2ReprojectImageTo3D).
        
        Uses OpenCV's reprojection with precomputed world coordinates.
        
        Args:
            world_coord_file: Path to world coordinate data
            rectified_left_pic: Rectified left image
            disparity_pic: Disparity image  
            rut_rectified: Rectified rut coordinates
            
        Returns:
            Processed 2D coordinate data
        """
        self.logger.info("Processing coordinates using Method 1 (cv2ReprojectImageTo3D)")
        
        # Load world coordinates
        world_coordinates = np.load(world_coord_file)
        
        # Convert coordinates to world coordinates
        rut_coords_world = self.rut_processor._convert_coordinates_to_world(
            world_coordinates, rectified_left_pic, disparity_pic, rut_rectified)
        
        # Extract x,y coordinates
        original_data = self.rut_processor._extract_xy_coordinates(rut_coords_world)
        
        return original_data

    def _process_method2(self, set_name: str, pair_name: str, 
                        rut_rectified: List[List[float]]) -> np.ndarray:
        """
        Process coordinates using Method 2 (epipolarCorrection).
        
        Uses epipolar geometry with optical axis correction and Q matrix transformation.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            rut_rectified: Rectified rut coordinates
            
        Returns:
            Processed 2D coordinate data
            
        Raises:
            ValueError: If required data cannot be loaded
        """
        self.logger.info("Processing coordinates using Method 2 (epipolarCorrection)")
        
        # Load required data
        rut_ref_opticalAxisX, rut_ref_opticalAxisY = self._load_corresponding_optical_axis(
            set_name, pair_name)
        rut_disparity = self._load_rut_disparity(set_name, pair_name)
        Q_matrix = self._load_q_matrix_for_pair(set_name, pair_name)
        
        if Q_matrix is None:
            raise ValueError("Q matrix not available for coordinate computation")
        
        # Calculate transformation parameters
        Tx = 1 / Q_matrix[3, 2]  # Baseline translation
        fx = Q_matrix[2, 3]      # Focal length
        
        # Convert inputs to numpy arrays
        rut_rectified_np = np.asarray(rut_rectified, dtype=np.float64)
        disp = np.asarray(rut_disparity, dtype=np.float64).reshape(-1)
        
        # Calculate X and Y coordinates using optical axis method
        X, Y = self._calculate_xy_from_optical_axes(
            rut_rectified_np, [rut_ref_opticalAxisX, rut_ref_opticalAxisY], disp, Tx)
        
        # Calculate Z coordinates
        Z = fx * Tx / disp
        
        # Combine into world coordinates
        rut_coords_world = np.column_stack((X, Y, Z))
        
        # Extract x,y coordinates
        original_data = self.rut_processor._extract_xy_coordinates(rut_coords_world)
        
        return original_data

    def _calculate_xy_from_optical_axes(self, rut_rectified: np.ndarray, 
                                      optical_axes: List[np.ndarray], 
                                      disparity: np.ndarray, Tx: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate X and Y coordinates using optical axis method.
        
        Uses the formulation:
        X_n = Tx * || (x_refX_n, y_refX_n) - (x_rect_rut_n, y_rect_rut_n) || / d_n
        Y_n = Tx * || (x_refY_n, y_refY_n) - (x_rect_rut_n, y_rect_rut_n) || / d_n
        
        Args:
            rut_rectified: Rectified rut coordinates
            optical_axes: List of [X_axis, Y_axis] optical reference points
            disparity: Disparity values
            Tx: Baseline translation parameter
            
        Returns:
            Tuple of (X_coordinates, Y_coordinates)
        """
        X = None
        Y = None
        
        for i, rut_ref_opticalAxis in enumerate(optical_axes):
            ref_axes = np.asarray(rut_ref_opticalAxis, dtype=np.float64)
            if ref_axes.ndim > 1 and ref_axes.shape[1] > 2:
                ref_axes = ref_axes[:, :2]
            
            # Ensure all arrays have the same length
            n = min(len(rut_rectified), len(ref_axes), len(disparity))
            if n == 0:
                raise ValueError("No points available to compute coordinates")
            
            rr = rut_rectified[:n, :2]
            ra = ref_axes[:n, :2]
            d = disparity[:n]
            
            # Calculate Euclidean distance between corresponding points
            diff = rr - ra
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            
            # Calculate coordinates with divide-by-zero protection
            with np.errstate(divide='ignore', invalid='ignore'):
                coords = np.where(d != 0, Tx * dist / d, 0.0)
                
                if i == 0:  # X coordinates
                    X = coords.copy()
                    # Apply sign correction for X coordinates
                    idx_min = int(np.argmin(X))
                    if idx_min > 0:
                        X[:idx_min] *= -1.0
                else:  # Y coordinates
                    Y = coords.copy()
        
        return X, Y

    def _load_corresponding_optical_axis(self, set_name: str, pair_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load corresponding optical axis data from rectifier temp folder.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            
        Returns:
            Tuple of (RefX_data, RefY_data) as numpy arrays
            
        Raises:
            FileNotFoundError: If optical axis files not found
        """
        rectified_temp_pair = self.temp_folder / self.rectify_folder_name / set_name / pair_name
        
        # Load X reference data
        artifact_x = rectified_temp_pair / f'rect_interpolated_RefX_{pair_name}.csv'
        if not artifact_x.exists():
            raise FileNotFoundError(f"Optical axis X artifact not found: {artifact_x}")
        rect_interpolated_RefX = np.genfromtxt(artifact_x, delimiter=',')
        
        # Load Y reference data
        artifact_y = rectified_temp_pair / f'rect_interpolated_RefY_{pair_name}.csv'
        if not artifact_y.exists():
            raise FileNotFoundError(f"Optical axis Y artifact not found: {artifact_y}")
        rect_interpolated_RefY = np.genfromtxt(artifact_y, delimiter=',')
        
        return rect_interpolated_RefX, rect_interpolated_RefY

    def _load_rut_disparity(self, set_name: str, pair_name: str) -> np.ndarray:
        """
        Load rut disparity data from disparity temp folder.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            
        Returns:
            Rut disparity data as numpy array
            
        Raises:
            FileNotFoundError: If disparity file not found
        """
        disparity_temp_pair = self.temp_folder / self.disparity_folder_name / set_name / pair_name
        artifact = disparity_temp_pair / f'rut_disparity_{pair_name}.csv'
        if not artifact.exists():
            raise FileNotFoundError(f"Rut disparity artifact not found: {artifact}")
        
        rut_disparity = np.genfromtxt(artifact, delimiter=',')
        return rut_disparity

    def _load_q_matrix_for_pair(self, set_name: str, pair_name: str) -> Optional[np.ndarray]:
        """
        Load the Q matrix for the given pair from rectified temp folder.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            
        Returns:
            Q matrix as numpy array, or None if not found
        """
        rectified_temp_pair = self.temp_folder / self.rectify_folder_name / set_name / pair_name
        q_rectified_path = rectified_temp_pair / f'Q_rectified_{pair_name}.csv'
        
        if q_rectified_path.exists():
            Q_matrix = np.genfromtxt(q_rectified_path, delimiter=',')
            self.logger.info(f"Loaded recalculated Q matrix for {pair_name} from {q_rectified_path}")
            return Q_matrix
        else:
            self.logger.warning(f"Q matrix not found for {pair_name} at {q_rectified_path}")
            return None
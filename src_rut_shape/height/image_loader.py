"""
Image loading and preprocessing module for height calculation.

This module handles:
- Loading rectified and disparity images
- Loading rectified interpolated points with fallback logic
- Image preprocessing and normalization
- Point interpolation between coordinates

Input formats:
- Rectified images: .npy files containing rectified stereo images
- Disparity images: .npy files containing disparity maps
- Point data: JSON files with interpolated point coordinates

Output formats:
- Loaded images as numpy arrays
- Interpolated point lists as [[x, y], ...] format
"""

import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class ImageLoader:
    """
    Handles loading and preprocessing of images and point data for height calculation.
    
    Attributes:
        pic_folder (Path): Path to rectified images folder
        disparity_folder (Path): Path to disparity images folder  
        temp_folder (Path): Path to temporary files folder
        root_folder (Path): Root project folder path
        config: Configuration object with processing parameters
        logger: Logger instance for this class
    """
    
    def __init__(self, pic_folder: Path, disparity_folder: Path, temp_folder: Path, 
                 root_folder: Path, config):
        """
        Initialize ImageLoader with folder paths and configuration.
        
        Args:
            pic_folder: Path to rectified images
            disparity_folder: Path to disparity images
            temp_folder: Path to temporary files
            root_folder: Root project folder
            config: Configuration object
        """
        self.pic_folder = pic_folder
        self.disparity_folder = disparity_folder
        self.temp_folder = temp_folder
        self.root_folder = root_folder
        self.config = config
        
        # Constants (moved from constants.py for consistency with other modules)
        self.rectify_folder_name = "target_pictures_set_rectified"
        self.default_interpolation_points = 50
        self.default_normalization_alpha = 0
        self.default_normalization_beta = 255
        self.default_colormap = 2  # cv2.COLORMAP_JET equivalent
        
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for this class.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.ImageLoader")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_images(self, set_name: str, pair_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load rectified and disparity images for a given set and pair.
        
        Args:
            set_name: Name of the image set (e.g., 'set_1')
            pair_name: Name of the image pair (e.g., 'pair_001')
            
        Returns:
            Tuple of (rectified_image, disparity_image) as numpy arrays
            
        Raises:
            FileNotFoundError: If required image files are not found
        """
        rectified_pic = self._load_rectified_pic(set_name, pair_name)
        disparity_pic = self._load_disparity_pic(set_name, pair_name)
        return rectified_pic, disparity_pic

    def load_rectified_points(self, set_name: str, pair_name: str) -> List[List[float]]:
        """
        Load rectified points with fallback logic.
        
        First attempts to load precomputed rectified interpolated points.
        If not found, falls back to loading from parameter files and interpolating.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            
        Returns:
            List of interpolated points as [[x, y], [x, y], ...]
            
        Raises:
            FileNotFoundError: If neither rectified points nor parameter files are found
            ValueError: If point data is invalid or incomplete
        """
        try:
            return self._load_rectified_interpolated_points(set_name, pair_name)
        except FileNotFoundError:
            self.logger.info(f"Rectified interpolated points not found for {pair_name}, "
                           f"falling back to parameter points")
            return self._load_and_interpolate_from_param(pair_name)

    def _get_rectified_pic_path(self, set_name: str, pair_name: str) -> Path:
        """Get path to rectified image file."""
        return self.pic_folder / set_name / pair_name
    
    def _load_rectified_pic(self, set_name: str, pair_name: str) -> np.ndarray:
        """
        Load rectified image from .npy file.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            
        Returns:
            Rectified image as numpy array
            
        Raises:
            FileNotFoundError: If rectified image file not found
        """
        path = self._get_rectified_pic_path(set_name, pair_name)
        file_path = path / f'left_rectified_{pair_name}.npy'
        if not file_path.exists():
            raise FileNotFoundError(f"Rectified picture not found: {file_path}")
        return np.load(file_path)
    
    def _get_disparity_pic_path(self, set_name: str, pair_name: str) -> Path:
        """Get path to disparity image file."""
        return self.disparity_folder / set_name / pair_name
   
    def _load_disparity_pic(self, set_name: str, pair_name: str) -> np.ndarray:
        """
        Load and preprocess disparity image from .npy file.
        
        Applies normalization and color mapping to the disparity data.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            
        Returns:
            Processed disparity image as numpy array (color-mapped)
            
        Raises:
            FileNotFoundError: If disparity image file not found
        """
        path = self._get_disparity_pic_path(set_name, pair_name)
        file_path = path / f'disparity_{pair_name}.npy'
        if not file_path.exists():
            raise FileNotFoundError(f"Disparity picture not found: {file_path}")
        
        disparity = np.load(file_path)
        disparity_normalized = cv2.normalize(disparity, None, alpha=self.default_normalization_alpha, 
                                           beta=self.default_normalization_beta, 
                                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity_pic = cv2.applyColorMap(disparity_normalized, self.default_colormap)
        return disparity_pic

    def _load_rectified_interpolated_points(self, set_name: str, pair_name: str) -> List[List[float]]:
        """
        Load precomputed rectified interpolated points from JSON file.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            
        Returns:
            List of interpolated points
            
        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If point data is invalid
        """
        rectified_temp_pair = self.temp_folder / self.rectify_folder_name / set_name / pair_name
        artifact = rectified_temp_pair / f'rectified_interpolated_points_{pair_name}.json'
        if not artifact.exists():
            raise FileNotFoundError(f"Rectified interpolated points artifact not found: {artifact}")
        
        with open(artifact, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        interpolated_points = data.get('interpolated_points', [])
        if not interpolated_points or len(interpolated_points) < 2:
            raise ValueError(f"Invalid interpolated points in artifact: {artifact}")
        
        self.logger.info(f"Loaded {len(interpolated_points)} precomputed rectified "
                        f"interpolated points for {pair_name}")
        return interpolated_points

    def _load_and_interpolate_from_param(self, pair_name: str) -> List[List[float]]:
        """
        Load points from parameter folder and interpolate between them.
        
        Loads rut_1 and rut_2 points from parameter JSON file and creates
        interpolated points between them.
        
        Args:
            pair_name: Name of the image pair
            
        Returns:
            List of interpolated points between rut_1 and rut_2
            
        Raises:
            FileNotFoundError: If parameter JSON file not found
            ValueError: If required rut points are missing
        """
        param_dir = self.root_folder / Path(self.config.parameter_path)
        expected_file = param_dir / f"left_{pair_name}.json"
        if not expected_file.exists():
            raise FileNotFoundError(f"Point JSON not found. Expected at: {expected_file}")
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get('shapes', [])
        label_to_point = {}
        
        for item in shapes:
            label = item.get('label')
            pts = item.get('points')
            if (label in ('rut_1', 'rut_2') and isinstance(pts, list) and 
                len(pts) > 0 and isinstance(pts[0], list) and len(pts[0]) >= 2):
                x, y = pts[0][0], pts[0][1]
                label_to_point[label] = (int(round(x)), int(round(y)))
        
        if 'rut_1' not in label_to_point or 'rut_2' not in label_to_point:
            raise ValueError(f"JSON missing required rut_1/rut_2 points: {expected_file}")
        
        # Interpolate between the two points
        p1, p2 = label_to_point['rut_1'], label_to_point['rut_2']
        interpolated_points = self._interpolate_points(p1, p2)
        
        self.logger.info(f"Interpolated {len(interpolated_points)} points from "
                        f"parameter file for {pair_name}")
        return interpolated_points

    def _interpolate_points(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                          num_points: Optional[int] = None) -> List[List[float]]:
        """
        Interpolate points between two endpoints using linear interpolation.
        
        Args:
            p1: Starting point (x, y)
            p2: Ending point (x, y)
            num_points: Number of points to generate (default from constants)
            
        Returns:
            List of interpolated points as [[x, y], ...]
        """
        if num_points is None:
            num_points = self.default_interpolation_points
        x1, y1 = p1
        x2, y2 = p2
        
        interpolated = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            interpolated.append([x, y])
        
        return interpolated
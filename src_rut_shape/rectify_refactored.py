"""
Refactored stereo rectification module.

This module provides a clean, modular interface for stereo image rectification
with improved code organization, error handling, and maintainability.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json

# Import base classes
from .base import BaseProcessor

# Import new modular components
from .rectification.engine import StereoRectificationEngine
from .rectification.matrix_calculator import MatrixCalculator
from .rectification.file_manager import RectificationFileManager

from utils.point_processor import PointProcessor, RutPointLoader
from utils.image_processing import ResizeStrategy, ImageProcessor
from utils.visualizer import (
    ImageOverlayCreator, PointVisualizer, OpticalCenterVisualizer
)
from utils.file_operations import PathManager
from utils.logger_config import get_logger

# Import legacy components for compatibility
from utils.image import ImageChartGenerator
from .reference.core.point_transformer import PointTransformer


class Rectifier(BaseProcessor):
    """
    Main rectification coordinator class.
    
    This class orchestrates the stereo rectification process using modular components
    for improved maintainability and code organization.
    """
    
    def __init__(self, config):
        """
        Initialize rectifier with configuration.
        
        Args:
            config: Configuration object with rectification parameters
        """
        super().__init__(config, "rectification")
        
        # Initialize modular components
        self.rectification_engine = StereoRectificationEngine(self._get_rectification_config())
        self.matrix_calculator = MatrixCalculator()
        self.point_processor = PointProcessor()
        self.resize_strategy = ResizeStrategy(self._get_resize_config())
        
        # Initialize file manager
        self.file_manager = RectificationFileManager(self.output_folder, self.temp_folder)
        
        # Setup input folder
        self._setup_input_folder()
        
        # Load calibration parameters
        self._load_calibration_parameters()

        # pass to other module
        self.resize_scale = None
        
        self.logger.info("Rectifier initialized with modular architecture")
    

    
    def _get_rectification_config(self) -> Dict[str, Any]:
        """Extract rectification-specific configuration."""
        return {
            'rectified_image_size': getattr(self.config, 'rectified_image_size', 'default'),
            'rectification_alpha': getattr(self.config, 'rectification_alpha', 0)
        }
    
    def _get_resize_config(self) -> Dict[str, Any]:
        """Extract resize-specific configuration."""
        return {
            'resize_target_width': getattr(self.config, 'resize_target_width', None),
            'resize_target_height': getattr(self.config, 'resize_target_height', None),
            'resize_scale': getattr(self.config, 'resize_scale', 1),
            'resize_max_pixels': getattr(self.config, 'resize_max_pixels', 3840 * 2160)
        }
    
    def _load_calibration_parameters(self) -> None:
        """Load stereo calibration parameters from files."""
        self.logger.info("Loading calibration parameters")
        
        parameter_files = {
            'K_left': 'K1.csv',
            'K_right': 'K2.csv',
            'd_left': 'd1.csv',
            'd_right': 'd2.csv',
            'R': 'R.csv',
            'T': 'T.csv'
        }
        
        parameters = {}
        for param_name, filename in parameter_files.items():
            file_path = self.root / self.config.parameter_path / filename
            parameters[param_name] = np.genfromtxt(file_path, delimiter=',')
        
        # Load parameters into rectification engine
        self.rectification_engine.load_calibration_parameters(
            parameters['K_left'], parameters['d_left'],
            parameters['K_right'], parameters['d_right'],
            parameters['R'], parameters['T']
        )
        
        self.logger.info("Calibration parameters loaded successfully")
    
    def _setup_input_folder(self) -> None:
        """Setup input folder path specific to rectification."""
        self.input_folder = self.root / Path(self.config.image_set_folder)
    
    def _get_processor_specific_config(self) -> Dict[str, Any]:
        """Get rectification-specific configuration parameters."""
        return {
            'rectified_image_size': getattr(self.config, 'rectified_image_size', 'default'),
            'rectification_alpha': getattr(self.config, 'rectification_alpha', 0),
            'resize_target_width': getattr(self.config, 'resize_target_width', None),
            'resize_target_height': getattr(self.config, 'resize_target_height', None),
            'resize_scale': getattr(self.config, 'resize_scale', 1),
            'resize_max_pixels': getattr(self.config, 'resize_max_pixels', 3840 * 2160)
        }
    
    def _is_processing_ready(self) -> bool:
        """Check if rectifier is ready for processing."""
        return (self.input_folder is not None and 
                self.rectification_engine is not None and
                self.file_manager is not None)

    def create_rectified_stereo_photos(self) -> None:
        """
        Main entry point for processing all image sets.
        
        This method processes all 'set_*' folders in the input directory.
        """
        self.process_all_sets()
    
    def _execute_processing_pipeline(self, pair_folder: Path) -> Dict[str, Any]:
        """
        Execute the main rectification pipeline for a single pair.
        
        Args:
            pair_folder: Path to the pair folder
            
        Returns:
            Dict[str, Any]: Processing results
        """
        pair_name = pair_folder.name
        
        # Load and validate images
        left_image, right_image = self._load_image_pair(pair_folder, pair_name)
        
        # Setup rectification for this image size
        image_size = left_image.shape[:2][::-1]  # (width, height)
        self._setup_rectification_for_images(image_size)
        
        # Perform rectification
        rectified_left, rectified_right = self._perform_rectification(left_image, right_image)
        
        # Apply resize strategy if needed
        rectified_left, rectified_right, resize_scale = self._apply_resize_strategy(
            rectified_left, rectified_right
        )
        self.resize_scale = resize_scale
        
        # Process rut points
        rectified_points_data = self._process_rut_points(pair_name, resize_scale)
        
        # Calculate final matrices
        matrices = self._calculate_final_matrices(resize_scale)
        
        return {
            'rectified_left': rectified_left,
            'rectified_right': rectified_right,
            'rectified_points_data': rectified_points_data,
            'matrices': matrices,
            'resize_scale': resize_scale
        }
    
    def _save_processing_results(self, processing_results: Dict[str, Any], pair_name: str) -> None:
        """
        Save rectification results using the file manager.
        
        Args:
            processing_results: Results from rectification processing
            pair_name: Name of the image pair
        """
        self._save_complete_results(
            processing_results['rectified_left'],
            processing_results['rectified_right'], 
            processing_results['rectified_points_data'],
            processing_results['matrices'],
            pair_name
        )
    

    
    def _load_image_pair(self, pair_folder: Path, pair_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and validate stereo image pair.
        
        Args:
            pair_folder: Path to pair folder
            pair_name: Name of the pair
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_image, right_image)
            
        Raises:
            ValueError: If images cannot be loaded or are invalid
        """
        left_path = pair_folder / f'left_{pair_name}.jpg'
        right_path = pair_folder / f'right_{pair_name}.jpg'
        
        left_image = cv2.imread(str(left_path))
        right_image = cv2.imread(str(right_path))
        
        if left_image is None:
            raise ValueError(f"Failed to load left image: {left_path}")
        if right_image is None:
            raise ValueError(f"Failed to load right image: {right_path}")
        
        # Validate image compatibility
        ImageProcessor.validate_image_pair(left_image, right_image)
        
        self.logger.debug(f"Loaded image pair: {left_image.shape}")
        return left_image, right_image
    
    def _setup_rectification_for_images(self, image_size: Tuple[int, int]) -> None:
        """
        Setup rectification engine for specific image size.
        
        Args:
            image_size: (width, height) of input images
        """
        config = self._get_rectification_config()
        size_mode = config.get('rectified_image_size', 'default')
        alpha = config.get('rectification_alpha', 0)
        
        self.rectification_engine.setup_rectification(image_size, size_mode, alpha)
        
        # Store rectification info for later use
        self.current_pair_info['rectification_info'] = self.rectification_engine.get_rectification_info()
    
    def _perform_rectification(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform stereo rectification on image pair.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rectified_left, rectified_right)
        """
        return self.rectification_engine.rectify_image_pair(left_image, right_image)
    
    def _apply_resize_strategy(
        self, 
        rectified_left: np.ndarray, 
        rectified_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """
        Apply resize strategy to rectified images.
        
        Args:
            rectified_left: Rectified left image
            rectified_right: Rectified right image
            
        Returns:
            Tuple containing resized images and scale factor
        """
        return self.resize_strategy.apply_resize_strategy(rectified_left, rectified_right)
    
    def _process_rut_points(
        self, 
        pair_name: str, 
        resize_scale: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Process rut points through rectification and interpolation.
        
        Args:
            pair_name: Name of the image pair
            resize_scale: Resize scaling factors
            
        Returns:
            Dict[str, Any]: Processed rut points data
        """
        # Load rut points from JSON
        param_dir = self.root / Path(self.config.parameter_path)
        rut_points_dict = RutPointLoader.load_rut_points_from_json(param_dir, pair_name)
        rut_left, rut_right = RutPointLoader.extract_start_end_coordinates(rut_points_dict)
        
        # Get camera parameters for optical axis calculations
        K_left = self.rectification_engine.K_left
        optical_center = (K_left[0, 2], K_left[1, 2])
        
        # Create optical axis reference points
        optical_ref_points = PointProcessor.create_optical_axis_reference_points(
            (rut_left, rut_right), optical_center
        )
        
        # Compute rectified points
        rectified_interpolated_points = self._compute_rectified_points(
            rut_left, rut_right, resize_scale
        )
        
        # Compute optical axis reference points
        ref_y_points = self._compute_rectified_points(
            optical_ref_points[0][0], optical_ref_points[0][1], resize_scale,
            num=len(rectified_interpolated_points)
        )
        
        ref_x_points = self._compute_rectified_points(
            optical_ref_points[1][0], optical_ref_points[1][1], resize_scale,
            num=len(rectified_interpolated_points)
        )
        
        return {
            'rectified_interpolated_points': rectified_interpolated_points,
            'optical_ref_x_points': ref_x_points,
            'optical_ref_y_points': ref_y_points,
            'original_rut_points': {'rut_left': rut_left, 'rut_right': rut_right}
        }
    
    def _compute_rectified_points(
        self, 
        left_point: np.ndarray, 
        right_point: np.ndarray,
        resize_scale: Tuple[float, float],
        num: int = 10
    ) -> List[List[int]]:
        """
        Compute rectified coordinates for interpolated points.
        
        Args:
            left_point: Left point coordinates
            right_point: Right point coordinates
            resize_scale: Resize scaling factors
            num: Number of interpolated points
            
        Returns:
            List[List[int]]: Rectified interpolated points
        """
        # Interpolate between original coordinates
        interpolated_orig = PointProcessor.interpolate_points_between_coordinates(
            left_point[0], right_point[0], num
        )
        
        # Apply rectification to all interpolated points
        rectified_points = []
        rectification_info = self.current_pair_info['rectification_info']
        
        # Use adjusted projection matrix if available
        P1 = (self.rectification_engine.P1_adjusted 
              if self.rectification_engine.P1_adjusted is not None 
              else self.rectification_engine.P1)
        
        for point in interpolated_orig:
            # Apply rectification transformation
            point_rect = PointTransformer.rectify_points(
                np.array([point], dtype=np.float64),
                self.rectification_engine.K_left,
                self.rectification_engine.d_left,
                self.rectification_engine.R1,
                P1
            )[0]
            rectified_points.append(point_rect.tolist())
        
        # Apply resize scaling if needed
        if resize_scale[0] != 1.0 or resize_scale[1] != 1.0:
            rectified_points = PointProcessor.apply_resize_scaling_to_points(
                rectified_points, resize_scale
            )
        
        # Round to integers
        return PointProcessor.round_points_to_integers(rectified_points, method='ceil')
    
    def _calculate_final_matrices(self, resize_scale: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """
        Calculate final projection and Q matrices accounting for resize.
        
        Args:
            resize_scale: Resize scaling factors
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of calculated matrices
        """
        # Get base matrices from rectification engine
        P1_base = (self.rectification_engine.P1_adjusted 
                  if self.rectification_engine.P1_adjusted is not None 
                  else self.rectification_engine.P1)
        P2_base = (self.rectification_engine.P2_adjusted 
                  if self.rectification_engine.P2_adjusted is not None 
                  else self.rectification_engine.P2)
        
        # Adjust for resize if needed
        if resize_scale[0] != 1.0 or resize_scale[1] != 1.0:
            P1_final, P2_final = self.matrix_calculator.adjust_projection_matrices_for_resize(
                P1_base, P2_base, resize_scale
            )
        else:
            P1_final, P2_final = P1_base.copy(), P2_base.copy()
        
        # Calculate Q matrix from final projection matrices
        Q_final = self.matrix_calculator.calculate_q_matrix_from_projection_matrices(
            P1_final, P2_final
        )
        
        return {
            'P1': P1_final,
            'P2': P2_final,
            'P1_original': self.rectification_engine.P1,
            'P2_original': self.rectification_engine.P2,
            'Q_rectified': Q_final,
            'Q_opencv': self.rectification_engine.Q,
            'R1': self.rectification_engine.R1,
            'R2': self.rectification_engine.R2
        }
    
    def _save_complete_results(
        self,
        rectified_left: np.ndarray,
        rectified_right: np.ndarray,
        points_data: Dict[str, Any],
        matrices: Dict[str, np.ndarray],
        pair_name: str
    ) -> None:
        """
        Save all rectification results including images, matrices, and metadata.
        
        Args:
            rectified_left: Rectified left image
            rectified_right: Rectified right image
            points_data: Processed rut points data
            matrices: Calculated matrices
            pair_name: Name of the image pair
        """
        set_name = self.current_pair_info['set_name']
        
        # Setup output directories
        output_paths = self.file_manager.setup_output_directories(set_name, pair_name)
        
        # Save rectified images
        image_results = self.file_manager.save_rectified_images(
            rectified_left, rectified_right, output_paths, pair_name
        )
        
        # Save matrices
        matrix_results = self.file_manager.save_rectification_matrices(
            matrices, output_paths, pair_name
        )
        
        # Save points data
        rectification_info = self.current_pair_info['rectification_info']
        points_results = self.file_manager.save_rectified_points_data(
            points_data['rectified_interpolated_points'],
            output_paths, pair_name,
            rectification_info['original_size'],
            rectification_info['rectified_size']
        )
        
        # Save optical axis reference data
        ref_results = self.file_manager.save_optical_axis_reference_data(
            points_data['optical_ref_x_points'],
            points_data['optical_ref_y_points'],
            output_paths, pair_name
        )
        
        # Create and save visualizations
        visualizations = self._create_visualizations(
            rectified_left, rectified_right, points_data
        )
        
        viz_results = self.file_manager.save_visualization_results(
            visualizations, output_paths['output'], pair_name
        )
        
        # Create and save comprehensive metadata
        metadata = self._create_comprehensive_metadata(matrices, points_data)
        metadata_results = self.file_manager.save_rectification_metadata(
            metadata, output_paths, pair_name
        )
        
        # Create legacy visualization if needed
        if self.need_show:
            self._create_legacy_visualization(
                rectified_left, rectified_right, output_paths['output'], pair_name
            )
        
        # Log results summary
        all_results = {
            'images': image_results,
            'matrices': matrix_results,
            'points': points_results,
            'references': ref_results,
            'visualizations': viz_results,
            'metadata': metadata_results
        }
        
        self._log_save_results(pair_name, all_results)
    
    def _create_visualizations(
        self,
        rectified_left: np.ndarray,
        rectified_right: np.ndarray,
        points_data: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Create visualization images.
        
        Args:
            rectified_left: Rectified left image
            rectified_right: Rectified right image
            points_data: Processed points data
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of visualization images
        """
        visualizations = {}
        
        # Create basic overlay
        visualizations['rectified_overlay'] = ImageOverlayCreator.create_stereo_overlay(
            rectified_left, rectified_right
        )
        
        # Create marked overlay with interpolated points
        visualizations['marked_overlay'] = ImageOverlayCreator.create_marked_overlay(
            rectified_left, rectified_right,
            points_data['rectified_interpolated_points']
        )
        
        # Create marked left image
        visualizations['marked_left'] = ImageOverlayCreator.create_marked_single_image(
            rectified_left, points_data['rectified_interpolated_points']
        )
        
        return visualizations
    
    def _create_legacy_visualization(
        self,
        rectified_left: np.ndarray,
        rectified_right: np.ndarray,
        output_path: Path,
        pair_name: str
    ) -> None:
        """
        Create legacy visualization using ImageChartGenerator for backward compatibility.
        
        Args:
            rectified_left: Rectified left image
            rectified_right: Rectified right image
            output_path: Output directory path
            pair_name: Name of the image pair
        """
        # Create overlay using legacy method
        overlay = ImageOverlayCreator.create_stereo_overlay(rectified_left, rectified_right)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Use legacy ImageChartGenerator
        painter = ImageChartGenerator(
            img=overlay_bgr,
            xlabel="pixel",
            ylabel="pixel",
            save_path_result=str(output_path),
            need_show=self.need_show
        )
        
        painter.create_rectify(photo_name=f"rectified_{pair_name}")
    
    def _create_comprehensive_metadata(
        self,
        matrices: Dict[str, np.ndarray],
        points_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for the rectification process.
        
        Args:
            matrices: Calculated matrices
            points_data: Processed points data
            
        Returns:
            Dict[str, Any]: Comprehensive metadata
        """
        rectification_info = self.current_pair_info['rectification_info']
        
        # Create matrix metadata
        matrix_metadata = self.matrix_calculator.create_matrix_metadata(
            matrices['P1'], matrices['P2'], matrices['Q_rectified'],
            rectification_info['original_size'],
            rectification_info['rectified_size'],
            getattr(self.resize_strategy, 'last_scale_factor', None)
        )
        
        # Combine with processing metadata
        comprehensive_metadata = {
            'pair_name': self.current_pair_info['pair_name'],
            'set_name': self.current_pair_info['set_name'],
            'processing_version': 'rectifier_refactored_v1.0',
            'rectification_info': rectification_info,
            'matrix_info': matrix_metadata,
            'points_info': {
                'num_interpolated_points': len(points_data['rectified_interpolated_points']),
                'endpoints': {
                    'left': points_data['rectified_interpolated_points'][0] if points_data['rectified_interpolated_points'] else None,
                    'right': points_data['rectified_interpolated_points'][-1] if points_data['rectified_interpolated_points'] else None
                }
            },
            'configuration': {
                'rectified_image_size_mode': self._get_rectification_config().get('rectified_image_size', 'default'),
                'rectification_alpha': self._get_rectification_config().get('rectification_alpha', 0),
                'resize_config': self._get_resize_config()
            }
        }
        
        return comprehensive_metadata
    
    def _log_save_results(self, pair_name: str, results: Dict[str, Dict[str, bool]]) -> None:
        """
        Log summary of save operation results.
        
        Args:
            pair_name: Name of the processed pair
            results: Dictionary of save results by category
        """
        total_operations = sum(len(category_results) for category_results in results.values())
        successful_operations = sum(
            sum(1 for success in category_results.values() if success)
            for category_results in results.values()
        )
        
        self.logger.info(f"Save results for {pair_name}: "
                        f"{successful_operations}/{total_operations} operations successful")
        
        # Log any failures
        for category, category_results in results.items():
            failed_ops = [op for op, success in category_results.items() if not success]
            if failed_ops:
                self.logger.warning(f"Failed {category} operations: {failed_ops}")


# Legacy compatibility function
def createRectifiedStereoPhoto(rectifier_instance):
    """Legacy function name for backward compatibility."""
    return rectifier_instance.create_rectified_stereo_photos()
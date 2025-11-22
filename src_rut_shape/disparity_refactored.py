"""
Refactored stereo disparity calculation module.

This module provides a clean, modular interface for stereo disparity calculation
with improved code organization, error handling, and maintainability.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import datetime

# Import base classes
from .base import BaseProcessor

# Import new modular components
from .disparity.sgbm_engine import SGBMEngine
from .disparity.parameter_calculator import DisparityParameterCalculator
from .disparity.disparity_processor import DisparityProcessor
from .disparity.file_manager import DisparityFileManager

from utils.file_operations import PathManager
from utils.stereo_math import StereoMath
from utils.logger_config import get_logger


class DisparityCalculator(BaseProcessor):
    """
    Main disparity calculation coordinator class.
    
    This class orchestrates the stereo disparity calculation process using modular
    components for improved maintainability and code organization.
    """
    
    def __init__(self, config, resize_scale: float = 1.0):
        """
        Initialize disparity calculator with configuration.
        
        Args:
            config: Configuration object with disparity parameters
            resize_scale: Scale factor applied during rectification
        """
        super().__init__(config, "disparity")
        
        self.resize_scale = resize_scale
        
        # Initialize modular components
        self.sgbm_engine = SGBMEngine()
        self.parameter_calculator = DisparityParameterCalculator()
        self.disparity_processor = DisparityProcessor()
        
        # Initialize file manager
        self.file_manager = DisparityFileManager(self.output_folder, self.temp_folder)
        
        # Setup input folder
        self._setup_input_folder()
        
        # Load camera parameters
        self.camera_parameters = {}
        self._load_camera_parameters()
        
        # Calculate SGBM parameters
        self.sgbm_parameters = {}
        self._calculate_sgbm_parameters()
        
        self.logger.info("DisparityCalculator initialized with modular architecture")
    
    
    def _load_camera_parameters(self) -> None:
        """Load stereo camera parameters from files."""
        self.logger.info("Loading camera parameters")
        
        parameter_files = {
            'K_left': 'K1.csv',
            'K_right': 'K2.csv',
            'T': 'T.csv'
        }
        
        for param_name, filename in parameter_files.items():
            file_path = self.root / self.config.parameter_path / filename
            self.camera_parameters[param_name] = np.genfromtxt(file_path, delimiter=',')
        
        # Calculate derived parameters
        K_left = self.camera_parameters['K_left']
        K_right = self.camera_parameters['K_right']
        T = self.camera_parameters['T']
        
        # Average focal length
        self.camera_parameters['focal_length_avg'] = np.ceil(
            (K_left[0, 0] + K_left[1, 1] + K_right[0, 0] + K_right[1, 1]) / 4
        )
        
        # Baseline
        self.camera_parameters['baseline'] = abs(T[0])
        
        self.logger.info(f"Camera parameters loaded: "
                        f"focal_length={self.camera_parameters['focal_length_avg']:.1f}px, "
                        f"baseline={self.camera_parameters['baseline']:.4f}m")
    
    def _calculate_sgbm_parameters(self) -> None:
        """Calculate optimal SGBM parameters based on camera setup and configuration."""
        self.logger.info("Calculating SGBM parameters")
        
        # Extract configuration parameters
        config_overrides = {
            'SGBM_MODE_FAST': getattr(self.config, 'SGBM_MODE_FAST', 'True'),
            'MIN_DISPARITY': getattr(self.config, 'MIN_DISPARITY', 0),
            'NUM_DISPARITIES': getattr(self.config, 'NUM_DISPARITIES', None),
            'BLOCKSIZE': getattr(self.config, 'BLOCKSIZE', None)
        }
        
        # Calculate parameters using parameter calculator
        self.sgbm_parameters = self.parameter_calculator.calculate_sgbm_parameters(
            camera_matrices=(self.camera_parameters['K_left'], self.camera_parameters['K_right']),
            translation_vector=self.camera_parameters['T'],
            target_depth=self.config.target_depth,
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
            resize_scale=self.resize_scale,
            config_overrides=config_overrides
        )
        
        # Validate parameters
        is_valid, warnings = self.parameter_calculator.validate_parameters(self.sgbm_parameters)
        
        if not is_valid:
            raise ValueError(f"Invalid SGBM parameters: {warnings}")
        
        if warnings:
            for warning in warnings:
                self.logger.warning(warning)
        
        # Configure SGBM engine
        self.sgbm_engine.configure_parameters(
            min_disparity=self.sgbm_parameters['min_disparity'],
            num_disparities=self.sgbm_parameters['num_disparities'],
            block_size=self.sgbm_parameters['block_size'],
            use_fast_mode=self.sgbm_parameters['use_fast_mode']
        )
        
        self.logger.info("SGBM parameters calculated and engine configured")
    
    def _setup_input_folder(self) -> None:
        """Setup input folder path specific to disparity processing."""
        self.input_folder = self.root / Path(self.config.save_path_temp) / "target_pictures_set_rectified"
    
    def _get_processor_specific_config(self) -> Dict[str, Any]:
        """Get disparity-specific configuration parameters."""
        return {
            'target_depth': getattr(self.config, 'target_depth', None),
            'min_depth': getattr(self.config, 'min_depth', None),
            'max_depth': getattr(self.config, 'max_depth', None),
            'SGBM_MODE_FAST': getattr(self.config, 'SGBM_MODE_FAST', 'True'),
            'MIN_DISPARITY': getattr(self.config, 'MIN_DISPARITY', 0),
            'NUM_DISPARITIES': getattr(self.config, 'NUM_DISPARITIES', None),
            'BLOCKSIZE': getattr(self.config, 'BLOCKSIZE', None),
            'resize_scale': self.resize_scale
        }
    
    def _is_processing_ready(self) -> bool:
        """Check if disparity calculator is ready for processing."""
        return (len(self.sgbm_parameters) > 0 and 
                len(self.camera_parameters) > 0 and
                self.file_manager is not None)

    def create_disparity(self) -> None:
        """
        Main entry point for processing all image sets.
        
        This method processes all 'set_*' folders in the input directory.
        """
        self.process_all_sets()
    
    def _execute_processing_pipeline(self, pair_folder: Path) -> Dict[str, Any]:
        """
        Execute the main disparity calculation pipeline for a single pair.
        
        Args:
            pair_folder: Path to the pair folder
            
        Returns:
            Dict[str, Any]: Processing results
        """
        pair_name = pair_folder.name
        
        # Load rectified images
        left_image, right_image = self._load_rectified_images(pair_folder, pair_name)
        if left_image is None or right_image is None:
            raise ValueError(f"Failed to load rectified images for {pair_name}")
        
        # Find rut points file
        rut_points_file = self.file_manager.find_rut_points_file(pair_folder, pair_name)
        if rut_points_file is None:
            raise ValueError(f"No rut points file found for {pair_name}")
        
        # Compute disparity
        disparity = self._compute_disparity(left_image, right_image) * self.resize_scale
        
        # Process disparity results
        processed_results = self._process_disparity_results(disparity, rut_points_file)
        
        return {
            'disparity': disparity,
            'processed_results': processed_results,
            'rut_points_file': rut_points_file
        }
    
    def _save_processing_results(self, processing_results: Dict[str, Any], pair_name: str) -> None:
        """
        Save disparity calculation results using the file manager.
        
        Args:
            processing_results: Results from disparity processing
            pair_name: Name of the image pair
        """
        self._save_complete_results(
            processing_results['disparity'],
            processing_results['processed_results'],
            pair_name
        )
    

    
    def _load_rectified_images(
        self, 
        pair_folder: Path, 
        pair_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load rectified stereo images for disparity processing.
        
        Args:
            pair_folder: Path to pair folder
            pair_name: Name of the pair
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_image, right_image) or (None, None) if failed
        """
        left_image, right_image = self.file_manager.load_rectified_images(pair_folder, pair_name)
        
        if left_image is not None and right_image is not None:
            self.logger.info(f"Loaded rectified images: {left_image.shape}")
        
        return left_image, right_image
    
    def _compute_disparity(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray
    ) -> np.ndarray:
        """
        Compute disparity map from stereo image pair.
        
        Args:
            left_image: Left rectified image
            right_image: Right rectified image
            
        Returns:
            np.ndarray: Computed disparity map
        """
        self.logger.info("Computing disparity using SGBM algorithm")
        
        # Compute raw disparity
        raw_disparity = self.sgbm_engine.compute_disparity(left_image, right_image)
        
        # Normalize disparity
        normalized_disparity = self.disparity_processor.normalize_disparity(raw_disparity)
        
        return normalized_disparity
    
    def _process_disparity_results(
        self, 
        disparity: np.ndarray,
        rut_points_file: Path
    ) -> Dict[str, Any]:
        """
        Process disparity results including quality assessment and rut point extraction.
        
        Args:
            disparity: Computed disparity map
            rut_points_file: Path to rut points JSON file
            
        Returns:
            Dict[str, Any]: Processing results
        """
        results = {}
        
        # Assess disparity quality
        quality_metrics = self.disparity_processor.assess_disparity_quality(disparity)
        results['quality_metrics'] = quality_metrics
        
        # Extract rut disparity values
        try:
            rut_disparity_values, rut_metadata = self.disparity_processor.extract_rut_disparity_values(
                disparity, rut_points_file
            )
            results['rut_disparity_values'] = rut_disparity_values
            results['rut_metadata'] = rut_metadata
        except Exception as e:
            self.logger.error(f"Failed to extract rut disparity values: {e}")
            results['rut_disparity_values'] = []
            results['rut_metadata'] = {}
        
        # Create marked disparity image
        marked_disparity = self.disparity_processor.create_marked_disparity_image(
            disparity, rut_points_file
        )
        results['marked_disparity'] = marked_disparity
        
        return results
    
    def _save_complete_results(
        self,
        disparity: np.ndarray,
        processed_results: Dict[str, Any],
        pair_name: str
    ) -> None:
        """
        Save all disparity calculation results.
        
        Args:
            disparity: Computed disparity map
            processed_results: Processed results from disparity analysis
            pair_name: Name of the image pair
        """
        set_name = self.current_pair_info['set_name']
        
        # Setup output directories
        output_paths = self.file_manager.setup_output_directories(set_name, pair_name)
        
        # # Save disparity map
        disparity_results = self.file_manager.save_disparity_map(
            disparity, {'temp': output_paths['temp']}, pair_name, save_formats=['npy']
        )

        
        # Save disparity visualization
        visualization_params = self.parameter_calculator.calculate_disparity_visualization_range(
            self.camera_parameters['focal_length_avg'],
            self.camera_parameters['baseline'],
            self.config.target_depth,
            self.config.min_depth,
            self.config.max_depth
        )
        
        viz_success = self.file_manager.save_disparity_visualization(
            disparity, output_paths['output'], pair_name, 
            visualization_params, self.need_show
        )
        
        # Save rut disparity values if available
        rut_results = {}
        if processed_results.get('rut_disparity_values'):
            rut_results = self.file_manager.save_rut_disparity_values(
                processed_results['rut_disparity_values'],
                processed_results['rut_metadata'],
                output_paths, pair_name
            )
        
        # Save marked disparity image if available
        marked_success = False
        if processed_results.get('marked_disparity') is not None:
            marked_success = self.file_manager.save_marked_disparity_image(
                processed_results['marked_disparity'],
                output_paths['output'], pair_name
            )
        
        # Create and save comprehensive metadata
        metadata = self._create_comprehensive_metadata(disparity, processed_results)
        metadata_results = self.file_manager.save_disparity_metadata(
            metadata, output_paths, pair_name
        )
        
        # Log results summary
        all_results = {
            'visualization': {'output_viz': viz_success},
            'rut_values': rut_results,
            'marked_image': {'output_marked': marked_success},
            'metadata': metadata_results
        }
        
        self._log_save_results(pair_name, all_results)
    
    def _create_comprehensive_metadata(
        self,
        disparity: np.ndarray,
        processed_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for the disparity processing.
        
        Args:
            disparity: Computed disparity map
            processed_results: Processed results from disparity analysis
            
        Returns:
            Dict[str, Any]: Comprehensive metadata
        """
        # Create disparity metadata using processor
        metadata = self.disparity_processor.create_disparity_metadata(
            disparity, self.sgbm_parameters, processed_results.get('quality_metrics')
        )
        
        # Add processing context
        metadata.update({
            'pair_info': self.current_pair_info,
            'processing_version': 'disparity_calculator_refactored_v1.0',
            'camera_parameters': {
                'focal_length_avg': float(self.camera_parameters['focal_length_avg']),
                'baseline': float(self.camera_parameters['baseline']),
                'resize_scale': self.resize_scale
            },
            'configuration': {
                'target_depth': self.config.target_depth,
                'min_depth': self.config.min_depth,
                'max_depth': self.config.max_depth,
                'need_show': self.need_show
            },
            'rut_analysis': processed_results.get('rut_metadata', {})
        })
        
        return metadata
    
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
    
    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about current processing setup.
        
        Returns:
            Dict[str, Any]: Processing information
        """
        return {
            'camera_parameters': self.camera_parameters,
            'sgbm_parameters': self.sgbm_parameters,
            'sgbm_engine_info': self.sgbm_engine.get_configuration_info(),
            'resize_scale': self.resize_scale,
            'input_folder': str(self.input_folder),
            'processing_ready': len(self.sgbm_parameters) > 0
        }


# Legacy compatibility function
def createDisparity(calculator_instance):
    """Legacy function name for backward compatibility."""
    return calculator_instance.create_disparity()
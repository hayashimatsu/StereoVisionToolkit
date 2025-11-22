"""
Refactored height calculation module for rut shape analysis.

This module serves as the main coordinator for height calculation operations,
delegating specific responsibilities to specialized sub-modules:

- ImageLoader: Handles image and point data loading
- CoordinateProcessor: Manages coordinate transformations  
- RutCalculator: Performs rut depth calculations
- FileManager: Handles all file operations

Input formats:
- Configuration object with processing parameters
- World coordinate files (.npy format)
- Rectified and disparity images

Output formats:
- Processed rut coordinate data
- Rut depth measurements in millimeters
- Intermediate processing files for debugging
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# Import base classes
from .base import BaseProcessor

from utils.rut_visualization import RutDepthVisualizer
from .height.processors import RutShapeProcessor
from .height import ImageLoader, CoordinateProcessor, RutCalculator
from .height.file_manager import HeightFileManager


class HeightCalculator(BaseProcessor):
    """
    Main coordinator for height calculation operations.
    
    Orchestrates the complete height calculation pipeline:
    1. Load images and coordinate data
    2. Process coordinates using configured method
    3. Apply rut shape processing pipeline
    4. Calculate rut depth using selected algorithm
    5. Save results and intermediate data
    
    Attributes:
        config: Configuration object with processing parameters
        root (Path): Root project directory path
        logger: Logger instance for this class
    """
    
    def __init__(self, config):
        """
        Initialize HeightCalculator with configuration.
        
        Args:
            config: Configuration object containing processing parameters
        """
        super().__init__(config, "height")
        
        # Folder name constants (moved from constants.py)
        self.depth_folder_name = "target_pictures_set_depth"
        self.rectify_folder_name = "target_pictures_set_rectified"
        self.disparity_folder_name = "target_pictures_set_disparity"
        self.rut_folder_name = "target_pictures_set_rutShape"
        
        # Processing constants (moved from constants.py)
        self.default_padding_method = "reflection_extrapolation"
        self.default_padding_size_percent = 25.0
        self.need_adjusting_ground_level = config.need_adjusting_ground_level == "True" 
        self.need_rotating_ground_level = config.need_rotating_ground_level == "True" 

        # Initialize additional paths
        self.disparity_folder = None
        self.pic_folder = None
        self._setup_additional_paths()
        
        # Setup input folder
        self._setup_input_folder()
        
        # Initialize sub-modules
        self.rut_processor = RutShapeProcessor(config)
        self.image_loader = ImageLoader(
            self.pic_folder, self.disparity_folder, self.temp_folder, self.root, config)
        self.coordinate_processor = CoordinateProcessor(
            config, self.temp_folder, self.rut_processor)
        self.rut_calculator = RutCalculator()
        self.file_manager = HeightFileManager(self.output_folder, self.temp_folder)
        self.rut_visualizer = RutDepthVisualizer(config)
    
    def _setup_additional_paths(self):
        """Initialize additional folder paths specific to height processing."""
        self.disparity_folder = self.root / Path(self.config.save_path_temp) / Path(self.disparity_folder_name)
        self.pic_folder = self.root / Path(self.config.save_path_temp) / Path(self.rectify_folder_name)
        
        self.logger.info(f"Rectified image folder: {self.pic_folder}")
        self.logger.info(f"Disparity folder: {self.disparity_folder}")

    def _setup_input_folder(self) -> None:
        """Setup input folder path specific to height processing."""
        self.input_folder = self.root / Path(self.config.save_path_temp) / Path(self.depth_folder_name)
    
    def _get_processor_specific_config(self) -> Dict[str, Any]:
        """Get height-specific configuration parameters."""
        return {
            'low_pass_filter_cutoff': getattr(self.config, 'low_pass_filter_cutoff', None),
            'lane_width': getattr(self.config, 'lane_width', None),
            'use_improved_rut_calculation': getattr(self.config, 'use_improved_rut_calculation', 'True'),
            'rut_min_distance_ratio': getattr(self.config, 'rut_min_distance_ratio', 0.1),
            'rut_fallback_to_local_peaks': getattr(self.config, 'rut_fallback_to_local_peaks', 'True'),
            'default_padding_method': self.default_padding_method,
            'default_padding_size_percent': self.default_padding_size_percent
        }
    
    def _is_processing_ready(self) -> bool:
        """Check if height calculator is ready for processing."""
        return (self.input_folder is not None and 
                self.rut_processor is not None and
                self.file_manager is not None)

    def create_height(self):
        """
        Main entry point for height calculation processing.
        
        Processes all image sets found in the input folder.
        """
        # Height processing has a unique workflow, so we override the base method
        self.logger.info(f"Starting to process all image sets for {self.processing_type}")
        
        for set_folder in self.input_folder.glob('set_*'):
            self.process_set(set_folder)
            
        self.logger.info(f"All image sets processed successfully for {self.processing_type}")

    def _execute_processing_pipeline(self, pair_folder: Path) -> Dict[str, Any]:
        """
        Execute the main height calculation pipeline for a single pair.
        
        Height processing is different - it processes world coordinate files directly.
        This method is overridden to handle the unique height processing workflow.
        
        Args:
            pair_folder: Path to the pair folder
            
        Returns:
            Dict[str, Any]: Processing results
        """
        # Height processing works differently - we process world coordinate files
        # This method should not be called directly for height processing
        raise NotImplementedError("Height processing uses a different workflow via process_set")
    
    def _save_processing_results(self, processing_results: Dict[str, Any], pair_name: str) -> None:
        """
        Save height calculation results using the file manager.
        
        Args:
            processing_results: Results from height processing
            pair_name: Name of the image pair
        """
        # This method is not used in height processing due to its unique workflow
        pass
    
    def process_set(self, set_folder: Path):
        """
        Process a single image set.
        
        Args:
            set_folder: Path to the set folder containing world coordinate files
        """
        set_name = set_folder.name
        self.logger.info(f"Processing set: {set_name}")
        
        self._process_world_coord_files(set_folder, set_name)    

    def _process_world_coord_files(self, set_folder: Path, set_name: str):
        """
        Process all world coordinate files in a set folder.
        
        Args:
            set_folder: Path to the set folder
            set_name: Name of the image set
        """
        for world_coord_file in set_folder.glob('*/world_coord_*.npy'):
            pair_name = world_coord_file.parent.name
            self.logger.info(f"Processing world coordinate file: {world_coord_file.name}")
            
            try:
                # Create output folders
                output_pair_folder, temp_pair_folder = self.file_manager.create_output_folders(
                    self.output_folder, self.temp_folder, self.rut_folder_name, set_name, pair_name)
                
                # Process single pair
                rut_data = self._process_single_pair(
                    world_coord_file, set_name, pair_name, output_pair_folder, temp_pair_folder)
                
                if rut_data is not None:
                    self._save_and_analyze_results(
                        rut_data, output_pair_folder, temp_pair_folder, pair_name)
                else:
                    self.logger.warning(f"No valid data processed for set {set_name}-{pair_name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing file {world_coord_file}: {str(e)}")
                continue

    def _process_single_pair(self, world_coord_file: Path, set_name: str, pair_name: str,
                           output_pair_folder: Path, temp_pair_folder: Path) -> Optional[np.ndarray]:
        """
        Process a single image pair to extract rut coordinates.
        
        Args:
            world_coord_file: Path to world coordinate file
            set_name: Name of the image set
            pair_name: Name of the image pair
            output_pair_folder: Output folder for this pair
            temp_pair_folder: Temporary folder for this pair
            
        Returns:
            Processed rut coordinate data, or None if processing failed
        """
        # Load images and rectified points
        rectified_left_pic, disparity_pic = self.image_loader.load_images(set_name, pair_name)
        rut_rectified = self.image_loader.load_rectified_points(set_name, pair_name)
        
        # Process coordinates using configured method
        original_data = self.coordinate_processor.process_coordinates(
            world_coord_file, set_name, pair_name, rectified_left_pic, disparity_pic, rut_rectified)
        
        # Apply rut shape processing pipeline with intermediate data saving
        return self._process_rut_shape_pipeline(
            pair_name, original_data, output_pair_folder, temp_pair_folder)

    def _process_rut_shape_pipeline(self, pair_name: str, original_data: np.ndarray,
                                  output_pair_folder: Path, temp_pair_folder: Path) -> np.ndarray:
        """
        Apply the complete rut shape processing pipeline with intermediate data saving.
        
        Args:
            pair_name: Name of the image pair
            original_data: Original coordinate data
            output_pair_folder: Output folder for saving intermediate data
            temp_pair_folder: Temporary folder for saving intermediate data
            
        Returns:
            Final processed coordinate data
        """
        # Save original data
        self.file_manager.save_intermediate_data(
            original_data, '0-original', output_pair_folder, temp_pair_folder, pair_name)
        
        # Step 1: Apply extreme value filtering
        filtered_data = self.rut_processor._apply_extreme_value_filtering(original_data)
        self.file_manager.save_intermediate_data(
            filtered_data, '1-extreme_filtered', output_pair_folder, temp_pair_folder, pair_name)
        
        # Step 2: Apply rotation calculation
        if self.need_rotating_ground_level == True:
            rotated_data = self.rut_processor._apply_rotation_calculation(filtered_data)
            self.file_manager.save_intermediate_data(
                rotated_data, '2-rotated', output_pair_folder, temp_pair_folder, pair_name)
        else:
            rotated_data = filtered_data
            self.logger.info(f"No rotation applied for {pair_name} because need_rotating_ground_level is False")
        
        # Step 3: Apply depth adjustment
        depth_adjusted_data = self.rut_processor._adjust_depth_to_camera_center(rotated_data)
        self.file_manager.save_intermediate_data(
            depth_adjusted_data, '3-depth_adjusted', output_pair_folder, temp_pair_folder, pair_name)
        
        # Step 4: Apply ground level adjustment
        if self.need_adjusting_ground_level == True:
            baseline_adjusted_data = self.rut_processor._adjust_ground_level(depth_adjusted_data)
            self.file_manager.save_intermediate_data(
                baseline_adjusted_data, '4-baseline_adjusted', output_pair_folder, temp_pair_folder, pair_name)
        else:
            baseline_adjusted_data = depth_adjusted_data
            self.logger.info(f"No ground level adjustment applied for {pair_name} because need_adjusting_ground_level is False")
        
        return baseline_adjusted_data

    def _save_and_analyze_results(self, rut_data: np.ndarray, output_pair_folder: Path,
                                temp_pair_folder: Path, pair_name: str):
        """
        Save rut data and perform depth analysis if filtering is enabled.
        
        Args:
            rut_data: Final rut coordinate data
            output_pair_folder: Output folder for this pair
            temp_pair_folder: Temporary folder for this pair
            pair_name: Name of the image pair
        """
        # # Save basic rut data
        # self.file_manager.save_rut_data(rut_data, output_pair_folder, temp_pair_folder, pair_name)
        # self.logger.info(f"Pair ({pair_name}) processed and saved")
        
        # Apply additional processing if low-pass filter is configured
        if self.config.low_pass_filter_cutoff is not None:
            self._apply_advanced_processing(
                rut_data, output_pair_folder, temp_pair_folder, pair_name)

    def _apply_advanced_processing(self, rut_data: np.ndarray, output_pair_folder: Path,
                                 temp_pair_folder: Path, pair_name: str):
        """
        Apply advanced processing including filtering, scaling, and depth calculation.
        
        Args:
            rut_data: Raw rut coordinate data
            output_pair_folder: Output folder for this pair
            temp_pair_folder: Temporary folder for this pair
            pair_name: Name of the image pair
        """
        # Apply low-pass filter
        filtered_rut_data = self.rut_processor.apply_low_pass_filter(
            rut_data, 
            padding_method=self.default_padding_method,
            padding_size_percent=self.default_padding_size_percent
        )
        
        # Apply width scaling
        scaled_rut_data, scaling_metadata = self._scale_and_translate_coordinates(
            filtered_rut_data, self.config.lane_width)
        
        # Calculate rut depth
        depth_mm = self._calculate_rut_depth(scaled_rut_data, output_pair_folder, pair_name)
        
        # Save results
        self.file_manager.save_filtered_rut_data(
            filtered_rut_data, output_pair_folder, temp_pair_folder, pair_name)
        self.file_manager.save_scaled_rut_data(
            scaled_rut_data, output_pair_folder, temp_pair_folder, pair_name)
        self.file_manager.save_scaling_metadata(
            scaling_metadata, output_pair_folder, temp_pair_folder, pair_name)
        self.file_manager.save_rut_depth_result(depth_mm, output_pair_folder, pair_name)
        
        print(f"Low pass filter適用後のわだちぼれ量: {round(depth_mm, 0)} mm")

    def _calculate_rut_depth(self, scaled_rut_data: np.ndarray, output_pair_folder: Path, 
                           pair_name: str) -> float:
        """
        Calculate rut depth using configured method.
        
        Args:
            scaled_rut_data: Scaled rut coordinate data
            output_pair_folder: Output folder for saving detailed results
            pair_name: Name of the image pair
            
        Returns:
            Rut depth in millimeters
        """
        use_improved = getattr(self.config, 'use_improved_rut_calculation', 'True') == 'True'
        
        if use_improved:
            # Use improved calculation method
            min_dist_ratio = float(getattr(self.config, 'rut_min_distance_ratio', 0.1))
            fallback_local = getattr(self.config, 'rut_fallback_to_local_peaks', 'True') == 'True'
            
            depth_mm, detailed_results = self.rut_calculator.calculate_rut_depth_improved(
                scaled_rut_data, 
                min_distance_ratio=min_dist_ratio,
                fallback_to_local_peaks=fallback_local,
                output_folder=str(output_pair_folder),
                pair_name=pair_name
            )
            
            self.logger.info(f"Used improved rut calculation method for {pair_name}")
            
            # Create visualization
            plot_path = self.rut_visualizer.create_rut_depth_plot(
                scaled_rut_data, detailed_results, str(output_pair_folder), pair_name)
            if plot_path:
                self.logger.info(f"Rut depth visualization saved to: {plot_path}")
                
        else:
            # Use original calculation method
            depth_m = self.rut_calculator.calculate_rut_depth_original(scaled_rut_data)
            depth_mm = depth_m * 1000  # Convert to mm
            self.logger.info(f"Used original rut calculation method for {pair_name}")
        
        return depth_mm

    def _scale_and_translate_coordinates(self, filtered_data: np.ndarray, 
                                       lane_width: float) -> tuple:
        """
        Scale x-coordinates to match lane_width and translate to start at 0.
        
        Args:
            filtered_data: Filtered coordinate data with shape (n, 2)
            lane_width: Target width in meters
        
        Returns:
            Tuple of (scaled_data, scaling_metadata)
        """
        # Use existing data scaling utility
        from utils.data_scaling import scale_width_to_target
        
        scaled_data, scaling_metadata = scale_width_to_target(filtered_data, lane_width)
        
        self.logger.info(f"Applied width scaling: original={scaling_metadata['original_width']:.3f}m, "
                        f"target={lane_width:.3f}m, scale_factor={scaling_metadata['scale_factor']:.3f}")
        
        return scaled_data, scaling_metadata


# Backward compatibility alias
HeightCounterCalculator = HeightCalculator
"""
File management utilities for disparity processing.

This module handles file operations specific to disparity processing,
including structured saving of disparity maps, extracted values, and metadata.
"""

import numpy as np
import json
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from utils.file_operations import DataSaver, PathManager
from utils.image import ImageChartGenerator
from ..base import BaseFileManager

logger = logging.getLogger(__name__)


class DisparityFileManager(BaseFileManager):
    """Manages file operations for disparity processing."""
    
    def __init__(self, base_output_path: Path, base_temp_path: Optional[Path] = None):
        """
        Initialize disparity file manager.
        
        Args:
            base_output_path: Base path for output files
            base_temp_path: Base path for temporary files (optional)
        """
        super().__init__(base_output_path, base_temp_path, "target_pictures_set_disparity")
    
    def get_folder_name(self) -> str:
        """Get the specific folder name for disparity processing."""
        return "target_pictures_set_disparity"
    
    def save_disparity_map(
        self,
        disparity: np.ndarray,
        output_paths: Dict[str, Path],
        pair_name: str,
        save_formats: List[str] = ['npy']
    ) -> Dict[str, bool]:
        """
        Save disparity map in specified formats.
        
        Args:
            disparity: Disparity map to save
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            save_formats: List of formats to save ('npy', 'csv', 'tiff')
            
        Returns:
            Dict[str, bool]: Save results for each format and location
        """
        results = {}
        
        for format_type in save_formats:
            if format_type == 'npy':
                # Save as numpy array (primary format for downstream processing)
                for location_name, path in output_paths.items():
                    success = DataSaver.save_numpy_array(
                        disparity, path, f'disparity_{pair_name}', 'npy'
                    )
                    results[f'{location_name}_npy'] = success
                    
            elif format_type == 'csv':
                # Save as CSV for analysis
                for location_name, path in output_paths.items():
                    success = DataSaver.save_numpy_array(
                        disparity, path, f'disparity_{pair_name}', 'csv'
                    )
                    results[f'{location_name}_csv'] = success
                    
            elif format_type == 'tiff':
                # Save as TIFF for external tools
                for location_name, path in output_paths.items():
                    try:
                        tiff_path = path / f'disparity_{pair_name}.tiff'
                        # Convert to 16-bit for TIFF
                        disparity_16bit = (disparity * 16).astype(np.uint16)
                        cv2.imwrite(str(tiff_path), disparity_16bit)
                        results[f'{location_name}_tiff'] = True
                    except Exception as e:
                        self.logger.error(f"Failed to save TIFF: {e}")
                        results[f'{location_name}_tiff'] = False
        
        self.logger.info(f"Saved disparity map for {pair_name} in formats: {save_formats}")
        return results
    
    def save_disparity_visualization(
        self,
        disparity: np.ndarray,
        output_path: Path,
        pair_name: str,
        visualization_params: Dict[str, Any],
        need_show: bool = False
    ) -> bool:
        """
        Save disparity visualization using ImageChartGenerator.
        
        Args:
            disparity: Raw disparity map (will be normalized internally)
            output_path: Output directory path
            pair_name: Name of the image pair
            visualization_params: Parameters for visualization
            need_show: Whether to show the image
            
        Returns:
            bool: True if successful
        """
        try:
            # Normalize disparity the same way as original implementation
            disparity16 = disparity.astype(np.float32) 
            
            # Extract visualization parameters
            disp_target = visualization_params.get('target_disparity', 0)
            disp_min = visualization_params.get('min_display', 0)
            disp_max = visualization_params.get('max_display', 100)
            
            # Create visualization using ImageChartGenerator with correct parameters
            painter = ImageChartGenerator(
                img=disparity16,
                xlabel="pixel",
                ylabel="pixel",
                save_path_result=str(output_path),
                need_show=need_show,
                range_max=disp_max,
                range_min=disp_min
            )
            
            painter.create_disparity(disp_target, photo_name=f"disparity_{pair_name}")
            
            self.logger.info(f"Saved disparity visualization for {pair_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save disparity visualization: {e}")
            return False
    
    def save_rut_disparity_values(
        self,
        disparity_values: List[float],
        metadata: Dict[str, Any],
        output_paths: Dict[str, Path],
        pair_name: str
    ) -> Dict[str, bool]:
        """
        Save extracted rut disparity values and metadata.
        
        Args:
            disparity_values: List of disparity values at rut points
            metadata: Metadata about the extraction
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            
        Returns:
            Dict[str, bool]: Save results for each location
        """
        results = {}
        
        # Convert to numpy array for saving
        disparity_array = np.array(disparity_values)
        
        # Save CSV files
        locations = [output_paths[key] for key in output_paths.keys()]
        csv_results = DataSaver.save_to_multiple_locations(
            disparity_array, locations, f'rut_disparity_{pair_name}', format_type='csv'
        )
        
        # Save metadata as JSON
        json_results = DataSaver.save_to_multiple_locations(
            metadata, locations, f'rut_disparity_metadata_{pair_name}'
        )
        
        # Combine results
        for location, success in csv_results.items():
            location_name = next(
                (key for key, path in output_paths.items() if path == location),
                'unknown'
            )
            results[f'{location_name}_csv'] = success
            
        for location, success in json_results.items():
            location_name = next(
                (key for key, path in output_paths.items() if path == location),
                'unknown'
            )
            results[f'{location_name}_metadata'] = success
        
        self.logger.info(f"Saved rut disparity values for {pair_name}: {len(disparity_values)} points")
        return results
    
    def save_marked_disparity_image(
        self,
        marked_image: np.ndarray,
        output_path: Path,
        pair_name: str,
        quality: int = 95
    ) -> bool:
        """
        Save marked disparity image.
        
        Args:
            marked_image: Disparity image with markers
            output_path: Output directory path
            pair_name: Name of the image pair
            quality: JPEG quality
            
        Returns:
            bool: True if successful
        """
        try:
            filename = f'disparity_marked_{pair_name}.jpg'
            full_path = output_path / filename
            
            cv2.imwrite(str(full_path), marked_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            self.logger.info(f"Saved marked disparity image: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save marked disparity image: {e}")
            return False
    
    def save_disparity_metadata(
        self,
        metadata: Dict[str, Any],
        output_paths: Dict[str, Path],
        pair_name: str
    ) -> Dict[str, bool]:
        """
        Save comprehensive disparity processing metadata.
        
        Args:
            metadata: Metadata dictionary
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            
        Returns:
            Dict[str, bool]: Save results for each location
        """
        # Save to all locations
        locations = [output_paths[key] for key in output_paths.keys()]
        
        save_results = DataSaver.save_to_multiple_locations(
            metadata, locations, f"disparity_metadata_{pair_name}"
        )
        
        # Convert results to use location names
        results = {}
        for location, success in save_results.items():
            location_name = next(
                (key for key, path in output_paths.items() if path == location),
                'unknown'
            )
            results[f'{location_name}_metadata'] = success
        
        self.logger.info(f"Saved disparity metadata for {pair_name}")
        return results
    
    def create_processing_summary(
        self,
        pair_name: str,
        processing_results: Dict[str, Any],
        save_results: Dict[str, Dict[str, bool]]
    ) -> Dict[str, Any]:
        """
        Create a comprehensive processing summary.
        
        Args:
            pair_name: Name of the processed image pair
            processing_results: Results from disparity processing
            save_results: Results from file save operations
            
        Returns:
            Dict[str, Any]: Processing summary
        """
        # Count successful saves
        total_saves = 0
        successful_saves = 0
        
        for category, results in save_results.items():
            for operation, success in results.items():
                total_saves += 1
                if success:
                    successful_saves += 1
        
        summary = {
            'pair_name': pair_name,
            'processing_timestamp': processing_results.get('timestamp'),
            'disparity_info': processing_results.get('disparity_info', {}),
            'quality_metrics': processing_results.get('quality_metrics', {}),
            'file_operations': {
                'total_saves': total_saves,
                'successful_saves': successful_saves,
                'success_rate': successful_saves / total_saves if total_saves > 0 else 0,
                'save_results': save_results
            },
            'processing_status': 'completed' if successful_saves == total_saves else 'partial',
            'warnings': processing_results.get('warnings', []),
            'errors': processing_results.get('errors', [])
        }
        
        return summary
    
    def load_rectified_images(
        self, 
        rectified_folder: Path, 
        pair_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load rectified stereo images for disparity processing.
        
        Args:
            rectified_folder: Path to rectified images folder
            pair_name: Name of the image pair
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_image, right_image) or (None, None) if failed
        """
        try:
            left_path = rectified_folder / f'left_rectified_{pair_name}.npy'
            right_path = rectified_folder / f'right_rectified_{pair_name}.npy'
            
            if not left_path.exists() or not right_path.exists():
                self.logger.error(f"Rectified images not found for pair {pair_name}")
                return None, None
            
            left_image = np.load(left_path)
            right_image = np.load(right_path)
            
            self.logger.info(f"Loaded rectified images for {pair_name}: {left_image.shape}")
            return left_image, right_image
            
        except Exception as e:
            self.logger.error(f"Failed to load rectified images for {pair_name}: {e}")
            return None, None
    
    def find_rut_points_file(
        self, 
        rectified_folder: Path, 
        pair_name: str
    ) -> Optional[Path]:
        """
        Find rut points JSON file for the given pair.
        
        Args:
            rectified_folder: Path to rectified images folder
            pair_name: Name of the image pair
            
        Returns:
            Path or None: Path to rut points file, or None if not found
        """
        rut_points_file = rectified_folder / f'rectified_interpolated_points_{pair_name}.json'
        
        if rut_points_file.exists():
            return rut_points_file
        else:
            self.logger.warning(f"Rut points file not found for pair {pair_name}")
            return None
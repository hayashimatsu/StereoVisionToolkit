"""
File management utilities for stereo rectification.

This module handles file operations specific to rectification processing,
including structured saving of rectification results, matrices, and metadata.
"""

import numpy as np
import json
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from utils.file_operations import DataSaver, MetadataSaver, PathManager
from utils.visualizer import VisualizationSaver
from ..base import BaseFileManager

logger = logging.getLogger(__name__)


class RectificationFileManager(BaseFileManager):
    """Manages file operations for rectification processing."""
    
    def __init__(self, base_output_path: Path, base_temp_path: Optional[Path] = None):
        """
        Initialize rectification file manager.
        
        Args:
            base_output_path: Base path for output files
            base_temp_path: Base path for temporary files (optional)
        """
        super().__init__(base_output_path, base_temp_path, "target_pictures_set_rectified")
    
    def get_folder_name(self) -> str:
        """Get the specific folder name for rectification processing."""
        return "target_pictures_set_rectified"
    
    def save_rectified_images(
        self,
        rectified_left: np.ndarray,
        rectified_right: np.ndarray,
        output_paths: Dict[str, Path],
        pair_name: str,
        save_formats: List[str] = ['jpg', 'npy']
    ) -> Dict[str, bool]:
        """
        Save rectified images in multiple formats.
        
        Args:
            rectified_left: Rectified left image
            rectified_right: Rectified right image
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            save_formats: List of formats to save ('jpg', 'npy', 'png')
            
        Returns:
            Dict[str, bool]: Save results for each format and location
        """
        results = {}
        
        for format_type in save_formats:
            if format_type in ['jpg', 'jpeg', 'png']:
                # Save as image files
                for location_name, path in output_paths.items():
                    left_filename = f'left_rectified_{pair_name}.{format_type}'
                    right_filename = f'right_rectified_{pair_name}.{format_type}'
                    
                    left_success = cv2.imwrite(str(path / left_filename), rectified_left)
                    right_success = cv2.imwrite(str(path / right_filename), rectified_right)
                    
                    results[f'{location_name}_{format_type}_left'] = left_success
                    results[f'{location_name}_{format_type}_right'] = right_success
                    
            elif format_type == 'npy':
                # Save as numpy arrays (typically for temp storage)
                if 'temp' in output_paths:
                    temp_path = output_paths['temp']
                    
                    left_success = DataSaver.save_numpy_array(
                        rectified_left, temp_path, f'left_rectified_{pair_name}', 'npy'
                    )
                    right_success = DataSaver.save_numpy_array(
                        rectified_right, temp_path, f'right_rectified_{pair_name}', 'npy'
                    )
                    
                    results[f'temp_npy_left'] = left_success
                    results[f'temp_npy_right'] = right_success
        
        self.logger.info(f"Saved rectified images for {pair_name} in formats: {save_formats}")
        return results
    
    def save_rectification_matrices(
        self,
        matrices: Dict[str, np.ndarray],
        output_paths: Dict[str, Path],
        pair_name: str
    ) -> Dict[str, bool]:
        """
        Save rectification matrices (P1, P2, Q, etc.).
        
        Args:
            matrices: Dictionary of matrices to save
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            
        Returns:
            Dict[str, bool]: Save results for each matrix and location
        """
        results = {}
        
        # Standard matrix names and their corresponding arrays
        matrix_files = {
            'P1': f'P1_{pair_name}',
            'P2': f'P2_{pair_name}',
            'P1_original': f'P1_{pair_name}_org',
            'P2_original': f'P2_{pair_name}_org',
            'Q_rectified': f'Q_rectified_{pair_name}',
            'Q_opencv': f'Q_fromCV_{pair_name}',
            'R1': f'R1_{pair_name}',
            'R2': f'R2_{pair_name}'
        }
        
        for matrix_key, filename in matrix_files.items():
            if matrix_key in matrices:
                matrix = matrices[matrix_key]
                
                # Save to all specified locations
                locations = [output_paths[key] for key in output_paths.keys()]
                save_results = DataSaver.save_to_multiple_locations(
                    matrix, locations, filename, format_type='csv'
                )
                
                # Record results
                for location, success in save_results.items():
                    location_name = next(
                        (key for key, path in output_paths.items() if path == location),
                        'unknown'
                    )
                    results[f'{location_name}_{matrix_key}'] = success
        
        self.logger.info(f"Saved rectification matrices for {pair_name}")
        return results
    
    def save_rectified_points_data(
        self,
        rectified_points: List[List[int]],
        output_paths: Dict[str, Path],
        pair_name: str,
        original_size: Tuple[int, int],
        rectified_size: Tuple[int, int]
    ) -> Dict[str, bool]:
        """
        Save rectified interpolated points data.
        
        Args:
            rectified_points: List of rectified point coordinates
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            original_size: Original image size
            rectified_size: Rectified image size
            
        Returns:
            Dict[str, bool]: Save results
        """
        results = {}
        
        # Create structured data for points
        points_data = {
            "source_image": f"left_{pair_name}.jpg",
            "original_size": list(original_size),
            "rectified_size": list(rectified_size),
            "interpolated_points": rectified_points,
            "endpoints": {
                "left": rectified_points[0] if rectified_points else None,
                "right": rectified_points[-1] if rectified_points else None
            },
            "num_interpolated_points": len(rectified_points),
            "labels": {"left": "rut_1", "right": "rut_2"},
            "version": "rectified_interpolated_points_v1"
        }
        
        # Save as JSON
        locations = [output_paths[key] for key in output_paths.keys()]
        json_results = DataSaver.save_to_multiple_locations(
            points_data, locations, f"rectified_interpolated_points_{pair_name}"
        )
        
        # Save as CSV
        points_array = np.array(rectified_points)
        csv_results = DataSaver.save_to_multiple_locations(
            points_array, locations, f'rectified_interpolated_points_{pair_name}', format_type='csv'
        )
        
        # Combine results
        for location, success in json_results.items():
            location_name = next(
                (key for key, path in output_paths.items() if path == location),
                'unknown'
            )
            results[f'{location_name}_points_json'] = success
            
        for location, success in csv_results.items():
            location_name = next(
                (key for key, path in output_paths.items() if path == location),
                'unknown'
            )
            results[f'{location_name}_points_csv'] = success
        
        self.logger.info(f"Saved rectified points data for {pair_name}")
        return results
    
    def save_optical_axis_reference_data(
        self,
        ref_x_points: List[List[int]],
        ref_y_points: List[List[int]],
        output_paths: Dict[str, Path],
        pair_name: str
    ) -> Dict[str, bool]:
        """
        Save optical axis reference points data.
        
        Args:
            ref_x_points: X-axis reference points
            ref_y_points: Y-axis reference points
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            
        Returns:
            Dict[str, bool]: Save results
        """
        results = {}
        
        # Convert to numpy arrays
        ref_x_array = np.array(ref_x_points)
        ref_y_array = np.array(ref_y_points)
        
        # Save to all locations
        locations = [output_paths[key] for key in output_paths.keys()]
        
        x_results = DataSaver.save_to_multiple_locations(
            ref_x_array, locations, f'rect_interpolated_RefX_{pair_name}', format_type='csv'
        )
        
        y_results = DataSaver.save_to_multiple_locations(
            ref_y_array, locations, f'rect_interpolated_RefY_{pair_name}', format_type='csv'
        )
        
        # Record results
        for location, success in x_results.items():
            location_name = next(
                (key for key, path in output_paths.items() if path == location),
                'unknown'
            )
            results[f'{location_name}_ref_x'] = success
            
        for location, success in y_results.items():
            location_name = next(
                (key for key, path in output_paths.items() if path == location),
                'unknown'
            )
            results[f'{location_name}_ref_y'] = success
        
        self.logger.info(f"Saved optical axis reference data for {pair_name}")
        return results
    
    def save_rectification_metadata(
        self,
        metadata: Dict[str, Any],
        output_paths: Dict[str, Path],
        pair_name: str
    ) -> Dict[str, bool]:
        """
        Save comprehensive rectification metadata.
        
        Args:
            metadata: Metadata dictionary
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            
        Returns:
            Dict[str, bool]: Save results
        """
        # Save to all locations
        locations = [output_paths[key] for key in output_paths.keys()]
        
        save_results = DataSaver.save_to_multiple_locations(
            metadata, locations, f"rectification_metadata_{pair_name}"
        )
        
        # Convert results to use location names
        results = {}
        for location, success in save_results.items():
            location_name = next(
                (key for key, path in output_paths.items() if path == location),
                'unknown'
            )
            results[f'{location_name}_metadata'] = success
        
        self.logger.info(f"Saved rectification metadata for {pair_name}")
        return results
    
    def save_visualization_results(
        self,
        visualizations: Dict[str, np.ndarray],
        output_path: Path,
        pair_name: str,
        quality: int = 95
    ) -> Dict[str, bool]:
        """
        Save visualization images.
        
        Args:
            visualizations: Dictionary of visualization images
            output_path: Output directory path
            pair_name: Name of the image pair
            quality: JPEG quality
            
        Returns:
            Dict[str, bool]: Save results for each visualization
        """
        results = {}
        
        # Standard visualization filenames
        visualization_files = {
            'rectified_overlay': f'rectified_{pair_name}.jpg',
            'marked_overlay': f'rectified_marked_{pair_name}.jpg',
            'marked_left': f'rectified_marked_left_{pair_name}.jpg',
            'optical_center': f'optical_center_{pair_name}.jpg'
        }
        
        for viz_key, filename in visualization_files.items():
            if viz_key in visualizations:
                success = VisualizationSaver.save_visualization(
                    visualizations[viz_key], output_path, filename, quality
                )
                results[viz_key] = success
        
        self.logger.info(f"Saved visualization results for {pair_name}")
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
            processing_results: Results from rectification processing
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
            'rectification_info': processing_results.get('rectification_info', {}),
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
    
    def cleanup_temporary_files(self, temp_path: Path, keep_patterns: List[str] = None) -> bool:
        """
        Clean up temporary files, optionally keeping files matching certain patterns.
        
        Args:
            temp_path: Path to temporary directory
            keep_patterns: List of filename patterns to keep (e.g., ['*.npy'])
            
        Returns:
            bool: True if cleanup successful
        """
        if not temp_path.exists():
            return True
            
        try:
            if keep_patterns:
                # Remove only files not matching keep patterns
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        should_keep = any(
                            file_path.match(pattern) for pattern in keep_patterns
                        )
                        if not should_keep:
                            file_path.unlink()
            else:
                # Remove entire temp directory
                import shutil
                shutil.rmtree(temp_path)
                
            self.logger.info(f"Cleaned up temporary files in {temp_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup temporary files: {e}")
            return False
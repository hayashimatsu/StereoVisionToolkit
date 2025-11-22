"""
File management operations module for height calculation.

This module handles:
- Saving intermediate processing data
- Managing output and temporary file operations
- Coordinate data scaling and metadata management
- Unified file saving interface with error handling

Input formats:
- Data arrays: numpy arrays with coordinate data
- Metadata: dictionaries with scaling/processing information
- Folder paths: Path objects for output locations

Output formats:
- CSV files: Coordinate data without headers
- JSON files: Metadata and scaling information
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List
from ..base import BaseFileManager


class HeightFileManager(BaseFileManager):
    """
    Handles file operations for height calculation results.
    
    Provides unified interface for saving different types of data:
    - Raw coordinate data
    - Intermediate processing results  
    - Filtered and scaled data
    - Metadata and scaling information
    
    All save operations write to both output and temporary folders for redundancy.
    """
    
    def __init__(self, base_output_path: Path, base_temp_path: Path = None):
        """
        Initialize HeightFileManager.
        
        Args:
            base_output_path: Base path for output files
            base_temp_path: Base path for temporary files (optional)
        """
        super().__init__(base_output_path, base_temp_path, "target_pictures_set_rutShape")
    
    def get_folder_name(self) -> str:
        """Get the specific folder name for height processing."""
        return "target_pictures_set_rutShape"

    def save_intermediate_data(self, data: np.ndarray, stage_name: str, 
                             output_folder: Path, temp_folder: Path, pair_name: str):
        """
        Save intermediate processing data to both output and temp folders.
        
        Args:
            data: Coordinate data array with shape (n, 2) or (n, 3)
            stage_name: Processing stage identifier (e.g., '1-extreme_filtered')
            output_folder: Output folder path
            temp_folder: Temporary folder path
            pair_name: Image pair name for file naming
        """
        filename = f'rut_{stage_name}_{pair_name}.csv'
        columns = ['x', 'y'] if data.shape[1] == 2 else ['x', 'y', 'z']
        
        df = pd.DataFrame(data, columns=columns)
        
        for folder in [output_folder, temp_folder]:
            file_path = folder / filename
            try:
                df.to_csv(file_path, index=False, header=False)
                self.logger.debug(f"Saved intermediate data to: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save intermediate data to {file_path}: {e}")

    def save_rut_data(self, data: np.ndarray, output_folder: Path, 
                     temp_folder: Path, pair_name: str):
        """
        Save final rut coordinate data.
        
        Args:
            data: Final rut coordinate data with shape (n, 2)
            output_folder: Output folder path
            temp_folder: Temporary folder path
            pair_name: Image pair name for file naming
        """
        filename = f'rut_{pair_name}.csv'
        df = pd.DataFrame(data, columns=['x', 'y'])
        
        for folder in [output_folder, temp_folder]:
            file_path = folder / filename
            try:
                df.to_csv(file_path, index=False, header=False)
                self.logger.info(f"Saved rut data to: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save rut data to {file_path}: {e}")

    def save_filtered_rut_data(self, data: np.ndarray, output_folder: Path, 
                              temp_folder: Path, pair_name: str):
        """
        Save filtered rut coordinate data.
        
        Args:
            data: Filtered rut coordinate data with shape (n, 2)
            output_folder: Output folder path
            temp_folder: Temporary folder path
            pair_name: Image pair name for file naming
        """
        filename = f'rut_filtered_{pair_name}.csv'
        df = pd.DataFrame(data, columns=['x', 'y'])
        
        for folder in [output_folder, temp_folder]:
            file_path = folder / filename
            try:
                df.to_csv(file_path, index=False, header=False)
                self.logger.info(f"Saved filtered rut data to: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save filtered rut data to {file_path}: {e}")

    def save_scaled_rut_data(self, data: np.ndarray, output_folder: Path, 
                           temp_folder: Path, pair_name: str):
        """
        Save scaled rut coordinate data.
        
        Args:
            data: Scaled rut coordinate data with shape (n, 2)
            output_folder: Output folder path
            temp_folder: Temporary folder path
            pair_name: Image pair name for file naming
        """
        filename = f'rut_filtered_scaled_{pair_name}.csv'
        df = pd.DataFrame(data, columns=['x', 'y'])
        
        for folder in [output_folder, temp_folder]:
            file_path = folder / filename
            try:
                df.to_csv(file_path, index=False, header=False)
                self.logger.info(f"Saved scaled rut data to: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save scaled rut data to {file_path}: {e}")

    def save_scaling_metadata(self, metadata: Dict[str, Any], output_folder: Path, 
                            temp_folder: Path, pair_name: str):
        """
        Save scaling metadata to JSON files.
        
        Args:
            metadata: Scaling metadata dictionary
            output_folder: Output folder path
            temp_folder: Temporary folder path
            pair_name: Image pair name for file naming
        """
        filename = f'rut_scaling_metadata_{pair_name}.json'
        
        for folder in [output_folder, temp_folder]:
            file_path = folder / filename
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved scaling metadata to: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save scaling metadata to {file_path}: {e}")

    def save_rut_depth_result(self, depth_mm: float, output_folder: Path, pair_name: str):
        """
        Save final rut depth result to CSV file.
        
        Args:
            depth_mm: Rut depth in millimeters
            output_folder: Output folder path
            pair_name: Image pair name for logging
        """
        file_path = output_folder / 'わだちぼれの深さ.csv'
        try:
            np.savetxt(file_path, [depth_mm], delimiter=',')
            self.logger.info(f"Saved rut depth result ({depth_mm:.0f} mm) to: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save rut depth result to {file_path}: {e}")

    def create_output_folders(self, output_base: Path, temp_base: Path, 
                            rut_folder_name: str, set_name: str, pair_name: str) -> tuple:
        """
        Create and prepare output and temporary folders for a pair.
        
        Legacy method for backward compatibility. Uses base class functionality.
        
        Args:
            output_base: Base output folder path
            temp_base: Base temporary folder path
            rut_folder_name: Rut folder name (e.g., 'target_pictures_set_rutShape')
            set_name: Image set name
            pair_name: Image pair name
            
        Returns:
            Tuple of (output_pair_folder, temp_pair_folder) as Path objects
        """
        # Use base class method
        paths = self.setup_output_directories(set_name, pair_name)
        
        output_pair_folder = paths['output']
        temp_pair_folder = paths.get('temp', output_pair_folder)
        
        return output_pair_folder, temp_pair_folder
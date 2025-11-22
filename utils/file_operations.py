"""
File operation utilities for stereo vision applications.

This module provides common file operations including path management,
batch file operations, and structured data saving that can be used
across multiple modules.
"""

import json
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import shutil
import os

logger = logging.getLogger(__name__)


class PathManager:
    """Manages paths and directory operations for stereo vision projects."""
    
    @staticmethod
    def ensure_directory_exists(path: Path, clear_if_exists: bool = False) -> Path:
        """
        Ensure directory exists, optionally clearing it if it already exists.
        
        Args:
            path: Directory path to create
            clear_if_exists: Whether to clear directory if it already exists
            
        Returns:
            Path: The created/validated directory path
        """
        if clear_if_exists and path.exists():
            shutil.rmtree(path)
            
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
        
        return path
    
    @staticmethod
    def create_output_structure(
        base_path: Path,
        set_name: str,
        pair_name: str,
        create_temp: bool = True
    ) -> Dict[str, Path]:
        """
        Create standard output directory structure.
        
        Args:
            base_path: Base output path
            set_name: Name of the image set
            pair_name: Name of the image pair
            create_temp: Whether to create temporary directories
            
        Returns:
            Dict[str, Path]: Dictionary of created paths
        """
        paths = {}
        
        # Main output directory
        output_dir = base_path / set_name / pair_name
        paths['output'] = PathManager.ensure_directory_exists(output_dir)
        
        # Temporary directory if requested
        if create_temp:
            temp_dir = base_path.parent / "temp" / set_name / pair_name
            paths['temp'] = PathManager.ensure_directory_exists(temp_dir, clear_if_exists=True)
        
        logger.info(f"Created output structure for {set_name}/{pair_name}")
        return paths
    
    @staticmethod
    def validate_input_structure(input_path: Path) -> List[Path]:
        """
        Validate and return list of set directories in input path.
        
        Args:
            input_path: Input directory path
            
        Returns:
            List[Path]: List of valid set directories
            
        Raises:
            ValueError: If no valid set directories found
        """
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
            
        set_folders = list(input_path.glob('set_*'))
        
        if not set_folders:
            raise ValueError(f"No 'set_*' folders found in {input_path}")
            
        logger.info(f"Found {len(set_folders)} set folders in {input_path}")
        return sorted(set_folders)


class DataSaver:
    """Handles saving of various data types in standard formats."""
    
    @staticmethod
    def save_numpy_array(
        array: np.ndarray,
        output_path: Path,
        filename: str,
        format_type: str = 'csv'
    ) -> bool:
        """
        Save numpy array in specified format.
        
        Args:
            array: Numpy array to save
            output_path: Output directory
            filename: Output filename (without extension)
            format_type: Format ('csv', 'npy', 'txt')
            
        Returns:
            bool: True if successful
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            if format_type == 'csv':
                full_path = output_path / f"{filename}.csv"
                np.savetxt(full_path, array, delimiter=',')
            elif format_type == 'npy':
                full_path = output_path / f"{filename}.npy"
                np.save(full_path, array)
            elif format_type == 'txt':
                full_path = output_path / f"{filename}.txt"
                np.savetxt(full_path, array)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            logger.debug(f"Saved array to {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save array {filename}: {e}")
            return False
    
    @staticmethod
    def save_json_data(
        data: Dict[str, Any],
        output_path: Path,
        filename: str,
        indent: int = 2
    ) -> bool:
        """
        Save dictionary data as JSON.
        
        Args:
            data: Data to save
            output_path: Output directory
            filename: Output filename (without extension)
            indent: JSON indentation
            
        Returns:
            bool: True if successful
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            full_path = output_path / f"{filename}.json"
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
                
            logger.debug(f"Saved JSON to {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON {filename}: {e}")
            return False
    
    @staticmethod
    def save_to_multiple_locations(
        data: Union[np.ndarray, Dict[str, Any]],
        locations: List[Path],
        filename: str,
        **save_kwargs
    ) -> Dict[Path, bool]:
        """
        Save data to multiple locations.
        
        Args:
            data: Data to save (numpy array or dictionary)
            locations: List of output directories
            filename: Filename (without extension)
            **save_kwargs: Additional arguments for save functions
            
        Returns:
            Dict[Path, bool]: Results for each location
        """
        results = {}
        
        for location in locations:
            if isinstance(data, np.ndarray):
                success = DataSaver.save_numpy_array(data, location, filename, **save_kwargs)
            elif isinstance(data, dict):
                success = DataSaver.save_json_data(data, location, filename, **save_kwargs)
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                success = False
                
            results[location] = success
            
        return results


class MetadataSaver:
    """Handles saving of metadata and processing information."""
    
    @staticmethod
    def create_processing_metadata(
        original_size: Tuple[int, int],
        processed_size: Tuple[int, int],
        processing_params: Dict[str, Any],
        version: str = "v1.0"
    ) -> Dict[str, Any]:
        """
        Create standardized processing metadata.
        
        Args:
            original_size: (width, height) of original data
            processed_size: (width, height) of processed data
            processing_params: Parameters used in processing
            version: Version identifier
            
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        metadata = {
            "version": version,
            "original_size": list(original_size),
            "processed_size": list(processed_size),
            "processing_params": processing_params,
            "size_change_ratio": {
                "width": processed_size[0] / original_size[0],
                "height": processed_size[1] / original_size[1]
            }
        }
        
        return metadata
    
    @staticmethod
    def save_rectification_metadata(
        output_paths: List[Path],
        pair_name: str,
        original_size: Tuple[int, int],
        rectified_size: Tuple[int, int],
        rectification_params: Dict[str, Any]
    ) -> Dict[Path, bool]:
        """
        Save rectification-specific metadata.
        
        Args:
            output_paths: List of output directories
            pair_name: Name of the image pair
            original_size: Original image size
            rectified_size: Rectified image size
            rectification_params: Rectification parameters
            
        Returns:
            Dict[Path, bool]: Save results for each path
        """
        metadata = MetadataSaver.create_processing_metadata(
            original_size, rectified_size, rectification_params, "rectification_v1.0"
        )
        
        # Add rectification-specific fields
        metadata.update({
            "pair_name": pair_name,
            "rectification_method": rectification_params.get("method", "opencv_stereoRectify"),
            "alpha": rectification_params.get("alpha", 0),
            "resize_applied": original_size != rectified_size
        })
        
        return DataSaver.save_to_multiple_locations(
            metadata, output_paths, f"rectification_metadata_{pair_name}"
        )


class BatchFileProcessor:
    """Handles batch file operations for multiple image sets."""
    
    def __init__(self, input_path: Path, output_path: Path):
        """
        Initialize batch processor.
        
        Args:
            input_path: Input directory containing set folders
            output_path: Output directory for results
        """
        self.input_path = input_path
        self.output_path = output_path
        self.logger = logging.getLogger(__name__)
    
    def get_image_pairs(self, set_folder: Path) -> List[Path]:
        """
        Get list of image pair directories in a set folder.
        
        Args:
            set_folder: Path to set folder
            
        Returns:
            List[Path]: List of pair directories
        """
        pair_folders = [p for p in set_folder.iterdir() if p.is_dir()]
        return sorted(pair_folders)
    
    def validate_image_pair(self, pair_folder: Path, pair_name: str) -> bool:
        """
        Validate that required images exist in pair folder.
        
        Args:
            pair_folder: Path to pair folder
            pair_name: Expected pair name
            
        Returns:
            bool: True if valid pair
        """
        left_image = pair_folder / f'left_{pair_name}.jpg'
        right_image = pair_folder / f'right_{pair_name}.jpg'
        
        if not left_image.exists():
            self.logger.warning(f"Missing left image: {left_image}")
            return False
            
        if not right_image.exists():
            self.logger.warning(f"Missing right image: {right_image}")
            return False
            
        return True
    
    def process_all_sets(self, processor_func, **processor_kwargs) -> Dict[str, Any]:
        """
        Process all sets using provided processor function.
        
        Args:
            processor_func: Function to process each image pair
            **processor_kwargs: Additional arguments for processor function
            
        Returns:
            Dict[str, Any]: Processing results summary
        """
        set_folders = PathManager.validate_input_structure(self.input_path)
        
        results = {
            'total_sets': len(set_folders),
            'processed_sets': 0,
            'total_pairs': 0,
            'processed_pairs': 0,
            'failed_pairs': [],
            'set_results': {}
        }
        
        for set_folder in set_folders:
            set_name = set_folder.name
            self.logger.info(f"Processing set: {set_name}")
            
            pair_folders = self.get_image_pairs(set_folder)
            set_result = {
                'total_pairs': len(pair_folders),
                'processed_pairs': 0,
                'failed_pairs': []
            }
            
            for pair_folder in pair_folders:
                pair_name = pair_folder.name
                
                if not self.validate_image_pair(pair_folder, pair_name):
                    set_result['failed_pairs'].append(pair_name)
                    results['failed_pairs'].append(f"{set_name}/{pair_name}")
                    continue
                
                try:
                    # Call processor function
                    processor_func(
                        set_name=set_name,
                        pair_name=pair_name,
                        pair_folder=pair_folder,
                        output_path=self.output_path,
                        **processor_kwargs
                    )
                    
                    set_result['processed_pairs'] += 1
                    results['processed_pairs'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {set_name}/{pair_name}: {e}")
                    set_result['failed_pairs'].append(pair_name)
                    results['failed_pairs'].append(f"{set_name}/{pair_name}")
            
            results['total_pairs'] += set_result['total_pairs']
            results['set_results'][set_name] = set_result
            
            if set_result['processed_pairs'] > 0:
                results['processed_sets'] += 1
                
            self.logger.info(f"Set {set_name}: {set_result['processed_pairs']}/{set_result['total_pairs']} pairs processed")
        
        return results


class ConfigurationManager:
    """Manages configuration files and parameters."""
    
    @staticmethod
    def load_config_file(config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration data
            
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    
    @staticmethod
    def save_config_file(config: Dict[str, Any], config_path: Path) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration data
            config_path: Path to save configuration
            
        Returns:
            bool: True if successful
        """
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            return False
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged
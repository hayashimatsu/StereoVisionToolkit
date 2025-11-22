"""
Base file management utilities for stereo vision processing.

This module provides a unified base class for file operations across
rectification, disparity, and height processing modules.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from utils.file_operations import PathManager, MetadataSaver


class BaseFileManager(ABC):
    """
    Base class for file management operations across stereo vision modules.
    
    Provides common functionality for:
    - Directory structure setup
    - Metadata saving
    - Error handling and logging
    - Processing summaries
    - Temporary file cleanup
    
    Subclasses should implement module-specific save operations.
    """
    
    def __init__(self, base_output_path: Path, base_temp_path: Optional[Path] = None, 
                 folder_name: str = ""):
        """
        Initialize base file manager.
        
        Args:
            base_output_path: Base path for output files
            base_temp_path: Base path for temporary files (optional)
            folder_name: Specific folder name for this processing type
        """
        self.base_output_path = Path(base_output_path)
        self.base_temp_path = Path(base_temp_path) if base_temp_path else None
        self.folder_name = folder_name
        self.logger = self._setup_logger()
        
        # Processing statistics
        self.processing_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this class."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def setup_output_directories(self, set_name: str, pair_name: str) -> Dict[str, Path]:
        """
        Set up output directory structure for a specific image pair.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            
        Returns:
            Dict[str, Path]: Dictionary containing output and temp paths
        """
        # Create output directory structure
        output_set_folder = self.base_output_path / self.folder_name / set_name
        output_pair_folder = output_set_folder / pair_name
        
        PathManager.ensure_directory_exists(output_pair_folder)
        
        paths = {'output': output_pair_folder}
        
        # Create temp directory if base temp path is provided
        if self.base_temp_path:
            temp_set_folder = self.base_temp_path / self.folder_name / set_name
            temp_pair_folder = temp_set_folder / pair_name
            
            PathManager.ensure_directory_exists(temp_pair_folder, clear_if_exists=True)
            paths['temp'] = temp_pair_folder
        
        self.logger.info(f"Set up directories for {set_name}/{pair_name}")
        return paths
    
    def save_metadata(self, metadata: Dict[str, Any], output_paths: Dict[str, Path], 
                     pair_name: str, filename_prefix: str = "metadata") -> Dict[str, bool]:
        """
        Save metadata to JSON files in all specified paths.
        
        Args:
            metadata: Metadata dictionary to save
            output_paths: Dictionary of output paths
            pair_name: Name of the image pair
            filename_prefix: Prefix for the metadata filename
            
        Returns:
            Dict[str, bool]: Save results for each location
        """
        results = {}
        filename = f'{filename_prefix}_{pair_name}.json'
        
        for location_name, path in output_paths.items():
            success = MetadataSaver.save_json_metadata(metadata, path, filename)
            results[f'{location_name}_metadata'] = success
            
            if success:
                self.processing_stats['successful_operations'] += 1
                self.logger.debug(f"Saved metadata to {location_name}: {path / filename}")
            else:
                self.processing_stats['failed_operations'] += 1
                self.logger.error(f"Failed to save metadata to {location_name}: {path / filename}")
            
            self.processing_stats['total_operations'] += 1
        
        return results
    
    def create_processing_summary(self, pair_name: str, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive processing summary.
        
        Args:
            pair_name: Name of the processed pair
            processing_results: Results from processing operations
            
        Returns:
            Dict[str, Any]: Processing summary
        """
        # Count successful operations
        total_saves = 0
        successful_saves = 0
        
        for category_results in processing_results.values():
            if isinstance(category_results, dict):
                for success in category_results.values():
                    if isinstance(success, bool):
                        total_saves += 1
                        if success:
                            successful_saves += 1
        
        summary = {
            'pair_name': pair_name,
            'processing_timestamp': processing_results.get('timestamp'),
            'processing_info': processing_results.get('processing_info', {}),
            'file_operations': {
                'total_saves': total_saves,
                'successful_saves': successful_saves,
                'success_rate': successful_saves / total_saves if total_saves > 0 else 0
            },
            'processing_statistics': self.processing_stats.copy()
        }
        
        return summary
    
    def cleanup_temporary_files(self, temp_path: Path, keep_patterns: List[str] = None) -> bool:
        """
        Clean up temporary files, optionally keeping files matching certain patterns.
        
        Args:
            temp_path: Path to temporary directory
            keep_patterns: List of filename patterns to keep (optional)
            
        Returns:
            bool: True if cleanup was successful
        """
        try:
            if not temp_path.exists():
                return True
            
            keep_patterns = keep_patterns or []
            
            for file_path in temp_path.rglob('*'):
                if file_path.is_file():
                    # Check if file should be kept
                    should_keep = any(pattern in file_path.name for pattern in keep_patterns)
                    
                    if not should_keep:
                        file_path.unlink()
                        self.logger.debug(f"Deleted temporary file: {file_path}")
            
            self.logger.info(f"Cleaned up temporary files in: {temp_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup temporary files: {e}")
            return False
    
    def log_save_results(self, pair_name: str, results: Dict[str, Dict[str, bool]]) -> None:
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
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        stats = self.processing_stats.copy()
        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
        else:
            stats['success_rate'] = 0
        
        return stats
    
    def reset_processing_statistics(self) -> None:
        """Reset processing statistics counters."""
        self.processing_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0
        }
    
    @abstractmethod
    def get_folder_name(self) -> str:
        """
        Get the specific folder name for this file manager type.
        
        Returns:
            str: Folder name for this processing type
        """
        pass
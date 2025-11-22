"""
Base processor class for stereo vision processing modules.

This module provides a unified base class for processing operations across
rectification, disparity, and height processing modules.
"""

import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from utils.file_operations import PathManager
from utils.logger_config import get_logger


class BaseProcessor(ABC):
    """
    Base class for stereo vision processing operations.
    
    Provides common functionality for:
    - Configuration management
    - Path setup and validation
    - Processing pipeline coordination
    - Error handling and logging
    - Processing state management
    
    Subclasses should implement module-specific processing logic.
    """
    
    def __init__(self, config, processing_type: str):
        """
        Initialize base processor.
        
        Args:
            config: Configuration object with processing parameters
            processing_type: Type of processing (e.g., 'rectification', 'disparity', 'height')
        """
        self.config = config
        self.processing_type = processing_type
        self.root = Path(__file__).parent.parent.parent
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Processing state
        self.current_pair_info = {}
        self.processing_results = {}
        
        # Initialize paths
        self.input_folder = None
        self.output_folder = None
        self.temp_folder = None
        self._setup_base_paths()
        
        # Configuration flags
        self.need_show = getattr(self.config, 'need_showPicture', 'False') == "True"
        
        self.logger.info(f"{self.__class__.__name__} initialized for {processing_type} processing")
    
    def _setup_base_paths(self) -> None:
        """Initialize base folder paths from configuration."""
        self.output_folder = self.root / Path(self.config.save_path_result)
        self.temp_folder = self.root / Path(self.config.save_path_temp)
        
        self.logger.info(f"Base paths configured:")
        self.logger.info(f"  Output folder: {self.output_folder}")
        self.logger.info(f"  Temp folder: {self.temp_folder}")
    
    def process_all_sets(self) -> None:
        """
        Main entry point for processing all image sets.
        
        This method processes all 'set_*' folders in the input directory.
        """
        self.logger.info(f"Starting to process all image sets for {self.processing_type}")
        
        try:
            set_folders = self._validate_input_structure()
        except ValueError as e:
            self.logger.error(str(e))
            return
        
        for set_folder in set_folders:
            self._process_set(set_folder)
        
        self.logger.info(f"All image sets processed successfully for {self.processing_type}")
    
    def _validate_input_structure(self) -> list:
        """
        Validate input directory structure and return set folders.
        
        Returns:
            List[Path]: List of valid set folders
            
        Raises:
            ValueError: If input structure is invalid
        """
        if not self.input_folder:
            raise ValueError("Input folder not configured")
        
        return PathManager.validate_input_structure(self.input_folder)
    
    def _process_set(self, set_folder: Path) -> None:
        """
        Process a single image set.
        
        Args:
            set_folder: Path to the set folder
        """
        set_name = set_folder.name
        self.logger.info(f"Processing set: {set_name}")
        
        # Get all pair directories in the set
        pair_folders = [p for p in set_folder.iterdir() if p.is_dir()]
        
        if not pair_folders:
            self.logger.warning(f"No pair directories found in {set_name}")
            return
        
        for pair_folder in pair_folders:
            try:
                self._process_image_pair(set_name, pair_folder)
            except Exception as e:
                self.logger.error(f"Failed to process pair {pair_folder.name} in set {set_name}: {e}")
                continue
        
        self.logger.info(f"Set {set_name} processed successfully")
    
    def _process_image_pair(self, set_name: str, pair_folder: Path) -> None:
        """
        Process a single image pair through the complete processing pipeline.
        
        Args:
            set_name: Name of the image set
            pair_folder: Path to the pair folder
        """
        pair_name = pair_folder.name
        self.logger.info(f"Processing image pair: {set_name}/{pair_name}")
        
        # Setup processing context
        self._setup_processing_context(set_name, pair_name, pair_folder)
        
        try:
            # Execute the main processing pipeline
            processing_results = self._execute_processing_pipeline(pair_folder)
            
            # Save results
            self._save_processing_results(processing_results, pair_name)
            
            self.logger.info(f"Successfully processed {set_name}/{pair_name}")
            
        except Exception as e:
            self.logger.error(f"Error processing {set_name}/{pair_name}: {e}")
            raise
    
    def _setup_processing_context(self, set_name: str, pair_name: str, pair_folder: Path) -> None:
        """
        Setup processing context for current image pair.
        
        Args:
            set_name: Name of the image set
            pair_name: Name of the image pair
            pair_folder: Path to the pair folder
        """
        self.current_pair_info = {
            'set_name': set_name,
            'pair_name': pair_name,
            'pair_folder': str(pair_folder),
            'timestamp': datetime.datetime.now().isoformat(),
            'processing_type': self.processing_type
        }
    
    def _create_comprehensive_metadata(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive metadata for the processing.
        
        Args:
            processing_results: Results from processing operations
            
        Returns:
            Dict[str, Any]: Comprehensive metadata
        """
        metadata = {
            'pair_info': self.current_pair_info,
            'processing_version': f'{self.processing_type}_processor_refactored_v1.0',
            'configuration': self._extract_relevant_config(),
            'processing_results': processing_results,
            'processing_statistics': getattr(self, 'processing_statistics', {})
        }
        
        return metadata
    
    def _extract_relevant_config(self) -> Dict[str, Any]:
        """
        Extract relevant configuration parameters for metadata.
        
        Returns:
            Dict[str, Any]: Relevant configuration parameters
        """
        # Base configuration that's common across all processors
        base_config = {
            'need_show': self.need_show,
            'save_path_result': str(self.config.save_path_result),
            'save_path_temp': str(self.config.save_path_temp)
        }
        
        # Add processor-specific configuration
        specific_config = self._get_processor_specific_config()
        base_config.update(specific_config)
        
        return base_config
    
    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about current processing setup.
        
        Returns:
            Dict[str, Any]: Processing information
        """
        return {
            'processing_type': self.processing_type,
            'input_folder': str(self.input_folder) if self.input_folder else None,
            'output_folder': str(self.output_folder),
            'temp_folder': str(self.temp_folder),
            'current_pair_info': self.current_pair_info,
            'configuration': self._extract_relevant_config(),
            'processing_ready': self._is_processing_ready()
        }
    
    @abstractmethod
    def _setup_input_folder(self) -> None:
        """Setup input folder path specific to processor type."""
        pass
    
    @abstractmethod
    def _execute_processing_pipeline(self, pair_folder: Path) -> Dict[str, Any]:
        """
        Execute the main processing pipeline for a single pair.
        
        Args:
            pair_folder: Path to the pair folder
            
        Returns:
            Dict[str, Any]: Processing results
        """
        pass
    
    @abstractmethod
    def _save_processing_results(self, processing_results: Dict[str, Any], pair_name: str) -> None:
        """
        Save processing results using the appropriate file manager.
        
        Args:
            processing_results: Results from processing
            pair_name: Name of the image pair
        """
        pass
    
    @abstractmethod
    def _get_processor_specific_config(self) -> Dict[str, Any]:
        """
        Get processor-specific configuration parameters.
        
        Returns:
            Dict[str, Any]: Processor-specific configuration
        """
        pass
    
    @abstractmethod
    def _is_processing_ready(self) -> bool:
        """
        Check if processor is ready for processing.
        
        Returns:
            bool: True if ready for processing
        """
        pass
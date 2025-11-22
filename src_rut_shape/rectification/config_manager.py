"""
Configuration management for rectification processing.

This module handles all configuration-related operations for the rectification
module, including parameter extraction, validation, and caching.
"""

from typing import Dict, Any, Optional
import logging

from utils.logger_config import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Manages configuration parameters for rectification processing."""
    
    def __init__(self, config: Any):
        """
        Initialize configuration manager.
        
        Args:
            config: Configuration object with rectification parameters
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Cache for processed configurations
        self._rectification_config_cache: Optional[Dict[str, Any]] = None
        self._resize_config_cache: Optional[Dict[str, Any]] = None
        self._processor_config_cache: Optional[Dict[str, Any]] = None
    
    def get_rectification_config(self) -> Dict[str, Any]:
        """
        Extract rectification-specific configuration parameters.
        
        Returns:
            Dict[str, Any]: Rectification configuration
        """
        if self._rectification_config_cache is None:
            self._rectification_config_cache = {
                'rectified_image_size': getattr(self.config, 'rectified_image_size', 'default'),
                'rectification_alpha': getattr(self.config, 'rectification_alpha', 0)
            }
            
            self.logger.debug("Cached rectification configuration")
        
        return self._rectification_config_cache.copy()
    
    def get_resize_config(self) -> Dict[str, Any]:
        """
        Extract resize-specific configuration parameters.
        
        Returns:
            Dict[str, Any]: Resize configuration
        """
        if self._resize_config_cache is None:
            self._resize_config_cache = {
                'resize_target_width': getattr(self.config, 'resize_target_width', None),
                'resize_target_height': getattr(self.config, 'resize_target_height', None),
                'resize_scale': getattr(self.config, 'resize_scale', 1),
                'resize_max_pixels': getattr(self.config, 'resize_max_pixels', 3840 * 2160)
            }
            
            self.logger.debug("Cached resize configuration")
        
        return self._resize_config_cache.copy()
    
    def get_processor_specific_config(self) -> Dict[str, Any]:
        """
        Get comprehensive processor-specific configuration parameters.
        
        Returns:
            Dict[str, Any]: Combined processor configuration
        """
        if self._processor_config_cache is None:
            # Combine rectification and resize configurations
            rectification_config = self.get_rectification_config()
            resize_config = self.get_resize_config()
            
            self._processor_config_cache = {
                **rectification_config,
                **resize_config
            }
            
            self.logger.debug("Cached processor-specific configuration")
        
        return self._processor_config_cache.copy()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration parameters and return validation results.
        
        Returns:
            Dict[str, Any]: Validation results with warnings and errors
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validate rectification parameters
        rect_config = self.get_rectification_config()
        
        # Check rectified image size parameter
        valid_size_modes = ['default', 'crop', 'full']
        if rect_config['rectified_image_size'] not in valid_size_modes:
            validation_results['warnings'].append(
                f"Unknown rectified_image_size: {rect_config['rectified_image_size']}, "
                f"expected one of {valid_size_modes}"
            )
        
        # Check alpha parameter
        alpha = rect_config['rectification_alpha']
        if not (0 <= alpha <= 1):
            validation_results['warnings'].append(
                f"rectification_alpha should be between 0 and 1, got {alpha}"
            )
        
        # Validate resize parameters
        resize_config = self.get_resize_config()
        
        # Check resize scale
        resize_scale = resize_config['resize_scale']
        if resize_scale <= 0:
            validation_results['errors'].append(
                f"resize_scale must be positive, got {resize_scale}"
            )
            validation_results['valid'] = False
        
        # Check target dimensions
        target_width = resize_config['resize_target_width']
        target_height = resize_config['resize_target_height']
        
        if target_width is not None and target_width <= 0:
            validation_results['errors'].append(
                f"resize_target_width must be positive, got {target_width}"
            )
            validation_results['valid'] = False
            
        if target_height is not None and target_height <= 0:
            validation_results['errors'].append(
                f"resize_target_height must be positive, got {target_height}"
            )
            validation_results['valid'] = False
        
        # Check max pixels
        max_pixels = resize_config['resize_max_pixels']
        if max_pixels <= 0:
            validation_results['errors'].append(
                f"resize_max_pixels must be positive, got {max_pixels}"
            )
            validation_results['valid'] = False
        
        # Log validation results
        if validation_results['warnings']:
            self.logger.warning(f"Configuration warnings: {validation_results['warnings']}")
        
        if validation_results['errors']:
            self.logger.error(f"Configuration errors: {validation_results['errors']}")
        
        return validation_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all configuration parameters.
        
        Returns:
            Dict[str, Any]: Configuration summary
        """
        return {
            'rectification': self.get_rectification_config(),
            'resize': self.get_resize_config(),
            'validation': self.validate_configuration()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached configuration data."""
        self._rectification_config_cache = None
        self._resize_config_cache = None
        self._processor_config_cache = None
        
        self.logger.debug("Cleared configuration cache")
    
    def update_config(self, new_config: Any) -> None:
        """
        Update configuration and clear cache.
        
        Args:
            new_config: New configuration object
        """
        self.config = new_config
        self.clear_cache()
        
        self.logger.info("Updated configuration and cleared cache")
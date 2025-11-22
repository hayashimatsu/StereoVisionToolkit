"""
Unified logging configuration for the stereo vision toolkit.

This module provides a centralized logging configuration that can be inherited
by all other modules in the project, ensuring consistent logging behavior.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path


class LoggerConfig:
    """Centralized logger configuration manager."""
    
    _configured = False
    _root_logger_name = 'stereo_vision_toolkit'
    
    @classmethod
    def setup_root_logger(
        cls,
        level: int = logging.INFO,
        format_string: Optional[str] = None,
        log_file: Optional[Path] = None
    ) -> logging.Logger:
        """
        Setup the root logger for the entire application.
        
        Args:
            level: Logging level (default: INFO)
            format_string: Custom format string (optional)
            log_file: Optional file path for logging to file
            
        Returns:
            logging.Logger: Configured root logger
        """
        if cls._configured:
            return logging.getLogger(cls._root_logger_name)
        
        # Create root logger
        root_logger = logging.getLogger(cls._root_logger_name)
        root_logger.setLevel(level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Default format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate messages
        root_logger.propagate = False
        
        cls._configured = True
        
        root_logger.info(f"Root logger configured: level={logging.getLevelName(level)}")
        if log_file:
            root_logger.info(f"Logging to file: {log_file}")
            
        return root_logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger that inherits from the root logger configuration.
        
        Args:
            name: Logger name (typically __name__ from calling module)
            
        Returns:
            logging.Logger: Configured logger
        """
        # Ensure root logger is configured
        if not cls._configured:
            cls.setup_root_logger()
        
        # Create child logger under the root logger
        full_name = f"{cls._root_logger_name}.{name}"
        logger = logging.getLogger(full_name)
        
        # Child loggers inherit configuration from parent
        # No need to add handlers - they will propagate to root
        logger.propagate = True
        
        return logger
    
    @classmethod
    def set_level(cls, level: int) -> None:
        """
        Change the logging level for all loggers.
        
        Args:
            level: New logging level
        """
        root_logger = logging.getLogger(cls._root_logger_name)
        root_logger.setLevel(level)
        
        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(level)
            
        root_logger.info(f"Logging level changed to: {logging.getLevelName(level)}")
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if the root logger has been configured."""
        return cls._configured
    
    @classmethod
    def get_configuration_info(cls) -> Dict[str, Any]:
        """
        Get information about current logger configuration.
        
        Returns:
            Dict[str, Any]: Configuration information
        """
        if not cls._configured:
            return {'configured': False}
        
        root_logger = logging.getLogger(cls._root_logger_name)
        
        return {
            'configured': True,
            'root_logger_name': cls._root_logger_name,
            'level': logging.getLevelName(root_logger.level),
            'handlers': [
                {
                    'type': type(handler).__name__,
                    'level': logging.getLevelName(handler.level)
                }
                for handler in root_logger.handlers
            ]
        }


def get_logger(name: str = None) -> logging.Logger:
    """
    Convenience function to get a properly configured logger.
    
    Args:
        name: Logger name (if None, uses calling module's __name__)
        
    Returns:
        logging.Logger: Configured logger
    """
    if name is None:
        # Try to get the calling module's name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return LoggerConfig.get_logger(name)


# Initialize root logger when module is imported
# This ensures consistent configuration across all modules
def initialize_default_logger():
    """Initialize the default logger configuration."""
    if not LoggerConfig.is_configured():
        LoggerConfig.setup_root_logger()


# Auto-initialize when imported
initialize_default_logger()
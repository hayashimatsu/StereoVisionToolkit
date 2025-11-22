"""
Image processing utilities for stereo vision applications.

This module provides image processing functions including resizing strategies,
color space conversions, and image quality validations that can be used
across multiple computer vision modules.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

from utils.logger_config import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Handles common image processing operations for stereo vision."""
    
    @staticmethod
    def validate_image_pair(
        left_image: np.ndarray, 
        right_image: np.ndarray
    ) -> bool:
        """
        Validate that two images are compatible for stereo processing.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            
        Returns:
            bool: True if images are compatible
            
        Raises:
            ValueError: If images are incompatible
        """
        if left_image is None or right_image is None:
            raise ValueError("One or both images are None")
            
        if left_image.shape != right_image.shape:
            raise ValueError(f"Image shapes don't match: "
                           f"left={left_image.shape}, right={right_image.shape}")
            
        if len(left_image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image dimensions: {left_image.shape}")
            
        if left_image.dtype != right_image.dtype:
            logger.warning(f"Image dtypes differ: "
                          f"left={left_image.dtype}, right={right_image.dtype}")
            
        return True
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive information about an image.
        
        Args:
            image: Input image
            
        Returns:
            Dict[str, Any]: Image information
        """
        if image is None:
            return {'valid': False, 'error': 'Image is None'}
            
        info = {
            'valid': True,
            'shape': image.shape,
            'dtype': image.dtype,
            'size_bytes': image.nbytes,
            'size_mb': image.nbytes / (1024 * 1024),
        }
        
        if len(image.shape) == 2:
            info['channels'] = 1
            info['height'], info['width'] = image.shape
        elif len(image.shape) == 3:
            info['height'], info['width'], info['channels'] = image.shape
        else:
            info['valid'] = False
            info['error'] = f'Invalid image dimensions: {image.shape}'
            return info
            
        info['total_pixels'] = info['height'] * info['width']
        
        # Calculate basic statistics
        info['min_value'] = float(np.min(image))
        info['max_value'] = float(np.max(image))
        info['mean_value'] = float(np.mean(image))
        info['std_value'] = float(np.std(image))
        
        return info


class ResizeStrategy:
    """Handles different image resizing strategies for memory optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize resize strategy with configuration.
        
        Args:
            config: Configuration dictionary with resize parameters
        """
        self.config = config or {}
        # self.logger = logging.getLogger(__name__)
    
    def determine_resize_parameters(
        self, 
        original_size: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """
        Determine optimal resize parameters based on configuration and image size.
        
        Args:
            original_size: (width, height) of original image
            
        Returns:
            Tuple containing:
                - (new_width, new_height): Target dimensions
                - (scale_x, scale_y): Scaling factors
        """
        original_width, original_height = original_size
        
        # Get configuration parameters
        target_width = self.config.get('resize_target_width')
        target_height = self.config.get('resize_target_height')
        resize_scale = self.config.get('resize_scale')
        max_pixels = self.config.get('resize_max_pixels', 3840 * 2160)
        
        # Strategy 1: Resize to specific dimensions
        if target_width and target_height:
            new_width, new_height = target_width, target_height
            scale_x = new_width / original_width
            scale_y = new_height / original_height
            
            logger.info(f"Using target dimensions strategy: {new_width}x{new_height}")
            
        # Strategy 2: Resize by scale factor
        elif resize_scale:
            scale_x = scale_y = resize_scale
            new_width = int(original_width * resize_scale)
            new_height = int(original_height * resize_scale)
            
            logger.info(f"Using scale factor strategy: {resize_scale} -> {new_width}x{new_height}")
            
        # Strategy 3: Auto-resize based on memory constraints
        else:
            current_pixels = original_width * original_height
            
            if current_pixels > max_pixels:
                scale_factor_auto = (max_pixels / current_pixels) ** 0.5
                scale_factor_auto = max(0.5, scale_factor_auto)  # Minimum 50% scale
                
                scale_x = scale_y = scale_factor_auto
                new_width = int(original_width * scale_factor_auto)
                new_height = int(original_height * scale_factor_auto)
                
                logger.info(f"Using auto-resize strategy: {scale_factor_auto:.3f} -> {new_width}x{new_height}")
            else:
                # No resize needed
                scale_x = scale_y = 1.0
                new_width, new_height = original_width, original_height
                
                logger.info("No resize needed - image within memory limits")
        
        return (new_width, new_height), (scale_x, scale_y)
    
    def apply_resize(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Apply resize to image with specified parameters.
        
        Args:
            image: Input image
            target_size: (width, height) target dimensions
            interpolation: OpenCV interpolation method
            
        Returns:
            np.ndarray: Resized image
        """
        if image is None:
            raise ValueError("Input image is None")
            
        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size
        
        if (original_width, original_height) == (target_width, target_height):
            return image.copy()
            
        resized_image = cv2.resize(image, target_size, interpolation=interpolation)
        
        logger.info(f"Resized image from {original_width}x{original_height} "
                        f"to {target_width}x{target_height}")
        
        return resized_image
    
    def apply_resize_strategy(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """
        Apply resize strategy to stereo image pair.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            
        Returns:
            Tuple containing:
                - Resized left image
                - Resized right image  
                - (scale_x, scale_y) scaling factors applied
        """
        # Validate input images
        ImageProcessor.validate_image_pair(left_image, right_image)
        
        original_height, original_width = left_image.shape[:2]
        original_size = (original_width, original_height)
        
        # Determine resize parameters
        target_size, scale_factor = self.determine_resize_parameters(original_size)
        
        # Apply resize if needed
        if scale_factor[0] != 1.0 or scale_factor[1] != 1.0:
            resized_left = self.apply_resize(left_image, target_size)
            resized_right = self.apply_resize(right_image, target_size)
        else:
            resized_left = left_image.copy()
            resized_right = right_image.copy()
        
        return resized_left, resized_right, scale_factor


class ColorSpaceConverter:
    """Handles color space conversions for image processing."""
    
    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got shape {image.shape}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to BGR."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got shape {image.shape}")
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def create_weighted_overlay(
        image1: np.ndarray, 
        image2: np.ndarray,
        weight1: float = 0.75, 
        weight2: float = 0.5,
        gamma: float = 0.0
    ) -> np.ndarray:
        """
        Create weighted overlay of two images.
        
        Args:
            image1: First image
            image2: Second image
            weight1: Weight for first image
            weight2: Weight for second image
            gamma: Gamma correction value
            
        Returns:
            np.ndarray: Blended image
        """
        ImageProcessor.validate_image_pair(image1, image2)
        
        # Convert to RGB if needed
        if len(image1.shape) == 3:
            img1_rgb = ColorSpaceConverter.bgr_to_rgb(image1)
            img2_rgb = ColorSpaceConverter.bgr_to_rgb(image2)
        else:
            img1_rgb = image1
            img2_rgb = image2
            
        # Create weighted blend
        blended = cv2.addWeighted(img1_rgb, weight1, img2_rgb, weight2, gamma)
        
        # Convert back to BGR if original was color
        if len(image1.shape) == 3:
            blended = ColorSpaceConverter.rgb_to_bgr(blended)
            
        return blended


class ImageQualityValidator:
    """Validates image quality for stereo vision processing."""
    
    @staticmethod
    def check_image_brightness(
        image: np.ndarray, 
        min_brightness: float = 10.0,
        max_brightness: float = 245.0
    ) -> Dict[str, Any]:
        """
        Check if image brightness is within acceptable range.
        
        Args:
            image: Input image
            min_brightness: Minimum acceptable mean brightness
            max_brightness: Maximum acceptable mean brightness
            
        Returns:
            Dict[str, Any]: Brightness analysis results
        """
        if len(image.shape) == 3:
            # Convert to grayscale for brightness analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        results = {
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'min_pixel': float(np.min(gray)),
            'max_pixel': float(np.max(gray)),
            'is_too_dark': mean_brightness < min_brightness,
            'is_too_bright': mean_brightness > max_brightness,
            'has_good_contrast': std_brightness > 20.0,
            'warnings': []
        }
        
        if results['is_too_dark']:
            results['warnings'].append(f"Image too dark: mean={mean_brightness:.1f}")
            
        if results['is_too_bright']:
            results['warnings'].append(f"Image too bright: mean={mean_brightness:.1f}")
            
        if not results['has_good_contrast']:
            results['warnings'].append(f"Low contrast: std={std_brightness:.1f}")
            
        return results
    
    @staticmethod
    def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Dict[str, Any]:
        """
        Detect if image is blurry using Laplacian variance.
        
        Args:
            image: Input image
            threshold: Blur detection threshold
            
        Returns:
            Dict[str, Any]: Blur analysis results
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        results = {
            'laplacian_variance': float(laplacian_var),
            'is_blurry': laplacian_var < threshold,
            'blur_threshold': threshold
        }
        
        if results['is_blurry']:
            results['warning'] = f"Image appears blurry: variance={laplacian_var:.1f}"
            
        return results


def resize_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    
    Args:
        image: Input image
        scale_factor: Scaling factor
        
    Returns:
        np.ndarray: Resized image
    """
    if scale_factor == 1.0:
        return image.copy()
        
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
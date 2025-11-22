"""
SGBM (Semi-Global Block Matching) engine for stereo disparity calculation.

This module provides the core SGBM algorithm implementation with optimized
parameter configuration and validation for high-quality disparity computation.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class SGBMEngine:
    """Core SGBM algorithm engine for stereo disparity calculation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SGBM engine with configuration.
        
        Args:
            config: Configuration dictionary with SGBM parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # SGBM parameters
        self.sgbm_mode = None
        self.min_disparity = None
        self.num_disparities = None
        self.block_size = None
        self.p1 = None
        self.p2 = None
        self.disp12_max_diff = None
        self.uniqueness_ratio = None
        self.speckle_window_size = None
        self.speckle_range = None
        
        # Stereo matcher instance
        self._stereo_matcher = None
    
    def configure_parameters(
        self,
        min_disparity: int,
        num_disparities: int,
        block_size: int,
        use_fast_mode: bool = True,
        image_channels: int = 3
    ) -> None:
        """
        Configure SGBM parameters for disparity calculation.
        
        Args:
            min_disparity: Minimum disparity value
            num_disparities: Number of disparities (must be divisible by 16)
            block_size: Block size for matching (must be odd)
            use_fast_mode: Whether to use SGBM_3WAY mode for speed
            image_channels: Number of image channels
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        self._validate_sgbm_parameters(min_disparity, num_disparities, block_size)
        
        # Set basic parameters
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size
        
        # Set SGBM mode
        self.sgbm_mode = (cv2.STEREO_SGBM_MODE_SGBM_3WAY if use_fast_mode 
                         else cv2.STEREO_SGBM_MODE_SGBM)
        
        # Calculate P1 and P2 parameters
        # P1: penalty for disparity changes of ±1
        # P2: penalty for larger disparity changes
        self.p1 = 8 * image_channels * block_size * block_size
        self.p2 = 32 * image_channels * block_size * block_size
        
        # Set other parameters with proven defaults
        self.disp12_max_diff = -1  # Disable left-right consistency check for speed
        self.uniqueness_ratio = 1  # Minimum margin in percentage
        self.speckle_window_size = 100  # Maximum speckle size
        self.speckle_range = 32  # Maximum disparity variation in speckle
        
        self.logger.info(f"SGBM parameters configured: "
                        f"minDisp={min_disparity}, numDisp={num_disparities}, "
                        f"blockSize={block_size}, mode={'3WAY' if use_fast_mode else 'SGBM'}")
    
    def _validate_sgbm_parameters(
        self, 
        min_disparity: int, 
        num_disparities: int, 
        block_size: int
    ) -> None:
        """
        Validate SGBM parameters for correctness.
        
        Args:
            min_disparity: Minimum disparity value
            num_disparities: Number of disparities
            block_size: Block size for matching
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_disparities <= 0 or num_disparities % 16 != 0:
            raise ValueError(f"num_disparities must be positive and divisible by 16, got {num_disparities}")
        
        if block_size <= 0 or block_size % 2 == 0:
            raise ValueError(f"block_size must be positive and odd, got {block_size}")
        
        if min_disparity < 0:
            raise ValueError(f"min_disparity must be non-negative, got {min_disparity}")
        
        if block_size > 21:
            self.logger.warning(f"Large block_size ({block_size}) may reduce accuracy")
    
    def create_stereo_matcher(self) -> cv2.StereoSGBM:
        """
        Create and configure OpenCV StereoSGBM matcher.
        
        Returns:
            cv2.StereoSGBM: Configured stereo matcher
            
        Raises:
            RuntimeError: If parameters are not configured
        """
        if self.min_disparity is None or self.num_disparities is None:
            raise RuntimeError("SGBM parameters not configured. Call configure_parameters() first.")
        
        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            disp12MaxDiff=self.disp12_max_diff,
            mode=self.sgbm_mode
        )
        
        # Set all parameters
        stereo.setMinDisparity(self.min_disparity)
        stereo.setNumDisparities(self.num_disparities)
        stereo.setBlockSize(self.block_size)
        stereo.setP1(self.p1)
        stereo.setP2(self.p2)
        stereo.setSpeckleRange(self.speckle_range)
        stereo.setSpeckleWindowSize(self.speckle_window_size)
        stereo.setUniquenessRatio(self.uniqueness_ratio)
        
        self._stereo_matcher = stereo
        
        self.logger.info("StereoSGBM matcher created and configured")
        return stereo
    
    def compute_disparity(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray,
        convert_to_grayscale: bool = True
    ) -> np.ndarray:
        """
        Compute disparity map from stereo image pair.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            convert_to_grayscale: Whether to convert images to grayscale
            
        Returns:
            np.ndarray: Disparity map (16-bit fixed point format)
            
        Raises:
            ValueError: If images are incompatible
            RuntimeError: If stereo matcher is not configured
        """
        # Validate input images
        self._validate_stereo_images(left_image, right_image)
        
        # Create stereo matcher if not exists
        if self._stereo_matcher is None:
            self.create_stereo_matcher()
        
        # Convert to grayscale if needed
        if convert_to_grayscale and len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        # Compute disparity
        self.logger.info("Computing disparity using SGBM algorithm")
        disparity = self._stereo_matcher.compute(left_gray, right_gray)
        
        # Log disparity statistics
        valid_disparity = disparity[disparity > 0]
        if len(valid_disparity) > 0:
            self.logger.info(f"Disparity computed: "
                           f"valid_pixels={len(valid_disparity)}/{disparity.size} "
                           f"({100*len(valid_disparity)/disparity.size:.1f}%), "
                           f"range=[{valid_disparity.min()//16:.1f}, {valid_disparity.max()//16:.1f}]")
        else:
            self.logger.warning("No valid disparity values computed")
        
        return disparity
    
    def _validate_stereo_images(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray
    ) -> None:
        """
        Validate stereo image pair for compatibility.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            
        Raises:
            ValueError: If images are incompatible
        """
        if left_image is None or right_image is None:
            raise ValueError("Input images cannot be None")
        
        if left_image.shape != right_image.shape:
            raise ValueError(f"Image shapes don't match: "
                           f"left={left_image.shape}, right={right_image.shape}")
        
        if len(left_image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image dimensions: {left_image.shape}")
        
        if left_image.dtype != right_image.dtype:
            self.logger.warning(f"Image dtypes differ: "
                              f"left={left_image.dtype}, right={right_image.dtype}")
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Get current SGBM configuration information.
        
        Returns:
            Dict[str, Any]: Configuration information
        """
        return {
            'configured': self.min_disparity is not None,
            'parameters': {
                'min_disparity': self.min_disparity,
                'num_disparities': self.num_disparities,
                'block_size': self.block_size,
                'sgbm_mode': 'SGBM_3WAY' if self.sgbm_mode == cv2.STEREO_SGBM_MODE_SGBM_3WAY else 'SGBM',
                'p1': self.p1,
                'p2': self.p2,
                'uniqueness_ratio': self.uniqueness_ratio,
                'speckle_window_size': self.speckle_window_size,
                'speckle_range': self.speckle_range
            }
        }
    
    def optimize_parameters_for_image_size(
        self, 
        image_size: Tuple[int, int],
        baseline: float,
        focal_length: float,
        target_depth: float
    ) -> Dict[str, int]:
        """
        Suggest optimized SGBM parameters based on image characteristics.
        
        Args:
            image_size: (width, height) of input images
            baseline: Stereo baseline in meters
            focal_length: Average focal length in pixels
            target_depth: Target depth in meters
            
        Returns:
            Dict[str, int]: Suggested parameters
        """
        width, height = image_size
        
        # Calculate suggested num_disparities based on depth range
        max_disparity = int(focal_length * baseline / target_depth)
        suggested_num_disparities = ((max_disparity // 16) + 1) * 16
        
        # Suggest block_size based on image resolution
        if width * height > 2000000:  # High resolution
            suggested_block_size = 7
        elif width * height > 1000000:  # Medium resolution
            suggested_block_size = 5
        else:  # Low resolution
            suggested_block_size = 3
        
        suggestions = {
            'num_disparities': suggested_num_disparities,
            'block_size': suggested_block_size,
            'min_disparity': 0
        }
        
        self.logger.info(f"Parameter suggestions for {width}x{height}: {suggestions}")
        return suggestions
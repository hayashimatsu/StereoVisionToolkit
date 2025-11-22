# """
# Stereo matching utilities for computer vision applications.

# This module provides tools for stereo matching analysis, including SAD calculation,
# point matching, and matching quality assessment that can be used across
# multiple stereo vision modules.
# """

# import cv2
# import numpy as np
# import pandas as pd
# import logging
# from typing import Tuple, Optional, Dict, Any, List

# logger = logging.getLogger(__name__)


# class StereoMatcher:
#     """General-purpose stereo matching utilities."""
    
#     @staticmethod
#     def calculate_sad(
#         left_block: np.ndarray, 
#         right_block: np.ndarray
#     ) -> float:
#         """
#         Calculate Sum of Absolute Differences (SAD) between two image blocks.
        
#         Args:
#             left_block: Left image block
#             right_block: Right image block
            
#         Returns:
#             float: SAD value
            
#         Raises:
#             ValueError: If blocks have different shapes
#         """
#         if left_block.shape != right_block.shape:
#             raise ValueError(f"Block shapes don't match: {left_block.shape} vs {right_block.shape}")
        
#         # Calculate absolute differences and sum
#         abs_diff = np.abs(left_block.astype(np.float32) - right_block.astype(np.float32))
#         sad = np.sum(abs_diff)
        
#         return float(sad)
    
#     @staticmethod
#     def calculate_ssd(
#         left_block: np.ndarray, 
#         right_block: np.ndarray
#     ) -> float:
#         """
#         Calculate Sum of Squared Differences (SSD) between two image blocks.
        
#         Args:
#             left_block: Left image block
#             right_block: Right image block
            
#         Returns:
#             float: SSD value
#         """
#         if left_block.shape != right_block.shape:
#             raise ValueError(f"Block shapes don't match: {left_block.shape} vs {right_block.shape}")
        
#         # Calculate squared differences and sum
#         diff = left_block.astype(np.float32) - right_block.astype(np.float32)
#         ssd = np.sum(diff * diff)
        
#         return float(ssd)
    
#     @staticmethod
#     def calculate_ncc(
#         left_block: np.ndarray, 
#         right_block: np.ndarray
#     ) -> float:
#         """
#         Calculate Normalized Cross Correlation (NCC) between two image blocks.
        
#         Args:
#             left_block: Left image block
#             right_block: Right image block
            
#         Returns:
#             float: NCC value (higher is better, range approximately [-1, 1])
#         """
#         if left_block.shape != right_block.shape:
#             raise ValueError(f"Block shapes don't match: {left_block.shape} vs {right_block.shape}")
        
#         # Convert to float and flatten
#         left_flat = left_block.astype(np.float32).flatten()
#         right_flat = right_block.astype(np.float32).flatten()
        
#         # Calculate means
#         left_mean = np.mean(left_flat)
#         right_mean = np.mean(right_flat)
        
#         # Calculate NCC
#         numerator = np.sum((left_flat - left_mean) * (right_flat - right_mean))
#         left_std = np.sqrt(np.sum((left_flat - left_mean) ** 2))
#         right_std = np.sqrt(np.sum((right_flat - right_mean) ** 2))
        
#         if left_std == 0 or right_std == 0:
#             return 0.0
        
#         ncc = numerator / (left_std * right_std)
#         return float(ncc)
    
#     @staticmethod
#     def extract_block(
#         image: np.ndarray, 
#         center_x: int, 
#         center_y: int, 
#         block_size: int
#     ) -> Optional[np.ndarray]:
#         """
#         Extract a square block from an image centered at given coordinates.
        
#         Args:
#             image: Input image
#             center_x: X coordinate of block center
#             center_y: Y coordinate of block center
#             block_size: Size of the block (must be odd)
            
#         Returns:
#             np.ndarray or None: Extracted block, or None if out of bounds
#         """
#         if block_size % 2 == 0:
#             raise ValueError("Block size must be odd")
        
#         half_size = block_size // 2
        
#         # Check bounds
#         if (center_x - half_size < 0 or center_x + half_size >= image.shape[1] or
#             center_y - half_size < 0 or center_y + half_size >= image.shape[0]):
#             return None
        
#         # Extract block
#         block = image[center_y - half_size:center_y + half_size + 1,
#                      center_x - half_size:center_x + half_size + 1]
        
#         return block


# class MatchPointAnalyzer:
#     """Analyzes stereo matching for specific points."""
    
#     def __init__(
#         self, 
#         left_image: np.ndarray, 
#         right_image: np.ndarray,
#         min_disparity: int = 0,
#         num_disparities: int = 64,
#         block_size: int = 5
#     ):
#         """
#         Initialize match point analyzer.
        
#         Args:
#             left_image: Left stereo image
#             right_image: Right stereo image
#             min_disparity: Minimum disparity to search
#             num_disparities: Number of disparities to search
#             block_size: Block size for matching (must be odd)
#         """
#         self.left_image = left_image
#         self.right_image = right_image
#         self.min_disparity = min_disparity
#         self.num_disparities = num_disparities
#         self.block_size = block_size
        
#         # Convert to grayscale if needed
#         if len(left_image.shape) == 3:
#             self.left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
#             self.right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
#         else:
#             self.left_gray = left_image
#             self.right_gray = right_image
        
#         self.logger = logging.getLogger(__name__)
    
#     def find_best_match(
#         self, 
#         left_x: int, 
#         left_y: int,
#         matching_method: str = 'sad'
#     ) -> Tuple[Optional[Tuple[int, int]], Optional[pd.DataFrame]]:
#         """
#         Find the best matching point in the right image for a given left image point.
        
#         Args:
#             left_x: X coordinate in left image
#             left_y: Y coordinate in left image
#             matching_method: Matching method ('sad', 'ssd', 'ncc')
            
#         Returns:
#             Tuple containing:
#                 - Best match coordinates (x, y) or None if no match found
#                 - DataFrame with search history or None if search failed
#         """
#         # Extract left block
#         left_block = StereoMatcher.extract_block(
#             self.left_gray, left_x, left_y, self.block_size
#         )
        
#         if left_block is None:
#             self.logger.warning(f"Cannot extract block at ({left_x}, {left_y})")
#             return None, None
        
#         # Search range in right image
#         search_start = max(left_x - self.min_disparity - self.num_disparities, 
#                           self.block_size // 2)
#         search_end = max(left_x - self.min_disparity, self.block_size // 2)
        
#         if search_start >= search_end:
#             self.logger.warning(f"Invalid search range: [{search_start}, {search_end}]")
#             return None, None
        
#         # Search for best match
#         search_history = []
#         best_match = None
#         best_score = float('inf') if matching_method in ['sad', 'ssd'] else float('-inf')
        
#         for right_x in range(search_start, search_end + 1):
#             # Extract right block
#             right_block = StereoMatcher.extract_block(
#                 self.right_gray, right_x, left_y, self.block_size
#             )
            
#             if right_block is None:
#                 continue
            
#             # Calculate matching score
#             if matching_method == 'sad':
#                 score = StereoMatcher.calculate_sad(left_block, right_block)
#                 is_better = score < best_score
#             elif matching_method == 'ssd':
#                 score = StereoMatcher.calculate_ssd(left_block, right_block)
#                 is_better = score < best_score
#             elif matching_method == 'ncc':
#                 score = StereoMatcher.calculate_ncc(left_block, right_block)
#                 is_better = score > best_score
#             else:
#                 raise ValueError(f"Unknown matching method: {matching_method}")
            
#             # Record search history
#             search_history.append({
#                 'X': right_x,
#                 'Y': left_y,
#                 'Score': score,
#                 'Method': matching_method.upper()
#             })
            
#             # Update best match
#             if is_better:
#                 best_score = score
#                 best_match = (right_x, left_y)
        
#         # Create search history DataFrame
#         history_df = pd.DataFrame(search_history) if search_history else None
        
#         if best_match:
#             self.logger.info(f"Best match for ({left_x}, {left_y}): {best_match} "
#                            f"with {matching_method.upper()}={best_score:.2f}")
#         else:
#             self.logger.warning(f"No match found for ({left_x}, {left_y})")
        
#         return best_match, history_df
    
#     def analyze_matching_quality(
#         self, 
#         left_x: int, 
#         left_y: int,
#         right_x: int, 
#         right_y: int
#     ) -> Dict[str, Any]:
#         """
#         Analyze the quality of a specific match.
        
#         Args:
#             left_x: X coordinate in left image
#             left_y: Y coordinate in left image
#             right_x: X coordinate in right image
#             right_y: Y coordinate in right image
            
#         Returns:
#             Dict[str, Any]: Quality analysis results
#         """
#         # Extract blocks
#         left_block = StereoMatcher.extract_block(
#             self.left_gray, left_x, left_y, self.block_size
#         )
#         right_block = StereoMatcher.extract_block(
#             self.right_gray, right_x, right_y, self.block_size
#         )
        
#         if left_block is None or right_block is None:
#             return {'valid': False, 'error': 'Cannot extract blocks'}
        
#         # Calculate multiple matching scores
#         sad_score = StereoMatcher.calculate_sad(left_block, right_block)
#         ssd_score = StereoMatcher.calculate_ssd(left_block, right_block)
#         ncc_score = StereoMatcher.calculate_ncc(left_block, right_block)
        
#         # Calculate disparity
#         disparity = left_x - right_x
        
#         # Analyze block statistics
#         left_stats = {
#             'mean': float(np.mean(left_block)),
#             'std': float(np.std(left_block)),
#             'min': float(np.min(left_block)),
#             'max': float(np.max(left_block))
#         }
        
#         right_stats = {
#             'mean': float(np.mean(right_block)),
#             'std': float(np.std(right_block)),
#             'min': float(np.min(right_block)),
#             'max': float(np.max(right_block))
#         }
        
#         # Quality assessment
#         quality_score = ncc_score  # Use NCC as primary quality indicator
#         if quality_score > 0.8:
#             quality_level = 'excellent'
#         elif quality_score > 0.6:
#             quality_level = 'good'
#         elif quality_score > 0.4:
#             quality_level = 'fair'
#         else:
#             quality_level = 'poor'
        
#         return {
#             'valid': True,
#             'coordinates': {
#                 'left': (left_x, left_y),
#                 'right': (right_x, right_y)
#             },
#             'disparity': disparity,
#             'matching_scores': {
#                 'sad': sad_score,
#                 'ssd': ssd_score,
#                 'ncc': ncc_score
#             },
#             'block_statistics': {
#                 'left': left_stats,
#                 'right': right_stats
#             },
#             'quality_assessment': {
#                 'score': quality_score,
#                 'level': quality_level
#             }
#         }
    
#     def get_pixel_values(
#         self, 
#         left_x: int, 
#         left_y: int,
#         right_x: int, 
#         right_y: int
#     ) -> Dict[str, Any]:
#         """
#         Get pixel values at specified coordinates.
        
#         Args:
#             left_x: X coordinate in left image
#             left_y: Y coordinate in left image
#             right_x: X coordinate in right image
#             right_y: Y coordinate in right image
            
#         Returns:
#             Dict[str, Any]: Pixel value information
#         """
#         pixel_info = {}
        
#         # Get left pixel values
#         if (0 <= left_x < self.left_image.shape[1] and 
#             0 <= left_y < self.left_image.shape[0]):
#             if len(self.left_image.shape) == 3:
#                 pixel_info['left_rgb'] = self.left_image[left_y, left_x].tolist()
#             pixel_info['left_gray'] = int(self.left_gray[left_y, left_x])
        
#         # Get right pixel values
#         if (0 <= right_x < self.right_image.shape[1] and 
#             0 <= right_y < self.right_image.shape[0]):
#             if len(self.right_image.shape) == 3:
#                 pixel_info['right_rgb'] = self.right_image[right_y, right_x].tolist()
#             pixel_info['right_gray'] = int(self.right_gray[right_y, right_x])
        
#         return pixel_info
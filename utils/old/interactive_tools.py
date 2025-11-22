# """
# Interactive tools for stereo vision debugging and analysis.

# This module provides interactive tools for point checking, matching analysis,
# and debugging that can be used across multiple stereo vision modules for
# post-processing analysis and validation.
# """

# import cv2
# import numpy as np
# import pandas as pd
# import tkinter as tk
# import tkinter.messagebox as tkMessageBox
# import os
# import logging
# from typing import Tuple, Optional, Dict, Any
# from pathlib import Path

# from utils.stereo_matching import MatchPointAnalyzer
# from utils.image import ImageChartGenerator

# logger = logging.getLogger(__name__)


# class InteractivePointChecker:
#     """
#     Interactive tool for checking stereo matching at specific points.
    
#     This tool allows users to click on points in the left image and see
#     the matching process and results in the right image.
#     """
    
#     def __init__(
#         self,
#         left_image: np.ndarray,
#         right_image: np.ndarray,
#         min_disparity: int = 0,
#         num_disparities: int = 64,
#         block_size: int = 5,
#         output_folder: str = "matching_analysis"
#     ):
#         """
#         Initialize interactive point checker.
        
#         Args:
#             left_image: Left stereo image (color)
#             right_image: Right stereo image (color)
#             min_disparity: Minimum disparity for matching
#             num_disparities: Number of disparities to search
#             block_size: Block size for matching
#             output_folder: Output folder for saving results
#         """
#         self.left_image = left_image
#         self.right_image = right_image
#         self.min_disparity = min_disparity
#         self.num_disparities = num_disparities
#         self.block_size = block_size
#         self.output_folder = output_folder
        
#         # Convert to grayscale for display and matching
#         if len(left_image.shape) == 3:
#             self.left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
#             self.right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
#         else:
#             self.left_gray = left_image
#             self.right_gray = right_image
        
#         # Initialize match analyzer
#         self.match_analyzer = MatchPointAnalyzer(
#             left_image, right_image, min_disparity, num_disparities, block_size
#         )
        
#         # UI state
#         self.current_x = None
#         self.current_y = None
#         self.canvas_name = "Interactive Point Checker"
        
#         self.logger = logging.getLogger(__name__)
    
#     def start_interactive_session(self) -> None:
#         """
#         Start interactive point checking session.
        
#         Users can click on points in the left image to analyze matching.
#         Press ESC to exit.
#         """
#         self.logger.info("Starting interactive point checking session")
#         self.logger.info("Click on points in the left image to analyze matching")
#         self.logger.info("Press ESC to exit")
        
#         while True:
#             self._setup_window()
#             cv2.setMouseCallback(self.canvas_name, self._on_mouse_click)
            
#             key = cv2.waitKey(0)
#             if key == 27:  # ESC key
#                 cv2.destroyAllWindows()
#                 self.logger.info("Interactive session ended")
#                 break
    
#     def _setup_window(self) -> None:
#         """Setup OpenCV window for interactive display."""
#         img_height, img_width = self.left_gray.shape[:2]
        
#         # Scale window for reasonable display size
#         scale_factor = min(1.0, 680 / img_height)
#         window_width = int(img_width * scale_factor)
#         window_height = int(img_height * scale_factor)
        
#         cv2.namedWindow(self.canvas_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(self.canvas_name, window_width, window_height)
#         cv2.imshow(self.canvas_name, self.left_gray)
    
#     def _on_mouse_click(self, event, x, y, flags, param) -> None:
#         """
#         Handle mouse click events.
        
#         Args:
#             event: OpenCV mouse event
#             x: X coordinate of click
#             y: Y coordinate of click
#             flags: Event flags
#             param: Additional parameters
#         """
#         if event == cv2.EVENT_LBUTTONDOWN:
#             self.current_x, self.current_y = x, y
            
#             print(f'\\nClicked at pixel coordinates: x = {x}, y = {y}')
            
#             # Ask user if they want to analyze this point
#             if tkMessageBox.askyesno('Point Analysis', 
#                                    f'Analyze matching for point ({x}, {y})?'):
#                 self._analyze_point_matching(x, y)
    
#     def _analyze_point_matching(self, x: int, y: int) -> None:
#         """
#         Analyze stereo matching for the selected point.
        
#         Args:
#             x: X coordinate in left image
#             y: Y coordinate in left image
#         """
#         try:
#             self.logger.info(f"Analyzing matching for point ({x}, {y})")
            
#             # Find best match
#             best_match, search_history = self.match_analyzer.find_best_match(x, y, 'sad')
            
#             if best_match is None or search_history is None:
#                 self.logger.warning(f"No match found for point ({x}, {y})")
#                 return
            
#             # Get pixel values
#             pixel_info = self.match_analyzer.get_pixel_values(x, y, best_match[0], best_match[1])
            
#             # Analyze match quality
#             quality_analysis = self.match_analyzer.analyze_matching_quality(
#                 x, y, best_match[0], best_match[1]
#             )
            
#             # Save analysis results
#             self._save_analysis_results(x, y, best_match, search_history, pixel_info, quality_analysis)
            
#             self.logger.info(f"Analysis complete for point ({x}, {y}) -> {best_match}")
            
#         except Exception as e:
#             self.logger.error(f"Failed to analyze point ({x}, {y}): {e}")
    
#     def _save_analysis_results(
#         self,
#         left_x: int,
#         left_y: int,
#         best_match: Tuple[int, int],
#         search_history: pd.DataFrame,
#         pixel_info: Dict[str, Any],
#         quality_analysis: Dict[str, Any]
#     ) -> None:
#         """
#         Save comprehensive analysis results for a point.
        
#         Args:
#             left_x: X coordinate in left image
#             left_y: Y coordinate in left image
#             best_match: Best match coordinates (x, y)
#             search_history: DataFrame with search history
#             pixel_info: Pixel value information
#             quality_analysis: Quality analysis results
#         """
#         # Create output directory for this point
#         point_folder = Path(self.output_folder) / f"left_rectify_pt({left_x},{left_y})"
#         point_folder.mkdir(parents=True, exist_ok=True)
        
#         # 1. Save search history
#         search_history.to_csv(
#             point_folder / f'searching_history(pt_{best_match}).csv', 
#             index=False
#         )
        
#         # 2. Save marked left image
#         painter_left = ImageChartGenerator(
#             img=self.left_gray,
#             xlabel="pixel",
#             ylabel="pixel",
#             save_path_result=str(point_folder)
#         )
#         painter_left.create_matching_point(
#             photo_name="rectified_gray_left",
#             matchPt=(left_x, left_y)
#         )
        
#         # 3. Save marked right image
#         painter_right = ImageChartGenerator(
#             img=self.right_gray,
#             xlabel="pixel",
#             ylabel="pixel",
#             save_path_result=str(point_folder)
#         )
#         painter_right.create_matching_point(
#             photo_name="rectified_gray_right",
#             matchPt=best_match
#         )
        
#         # 4. Save pixel information
#         pixel_text = []
#         if 'left_rgb' in pixel_info:
#             pixel_text.append(f"RGB of left: {pixel_info['left_rgb']}")
#         if 'right_rgb' in pixel_info:
#             pixel_text.append(f"RGB of right: {pixel_info['right_rgb']}")
#         if 'left_gray' in pixel_info:
#             pixel_text.append(f"Gray of left: {pixel_info['left_gray']}")
#         if 'right_gray' in pixel_info:
#             pixel_text.append(f"Gray of right: {pixel_info['right_gray']}")
        
#         with open(point_folder / 'pixel_info.txt', 'w') as f:
#             f.write('\\n'.join(pixel_text))
        
#         # 5. Save quality analysis
#         with open(point_folder / 'quality_analysis.json', 'w') as f:
#             import json
#             json.dump(quality_analysis, f, indent=2)
        
#         # 6. Save summary report
#         summary_text = [
#             f"Point Analysis Summary",
#             f"=" * 30,
#             f"Left point: ({left_x}, {left_y})",
#             f"Best match: {best_match}",
#             f"Disparity: {quality_analysis.get('disparity', 'N/A')}",
#             f"Quality level: {quality_analysis.get('quality_assessment', {}).get('level', 'N/A')}",
#             f"NCC score: {quality_analysis.get('matching_scores', {}).get('ncc', 'N/A'):.3f}",
#             f"SAD score: {quality_analysis.get('matching_scores', {}).get('sad', 'N/A'):.1f}",
#             f"",
#             f"Search parameters:",
#             f"- Min disparity: {self.min_disparity}",
#             f"- Num disparities: {self.num_disparities}",
#             f"- Block size: {self.block_size}",
#             f"",
#             f"Files saved:",
#             f"- searching_history(pt_{best_match}).csv",
#             f"- rectified_gray_left.png",
#             f"- rectified_gray_right.png",
#             f"- pixel_info.txt",
#             f"- quality_analysis.json"
#         ]
        
#         with open(point_folder / 'analysis_summary.txt', 'w') as f:
#             f.write('\\n'.join(summary_text))
        
#         self.logger.info(f"Analysis results saved to: {point_folder}")


# class BatchPointAnalyzer:
#     """
#     Batch analyzer for multiple points without interactive GUI.
    
#     Useful for automated analysis of predefined points.
#     """
    
#     def __init__(
#         self,
#         left_image: np.ndarray,
#         right_image: np.ndarray,
#         min_disparity: int = 0,
#         num_disparities: int = 64,
#         block_size: int = 5
#     ):
#         """
#         Initialize batch point analyzer.
        
#         Args:
#             left_image: Left stereo image
#             right_image: Right stereo image
#             min_disparity: Minimum disparity for matching
#             num_disparities: Number of disparities to search
#             block_size: Block size for matching
#         """
#         self.match_analyzer = MatchPointAnalyzer(
#             left_image, right_image, min_disparity, num_disparities, block_size
#         )
#         self.logger = logging.getLogger(__name__)
    
#     def analyze_points(
#         self,
#         points: list,
#         output_folder: str = "batch_analysis"
#     ) -> Dict[str, Any]:
#         """
#         Analyze multiple points in batch mode.
        
#         Args:
#             points: List of (x, y) coordinates to analyze
#             output_folder: Output folder for results
            
#         Returns:
#             Dict[str, Any]: Analysis results for all points
#         """
#         results = {}
#         output_path = Path(output_folder)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         for i, (x, y) in enumerate(points):
#             self.logger.info(f"Analyzing point {i+1}/{len(points)}: ({x}, {y})")
            
#             try:
#                 # Find best match
#                 best_match, search_history = self.match_analyzer.find_best_match(x, y, 'sad')
                
#                 if best_match is None:
#                     results[f"point_{i}"] = {'error': 'No match found'}
#                     continue
                
#                 # Analyze quality
#                 quality_analysis = self.match_analyzer.analyze_matching_quality(
#                     x, y, best_match[0], best_match[1]
#                 )
                
#                 results[f"point_{i}"] = {
#                     'left_point': (x, y),
#                     'best_match': best_match,
#                     'quality_analysis': quality_analysis,
#                     'search_history_length': len(search_history) if search_history is not None else 0
#                 }
                
#             except Exception as e:
#                 self.logger.error(f"Failed to analyze point ({x}, {y}): {e}")
#                 results[f"point_{i}"] = {'error': str(e)}
        
#         # Save batch results
#         import json
#         with open(output_path / 'batch_analysis_results.json', 'w') as f:
#             json.dump(results, f, indent=2)
        
#         self.logger.info(f"Batch analysis complete. Results saved to {output_path}")
#         return results
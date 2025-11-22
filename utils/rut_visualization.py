"""
Rut Depth Visualization Tool

This module provides visualization capabilities for rut depth analysis results.
It creates plots similar to the improved_rut_calculator.py visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class RutDepthVisualizer:
    """
    A class for visualizing rut depth analysis results.
    
    Creates plots showing:
    - Road surface profile
    - Key points (minimum, peaks)
    - Baseline and depth measurement
    - Analysis results
    """
    
    def __init__(self, config=None):
        """
        Initialize the visualizer with configuration.
        
        Args:
            config: Configuration object containing plot settings
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # Default plot settings
        self.default_settings = {
            'create_rut_plot': True,
            'rut_plot_width': 12,
            'rut_plot_height': 8,
            'rut_plot_filename': 'rut_depth_analysis_{pair_name}.png',
            'show_rut_plot': False,
            'plot_dpi': 300
        }
    
    def _setup_logger(self):
        """Setup logger for the visualizer."""
        logger = logging.getLogger(f"{__name__}.RutDepthVisualizer")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _get_setting(self, key: str):
        """Get setting value from config or default."""
        if self.config and hasattr(self.config, key):
            return getattr(self.config, key)
        return self.default_settings.get(key)
    
    def create_rut_depth_plot(self, 
                             points: np.ndarray, 
                             calculation_result: Dict[str, Any],
                             output_folder: str,
                             pair_name: str) -> Optional[str]:
        """
        Create a comprehensive rut depth analysis plot.
        
        Args:
            points: Input coordinate data (x in meters, y in millimeters)
            calculation_result: Result from calculate_rut_value_improved()
            output_folder: Output directory path
            pair_name: Pair name for file naming
            
        Returns:
            str: Path to saved plot file, or None if plotting disabled
        """
        # Check if plotting is enabled
        create_plot = self._get_setting('create_rut_plot')
        if isinstance(create_plot, str):
            create_plot = create_plot.lower() == 'true'
        
        if not create_plot:
            self.logger.info("Rut depth plotting is disabled in configuration")
            return None
        
        try:
            return self._generate_plot(points, calculation_result, output_folder, pair_name)
        except Exception as e:
            self.logger.error(f"Error creating rut depth plot: {str(e)}")
            return None
    
    def _generate_plot(self, 
                      points: np.ndarray, 
                      result: Dict[str, Any],
                      output_folder: str,
                      pair_name: str) -> str:
        """Generate the actual plot."""
        
        # Get plot settings
        plot_width = self._get_setting('rut_plot_width')
        plot_height = self._get_setting('rut_plot_height')
        plot_dpi = self._get_setting('plot_dpi')
        show_plot = self._get_setting('show_rut_plot')
        
        if isinstance(show_plot, str):
            show_plot = show_plot.lower() == 'true'
        
        # Create figure
        plt.figure(figsize=(plot_width, plot_height))
        
        # Plot road surface profile
        plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=1.5, label='Road Surface Profile')
        
        # Plot key points
        min_point = result["min_point"]
        peak1 = result["peak1"]
        peak2 = result["peak2"]
        intersection = result["intersection_point"]
        
        plt.plot(min_point["x"], min_point["y"], 'ro', markersize=10, 
                label=f'Minimum Point (idx: {min_point["index"]})', zorder=5)
        plt.plot(peak1["x"], peak1["y"], 'go', markersize=8, 
                label=f'Peak 1 (idx: {peak1["index"]})', zorder=5)
        plt.plot(peak2["x"], peak2["y"], 'go', markersize=8, 
                label=f'Peak 2 (idx: {peak2["index"]})', zorder=5)
        
        # Plot baseline
        baseline_start = result["baseline_start"]
        baseline_end = result["baseline_end"]
        plt.plot([baseline_start["x"], baseline_end["x"]], 
                [baseline_start["y"], baseline_end["y"]], 
                'g--', linewidth=2, label='Baseline', zorder=4)
        
        # Plot intersection point
        plt.plot(intersection["x"], intersection["y"], 'mo', markersize=6, 
                label='Intersection Point', zorder=5)
        
        # Plot depth measurement line (perpendicular distance)
        plt.plot([min_point["x"], intersection["x"]], 
                [min_point["y"], intersection["y"]], 
                'r-', linewidth=3, 
                label=f'Rut Depth: {result["depth_mm"]:.2f} mm', zorder=4)
        
        # Add depth annotation
        mid_x = (min_point["x"] + intersection["x"]) / 2
        mid_y = (min_point["y"] + intersection["y"]) / 2
        plt.annotate(f'{result["depth_mm"]:.2f} mm', 
                    xy=(mid_x, mid_y), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # Set labels and title
        plt.xlabel('X (m)', fontsize=12)
        plt.ylabel('Y (mm)', fontsize=12)
        # plt.title(f'Rut Depth Analysis - {pair_name}\n'
        #          f'Depth: {result["depth_mm"]:.2f} mm | Method: {result["calculation_method"]}', 
        #          fontsize=14, fontweight='bold')
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        
        # Add analysis information as text box
        info_text = self._create_info_text(result)
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        filename_template = self._get_setting('rut_plot_filename')
        filename = filename_template.format(pair_name=pair_name)
        plot_path = Path(output_folder) / filename
        
        plt.savefig(plot_path, dpi=plot_dpi, bbox_inches='tight')
        self.logger.info(f"Rut depth plot saved to: {plot_path}")
        
        # Show or close plot
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return str(plot_path)
    
    def _create_info_text(self, result: Dict[str, Any]) -> str:
        """Create information text box content."""
        info_lines = [
            f"Analysis Results:",
            f"• Data Points: {result['input_data_points']}",
            f"• Min Point: ({result['min_point']['x']:.3f}m, {result['min_point']['y']:.2f}mm)",
            f"• Peak 1: ({result['peak1']['x']:.3f}m, {result['peak1']['y']:.2f}mm)",
            f"• Peak 2: ({result['peak2']['x']:.3f}m, {result['peak2']['y']:.2f}mm)",
            f"• Baseline Slope: {result['baseline_equation']['slope']:.6f}",
            f"• Intersection Valid: {result['intersection_between_peaks']}"
        ]
        return '\n'.join(info_lines)
    
    # def create_comparison_plot(self, 
    #                           points: np.ndarray,
    #                           original_result: float,
    #                           improved_result: Dict[str, Any],
    #                           output_folder: str,
    #                           pair_name: str) -> Optional[str]:
    #     """
    #     Create a comparison plot between original and improved methods.
        
    #     Args:
    #         points: Input coordinate data
    #         original_result: Result from original method (in mm)
    #         improved_result: Result from improved method
    #         output_folder: Output directory path
    #         pair_name: Pair name for file naming
            
    #     Returns:
    #         str: Path to saved comparison plot, or None if disabled
    #     """
    #     create_plot = self._get_setting('create_rut_plot')
    #     if isinstance(create_plot, str):
    #         create_plot = create_plot.lower() == 'true'
        
    #     if not create_plot:
    #         return None
        
    #     try:
    #         # Create figure with subplots
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
    #         # Left plot: Original method visualization (simplified)
    #         ax1.plot(points[:, 0], points[:, 1], 'b-', linewidth=1.5, label='Road Surface')
    #         ax1.set_title(f'Original Method\nDepth: {original_result:.2f} mm')
    #         ax1.set_xlabel('X Coordinate (m)')
    #         ax1.set_ylabel('Y Coordinate (mm)')
    #         ax1.grid(True, alpha=0.3)
    #         ax1.legend()
            
    #         # Right plot: Improved method visualization
    #         self._plot_improved_method(ax2, points, improved_result)
            
    #         plt.tight_layout()
            
    #         # Save comparison plot
    #         filename = f'rut_depth_comparison_{pair_name}.png'
    #         plot_path = Path(output_folder) / filename
    #         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
    #         show_plot = self._get_setting('show_rut_plot')
    #         if isinstance(show_plot, str):
    #             show_plot = show_plot.lower() == 'true'
            
    #         if show_plot:
    #             plt.show()
    #         else:
    #             plt.close()
            
    #         self.logger.info(f"Comparison plot saved to: {plot_path}")
    #         return str(plot_path)
            
    #     except Exception as e:
    #         self.logger.error(f"Error creating comparison plot: {str(e)}")
    #         return None
    
    def _plot_improved_method(self, ax, points: np.ndarray, result: Dict[str, Any]):
        """Plot improved method results on given axes."""
        # Plot road surface
        ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=1.5, label='Road Surface')
        
        # Plot key points
        min_point = result["min_point"]
        peak1 = result["peak1"]
        peak2 = result["peak2"]
        intersection = result["intersection_point"]
        
        ax.plot(min_point["x"], min_point["y"], 'ro', markersize=8, label='Min Point')
        ax.plot(peak1["x"], peak1["y"], 'go', markersize=6, label='Peaks')
        ax.plot(peak2["x"], peak2["y"], 'go', markersize=6)
        
        # Plot baseline and depth
        baseline_start = result["baseline_start"]
        baseline_end = result["baseline_end"]
        ax.plot([baseline_start["x"], baseline_end["x"]], 
               [baseline_start["y"], baseline_end["y"]], 
               'g--', linewidth=2, label='Baseline')
        
        ax.plot([min_point["x"], intersection["x"]], 
               [min_point["y"], intersection["y"]], 
               'r-', linewidth=3, label=f'Depth: {result["depth_mm"]:.2f} mm')
        
        ax.set_title(f'Improved Method\nDepth: {result["depth_mm"]:.2f} mm')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (mm)')
        ax.grid(True, alpha=0.3)
        ax.legend()
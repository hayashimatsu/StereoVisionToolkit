"""
Interactive coordinate selection tool for height calculation.

This module provides a specialized coordinate selector for rut analysis,
moved from util_height.py and enhanced for better integration.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from typing import Tuple, Optional


class CoordinateSelector:
    """
    Interactive tool for selecting rut coordinates on rectified images.
    
    Provides a graphical interface for users to manually select left and right
    rut coordinates by clicking on overlaid rectified and disparity images.
    """
    
    def __init__(self, image: np.ndarray, disparity: np.ndarray, pair_name: Optional[str] = None):
        """
        Initialize coordinate selector.
        
        Args:
            image: Rectified image for coordinate selection
            disparity: Disparity image for visualization overlay
            pair_name: Optional name for the image pair (for window title)
        """
        self.image = image
        self.disparity = disparity
        self.pair_name = pair_name or "Unknown"
        
        # Selection state
        self.rut_coordinates = {'left': None, 'right': None}
        self.current_side = None
        self.is_left_yet = True
        self.is_right_yet = True
        self.is_both_checked = False
        self.need_get_out = False
        
        # UI configuration
        self.canvas_name = f'Rectified Image ({self.pair_name})'
        self.overlay_weights = (0.4, 0.8)  # (rectified_weight, disparity_weight)

    def select_coordinates(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Start interactive coordinate selection process.
        
        Returns:
            Tuple containing (left_coordinate, right_coordinate) as (x, y) tuples
        """
        self._setup_window()
        
        while True:      
            cv2.setMouseCallback(self.canvas_name, self._on_mouse_click)
            key = cv2.waitKey(0)
            
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                return self.rut_coordinates['left'], self.rut_coordinates['right']

    def _setup_window(self) -> None:
        """Setup OpenCV window with image overlay."""
        img_height, img_width = self.image.shape[:2]
        
        # Calculate window size for reasonable display
        scale_factor = 800 / img_height
        window_width = int(img_width * scale_factor)
        window_height = int(img_height * scale_factor)

        cv2.namedWindow(self.canvas_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.canvas_name, window_width, window_height)

        # Create overlay of rectified and disparity images
        rectify_depth = cv2.addWeighted(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 
            self.overlay_weights[0],
            self.disparity, 
            self.overlay_weights[1], 
            1
        )
        rectify_depth_display = cv2.cvtColor(rectify_depth, cv2.COLOR_BGR2RGB)

        cv2.imshow(self.canvas_name, rectify_depth_display)

    def _on_mouse_click(self, event, x: int, y: int, flags, param) -> None:
        """
        Handle mouse click events for coordinate selection.
        
        Args:
            event: OpenCV mouse event type
            x: X coordinate of mouse click
            y: Y coordinate of mouse click
            flags: Event flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._print_coordinate_info(x, y)

            if self.is_left_yet:
                if self._confirm_selection():
                    self._update_rut_coordinates(x, y)
                    self.is_left_yet = False
                    return
                    
            if self.is_right_yet:
                if self._confirm_selection():
                    self._update_rut_coordinates(x, y)
                    self.is_right_yet = False
                    return
                    
            if not self.is_right_yet and not self.is_left_yet:
                if self.need_get_out:
                    self._need_finish()
                    return
                self._finalize_selection()
                return

    def _need_finish(self) -> None:
        """Display message when selection is complete."""
        message = "You have already done all the selecting\\nNow please press [esc] to go to the next step."
        messagebox.showerror("Notice", message)

    def _confirm_selection(self) -> bool:
        """
        Ask user to confirm coordinate selection.
        
        Returns:
            bool: True if user confirms selection
        """
        if self.is_left_yet:
            self.current_side = "left"
            message = f'Do you want to set this point as the ◆{self.current_side}◇ point?'
        else:
            if self.is_right_yet:
                self.current_side = "right"
                message = f'Do you want to set this point as the ▲{self.current_side}△ point?'
                
        return messagebox.askyesno('Confirm Selection', message)

    def _update_rut_coordinates(self, x: int, y: int) -> None:
        """
        Update stored coordinates for current side.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.rut_coordinates[self.current_side] = (x, y)

    def _finalize_selection(self) -> None:
        """Handle final confirmation of both coordinates."""
        if messagebox.askyesno('Confirm', 'Have you selected both left and right points?'):
            self.need_get_out = True
            word_confirm = (f"left coordinate(x,y) : {self.rut_coordinates['left']}\\n"
                          f"right coordinate(x,y) : {self.rut_coordinates['right']}\\n")
            messagebox.showinfo('Congrats', f'{word_confirm}Please press [esc] to leave')
        else:
            messagebox.showinfo('Reset', 'Please select both points again.')
            self._reset_selection()
            
    def _reset_selection(self) -> None:
        """Reset selection state to start over."""
        self.rut_coordinates = {'left': None, 'right': None}
        self.is_left_yet = True
        self.is_right_yet = True
        self.need_get_out = False
            
    def _print_coordinate_info(self, x: int, y: int) -> None:
        """
        Print coordinate information to console.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        print(f'\\nPixel coordinates: x = {x}, y = {y}')

    def get_selection_status(self) -> dict:
        """
        Get current selection status.
        
        Returns:
            dict: Status information including coordinates and completion state
        """
        return {
            'left_selected': not self.is_left_yet,
            'right_selected': not self.is_right_yet,
            'both_complete': not self.is_left_yet and not self.is_right_yet,
            'coordinates': self.rut_coordinates.copy()
        }

    def set_overlay_weights(self, rectified_weight: float, disparity_weight: float) -> None:
        """
        Set overlay weights for image blending.
        
        Args:
            rectified_weight: Weight for rectified image (0.0-1.0)
            disparity_weight: Weight for disparity image (0.0-1.0)
        """
        self.overlay_weights = (rectified_weight, disparity_weight)
        self._setup_window()  # Refresh display with new weights
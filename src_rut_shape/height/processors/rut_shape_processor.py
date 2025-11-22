"""
Simplified rut shape processor focused on core rut analysis operations.

This processor handles rut-specific operations while delegating common
functionality to utility modules.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Union
from utils.point_processor import PointProcessor
from utils.signal_processor import OutlierFilter
from utils.low_pass_filter import apply_rut_low_pass_filter
from .coordinate_selector import CoordinateSelector


class RutShapeProcessor:
    """
    Simplified processor for rut shape analysis operations.
    
    Focuses on rut-specific processing while leveraging utility functions
    for common operations like interpolation and filtering.
    """
    
    def __init__(self, config):
        """
        Initialize RutShapeProcessor with configuration.

        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.need_fix_extreme_value = config.need_fix_extreme_value == "True"
        self.need_rotating_ground_level = config.need_rotating_ground_level == "True" 
        self.lane_marker_range_L = config.lane_marker_range_L
        self.lane_marker_range_R = config.lane_marker_range_R
        # Diagnostics placeholders for filtering stage
        self.last_outlier_mask = None
        self.last_outlier_stats = None

    # ---- Helper methods for extreme filtering (scaffolding) ----
    def _get_block_indices(self, i: int, n: int, L: int) -> tuple:
        """Compute right-biased block [start, end) of size L for index i.

        If right side lacks points, borrow from left side to keep length L.
        Returns (start, end) as a half-open interval within [0, n].
        """
        if L >= n:
            return 0, n
        forward = min(L, n - i)
        if forward == L:
            return i, i + L
        back = L - forward
        start = max(0, i - back)
        end = n
        # ensure exact length if possible
        if end - start > L:
            start = end - L
        return start, end

    def _detect_true_extremes_blockwise(self, y: np.ndarray, L: int,
                                        allowed_step_y_max: float,
                                        k_y_block: float,
                                        y_ratio_threshold: float,
                                        use_leave_one_out: bool = True,
                                        use_multiscale: bool = True) -> np.ndarray:
        """Blockwise robust detection (median/MAD) of extreme values on Y.

        Returns a boolean mask where True indicates outliers to be repaired.
        Note: This scaffolding implementation returns an all-False mask and
        will be implemented in the next TODO step.
        """
        n = len(y)
        if n == 0:
            return np.zeros(0, dtype=bool)

        def robust_std_mad(vals: np.ndarray) -> float:
            if len(vals) == 0:
                return 1.0
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            s = 1.4826 * mad
            return s if s > 0 else max(np.std(vals), 1e-9)

        mask_L = np.zeros(n, dtype=bool)
        L2 = min(n, max(L * 3 // 2, L + 1))  # a larger window for multiscale voting
        mask_L2 = np.zeros(n, dtype=bool)

        for i in range(n):
            # primary scale
            s, e = self._get_block_indices(i, n, L)
            block = y[s:e]
            if use_leave_one_out and e - s > 1:
                # exclude self
                rel = i - s
                block = np.concatenate([block[:rel], block[rel+1:]])
            med = np.median(block) if len(block) else y[i]
            srob = robust_std_mad(block)
            z = abs(y[i] - med) / srob
            ratio = abs(y[i]) / (abs(med) + 1e-9)
            # step preservation relative to local median
            step_ok = abs(y[i] - med) <= allowed_step_y_max
            mask_L[i] = (z > k_y_block) and (ratio > y_ratio_threshold) and (not step_ok)

            if use_multiscale and L2 != L:
                s2, e2 = self._get_block_indices(i, n, L2)
                block2 = y[s2:e2]
                if use_leave_one_out and e2 - s2 > 1:
                    rel2 = i - s2
                    block2 = np.concatenate([block2[:rel2], block2[rel2+1:]])
                med2 = np.median(block2) if len(block2) else y[i]
                srob2 = robust_std_mad(block2)
                z2 = abs(y[i] - med2) / srob2
                ratio2 = abs(y[i]) / (abs(med2) + 1e-9)
                step_ok2 = abs(y[i] - med2) <= allowed_step_y_max
                mask_L2[i] = (z2 > k_y_block) and (ratio2 > y_ratio_threshold) and (not step_ok2)

        return mask_L | mask_L2

    def _repair_outliers_linear(self, arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Repair only masked indices using linear interpolation/extrapolation.

        Placeholder: will be implemented in a later TODO step.
        Returns a copy of arr.
        """
        n = len(arr)
        if n == 0 or not np.any(mask):
            return arr.copy()

        repaired = arr.copy()
        x = repaired[:, 0]
        y = repaired[:, 1]

        # Precompute nearest valid neighbors on both sides
        prev_valid = np.full(n, -1, dtype=int)
        next_valid = np.full(n, -1, dtype=int)
        last = -1
        for i in range(n):
            if not mask[i]:
                last = i
            prev_valid[i] = last
        nxt = -1
        for i in range(n - 1, -1, -1):
            if not mask[i]:
                nxt = i
            next_valid[i] = nxt

        def clamp(val, a, b):
            lo, hi = (a, b) if a <= b else (b, a)
            return min(max(val, lo), hi)

        for i in range(n):
            if not mask[i]:
                continue
            lv = prev_valid[i]
            rv = next_valid[i]

            if lv != -1 and rv != -1 and rv != lv:
                # interpolate between lv and rv
                alpha = (i - lv) / float(rv - lv)
                xi = x[lv] + (x[rv] - x[lv]) * alpha
                yi = y[lv] + (y[rv] - y[lv]) * alpha
                # clamp within endpoints
                xi = clamp(xi, x[lv], x[rv])
                yi = clamp(yi, y[lv], y[rv])
                x[i], y[i] = xi, yi
            elif lv != -1:
                # one-sided extrapolation using two left-side valid points if possible
                l2 = prev_valid[lv]
                if l2 != -1 and l2 != lv:
                    # slope from (l2 -> lv)
                    dx = (x[lv] - x[l2]) / float(lv - l2)
                    dy = (y[lv] - y[l2]) / float(lv - l2)
                    xi = x[lv] + dx * (i - lv)
                    yi = y[lv] + dy * (i - lv)
                    # clamp toward x[l2], x[lv]
                    xi = clamp(xi, x[l2], x[lv])
                    yi = clamp(yi, y[l2], y[lv])
                    x[i], y[i] = xi, yi
                else:
                    # fallback to nearest valid
                    x[i], y[i] = x[lv], y[lv]
            elif rv != -1:
                # one-sided extrapolation using two right-side valid points if possible
                r2 = next_valid[rv]
                if r2 != -1 and r2 != rv:
                    dx = (x[rv] - x[r2]) / float(rv - r2)
                    dy = (y[rv] - y[r2]) / float(rv - r2)
                    xi = x[rv] + dx * (i - rv)
                    yi = y[rv] + dy * (i - rv)
                    xi = clamp(xi, x[rv], x[r2])
                    yi = clamp(yi, y[rv], y[r2])
                    x[i], y[i] = xi, yi
                else:
                    x[i], y[i] = x[rv], y[rv]
            else:
                # all masked (degenerate)
                pass

        # Enforce local monotonicity for replaced points (w.r.t neighbors)
        # If both neighbors exist and x should be monotonic between them, clamp accordingly.
        for i in range(n):
            if not mask[i]:
                continue
            lv = prev_valid[i]
            rv = next_valid[i]
            if lv != -1 and rv != -1 and rv != lv:
                # enforce x between x[lv] and x[rv]
                x[i] = clamp(x[i], x[lv], x[rv])

        repaired[:, 0] = x
        repaired[:, 1] = y
        return repaired

    def process_rut_shape_with_reference(self,
                                       world_coordinates: np.ndarray,
                                       rectified_image: Optional[np.ndarray] = None,
                                       disparity_image: Optional[np.ndarray] = None,
                                       coordinates: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                                       reference_shape: str = None) -> np.ndarray:
        """
        Process rut shape data with reference to a template shape.
        
        Args:
            world_coordinates: 3D world coordinate data
            rectified_image: Optional rectified image for manual coordinate selection
            disparity_image: Optional disparity image for visualization
            coordinates: Optional pre-defined coordinates
            reference_shape: Path to reference shape data
            
        Returns:
            np.ndarray: Processed coordinates aligned with reference shape
        """
        # Get initial rut shape
        if coordinates is None and rectified_image is not None and disparity_image is not None:
            coordinates = self._get_user_selected_coordinates(rectified_image, disparity_image)
            
        _, rut_coords_world = self.interpolate_rut_shape(world_coordinates, coordinates)
        data = rut_coords_world[:, :2]  # Extract x,y coordinates
        
        # Load and process reference shape
        try:
            reference_data = np.genfromtxt(reference_shape, delimiter=',', skip_header=1, encoding='utf-8')
        except Exception as e:
            print(f"Error loading reference data: {str(e)}")
            return None
        
        # Align data with reference shape
        data_aligned = self._align_with_reference(data, reference_data)
        
        return data_aligned

    def interpolate_rut_shape(self, 
                            world_coordinates: np.ndarray,
                            coordinates: Tuple[Tuple[int, int], Tuple[int, int]]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate points between two coordinates to create a rut shape.
        
        Args:
            world_coordinates: 3D world coordinate data
            coordinates: Tuple of ((x1,y1), (x2,y2)) coordinates
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (image coordinates, world coordinates)
        """
        (x1, y1), (x2, y2) = coordinates
        
        # Use utility function for interpolation
        num_points = max(abs(x2 - x1) + 1, 10)
        interpolated_points = PointProcessor.interpolate_points_between_coordinates(
            (float(x1), float(y1)), (float(x2), float(y2)), num_points
        )
        
        # Convert to integer coordinates for image indexing
        rut_image = np.round(interpolated_points).astype(int)
        rut_world = self._calculate_world_coordinates(rut_image, world_coordinates)
        
        return rut_image, rut_world

    def _calculate_world_coordinates(self, 
                                  image_points: Union[List[Tuple[int, int]], np.ndarray],
                                  world_coordinates: np.ndarray
                                  ) -> np.ndarray:
        """
        Convert image coordinates to world coordinates.

        Args:
            image_points: Array or list of image coordinates
            world_coordinates: 3D world coordinate data

        Returns:
            np.ndarray: Array of world coordinates with shape (n, 3)
        """
        if isinstance(image_points, list):
            image_points = np.array(image_points)
            
        world_coords = []
        for x, y in image_points:
            world_coord = world_coordinates[y, x] / 1000.0
            world_coords.append(np.round(world_coord, 4))
            
        return np.array(world_coords)
    
    def _align_with_reference(self, data: np.ndarray, reference_data: np.ndarray) -> np.ndarray:
        """
        Align rut shape data with reference shape.
        
        Args:
            data: Rut shape data to align
            reference_data: Reference shape data
            
        Returns:
            np.ndarray: Aligned rut shape data
        """
        # First rotate data to horizontal
        data_horizontal = self._rotate_to_horizontal(data)
        data_ground_adjust = self._adjust_ground_level(data_horizontal)
        
        # Calculate reference shape angle
        ref_angle = self._calculate_regression_angle(reference_data, is_reference=True)
        
        # Interpolate to match reference x-coordinates
        data_interpolated = self._interpolate_to_reference(data_ground_adjust, reference_data)
        
        # Rotate to match reference angle
        data_aligned = self._rotate_coordinates(data_interpolated, ref_angle)
        
        return data_aligned
    
    def _adjust_ground_level(self, data: np.ndarray) -> np.ndarray:
        """
        Adjust ground level by normalizing to average ground height.
        
        Args:
            data: 2D array of x,y coordinates with shape (n, 2)
            
        Returns:
            np.ndarray: Ground-level adjusted coordinates with shape (n, 2)
        """
        ground_start = data[1:self.lane_marker_range_L]
        ground_end = data[-self.lane_marker_range_L:]
        average_ground_level = np.mean([
            np.mean(ground_start[:, 1]),
            np.mean(ground_end[:, 1])
        ])
        
        adjusted_coords = data.copy()
        adjusted_coords[:, 1] -= average_ground_level
        return adjusted_coords

    def _get_user_selected_coordinates(self, 
                                    rectified_image: np.ndarray,
                                    disparity_image: np.ndarray
                                    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get coordinates through user selection interface.

        Args:
            rectified_image: Image for coordinate selection
            disparity_image: Disparity image for visualization

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: Selected coordinates
        """
        selector = CoordinateSelector(rectified_image, disparity_image)
        return selector.select_coordinates()

    def _convert_rectified_to_world_coordinates(self,
                                              rectified_points: List[List[float]],
                                              world_coordinates: np.ndarray
                                              ) -> np.ndarray:
        """
        Convert a list of rectified image coordinates to world coordinates.
        
        Args:
            rectified_points: List of [x, y] coordinates from the rectified image
            world_coordinates: 3D world coordinate data
            
        Returns:
            np.ndarray: Array of world coordinates with shape (n, 3)
        """
        world_coords = []
        for point in rectified_points:
            x, y = point
            world_coord = world_coordinates[y, x]
            world_coords.append(np.round(world_coord, 4))
        return np.array(world_coords)

    def _calculate_regression_angle(self, data: np.ndarray, is_reference=False) -> float:
        """
        Calculate regression line angle from ground points.
        
        Args:
            data: Input coordinate data
            is_reference: Whether this is reference data
            
        Returns:
            float: Angle in radians
        """
        

        ground_start = data[1:self.lane_marker_range_L]
        ground_end = data[-self.lane_marker_range_R:]
        data_rotate_ref = np.vstack((ground_start, ground_end))
        # data_rotate_ref =data[self.lane_marker_range_L:-self.lane_marker_range_R]
        
        slope, _ = np.polyfit(data_rotate_ref[:, 0], data_rotate_ref[:, 1], 1)
        return np.arctan(slope)

    def _rotate_coordinates(self, data: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate coordinates by given angle.
        
        Args:
            data: Input coordinate data
            angle: Rotation angle in radians
            
        Returns:
            np.ndarray: Rotated coordinates
        """
        rotation_matrix = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        return np.dot(data, rotation_matrix)

    def _convert_coordinates_to_world(self, world_coordinates: np.ndarray, rectified_image: Optional[np.ndarray], 
                                    disparity_image: Optional[np.ndarray], coordinates) -> np.ndarray:
        """Convert input coordinates to world coordinates."""
        if coordinates is None and rectified_image is not None and disparity_image is not None:
            coordinates = self._get_user_selected_coordinates(rectified_image, disparity_image)
            _, rut_coords_world = self.interpolate_rut_shape(world_coordinates, coordinates)
        elif isinstance(coordinates, list) and len(coordinates) > 2:
            rut_coords_world = self._convert_rectified_to_world_coordinates(coordinates, world_coordinates)
        elif isinstance(coordinates, tuple) and len(coordinates) == 2:
            _, rut_coords_world = self.interpolate_rut_shape(world_coordinates, coordinates)
        else:
            raise ValueError("Invalid coordinates format. Expected list of interpolated points or tuple of two points.")
        return rut_coords_world
    
    def _extract_xy_coordinates(self, rut_coords_world: np.ndarray) -> np.ndarray:
        """Extract x,y coordinates from world coordinates."""
        return rut_coords_world[:, :2]
    
    def _apply_extreme_value_filtering(self, data: np.ndarray) -> np.ndarray:
        """Apply extreme value filtering if enabled.

        Strategy (phase 1: core logic without config wiring/logging):
        - Robust detection on both X and Y using median/MAD globally and locally.
        - Additional detection via first differences (ΔX, ΔY) to catch spikes.
        - Preserve legitimate Y-steps by default threshold while X remains normal.
        - Interpolate removed segments linearly to keep length unchanged.
        """
        if not self.need_fix_extreme_value:
            return data

        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 2:
            return data

        arr = data[:, :2].astype(float, copy=True)
        n = len(arr)
        if n == 0:
            return arr

        x = arr[:, 0]
        y = arr[:, 1]

        # Parameters for blockwise detection
        block_size_factor = float(getattr(self.config, 'block_size_factor', 0.2))
        min_block_size = int(getattr(self.config, 'min_block_size', 25))
        k_y_block = float(getattr(self.config, 'k_y_block', 6.0))
        y_ratio_threshold = float(getattr(self.config, 'y_ratio_threshold', 10.0))
        allowed_step_y = float(getattr(self.config, 'allowed_step_y_max', 120.0))

        # Determine block size L
        L = max(int(np.ceil(n * block_size_factor)), min_block_size)
        if L > n:
            L = n
        if L < 3:
            L = 3

        # Non-finite mask passes through as outliers to be repaired
        non_finite = (~np.isfinite(x)) | (~np.isfinite(y))

        # Blockwise detection on Y only (relative extremes)
        mask_block = self._detect_true_extremes_blockwise(
            y, L, allowed_step_y, k_y_block, y_ratio_threshold,
            use_leave_one_out=True, use_multiscale=True
        )

        is_outlier = non_finite | mask_block

        if not np.any(is_outlier):
            # store diagnostics
            self.last_outlier_mask = is_outlier
            self.last_outlier_stats = {
                'total': int(n),
                'removed': 0,
                'non_finite': int(non_finite.sum())
            }
            return arr

        # Repair only marked indices, keep length
        repaired = self._repair_outliers_linear(arr, is_outlier)

        # store diagnostics
        self.last_outlier_mask = is_outlier
        self.last_outlier_stats = {
            'total': int(n),
            'removed': int(is_outlier.sum()),
            'non_finite': int(non_finite.sum())
        }

        return repaired
    
    def _apply_rotation_calculation(self, data: np.ndarray) -> np.ndarray:
        """Calculate and apply rotation to make data horizontal."""
        angle = self._calculate_regression_angle(data)
        return self._rotate_coordinates(data, -angle)
    
    def _adjust_depth_to_camera_center(self, data: np.ndarray) -> np.ndarray:
        """Adjust depth values relative to camera center position."""
        adjusted_data = data.copy()
        adjusted_data[:,1] = adjusted_data[:,1][0] - adjusted_data[:,1]
        return adjusted_data

    def _interpolate_to_reference(self, data: np.ndarray, reference_data: np.ndarray) -> np.ndarray:
        """
        Interpolate data to match reference x-axis range while maintaining original data length.
        
        Args:
            data: Input coordinate data
            reference_data: Reference shape data
            
        Returns:
            np.ndarray: Interpolated coordinates with original data length but scaled to reference x-range
        """
        # Get reference x-axis range
        ref_x_start = reference_data[0, 0]
        ref_x_end = reference_data[-1, 0]
        
        # Create new x coordinates with same length as data but scaled to reference range
        new_x = np.linspace(ref_x_start, ref_x_end, len(data))
        
        # Keep original y coordinates
        new_y = data[:, 1]*1000
        
        return np.column_stack((new_x, new_y))

    def apply_low_pass_filter(
        self, 
        rut_data: np.ndarray, 
        padding_method: str = "reflection_extrapolation",
        padding_size_percent: float = 25.0
    ) -> np.ndarray:
        """
        Apply low pass filter to rut shape data.
        
        Args:
            rut_data (np.ndarray): Input rut data with shape (n, 2) where columns are [x, y]
            padding_method (str): Method for padding edges
            padding_size_percent (float): Padding size as percentage of data length
                
        Returns:
            np.ndarray: Filtered rut data with shape (n, 2)
        """
        # Use the unified low-pass filter from utils
        return apply_rut_low_pass_filter(
            rut_data, 
            self.config.low_pass_filter_cutoff,
            padding_method,
            padding_size_percent
        )

"""
Data scaling and normalization utilities.

This module provides functions for scaling, normalizing, and transforming
coordinate data used in stereo vision and rut analysis:
- Width scaling operations
- Data normalization and standardization
- Unit conversions
- Range transformations

Author: Extracted from height module refactoring
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import logging


logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_LANE_WIDTH = 3.5  # meters
DEFAULT_SCALING_VERSION = "width_scaling_v1"
MM_TO_M = 0.001
M_TO_MM = 1000.0


def scale_width_to_target(data: np.ndarray, target_width: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Scale x-coordinates to match target width and translate to start at 0.
    
    Args:
        data: Coordinate data with shape (n, 2) where columns are [x, y]
        target_width: Target width in meters
    
    Returns:
        Tuple of (scaled_data, scaling_metadata)
    """
    if data.shape[1] < 2:
        raise ValueError("Data must have at least 2 columns")
    
    # Extract coordinates
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    
    # Calculate scaling parameters
    current_width = np.max(x_coords) - np.min(x_coords)
    
    if current_width == 0:
        logger.warning("Current width is zero, cannot scale")
        return data.copy(), _create_scaling_metadata(0, target_width, 1.0, 0, x_coords)
    
    scale_factor = target_width / current_width
    
    # Apply scaling and translation
    x_scaled = x_coords * scale_factor
    x_translated = x_scaled - np.min(x_scaled)
    
    # Combine results
    scaled_data = np.column_stack((x_translated, y_coords))
    
    # Create metadata
    metadata = _create_scaling_metadata(current_width, target_width, scale_factor, 
                                      np.min(x_scaled), x_coords, x_translated)
    
    logger.info(f"Applied width scaling: original={current_width:.3f}m, "
                f"target={target_width:.3f}m, scale_factor={scale_factor:.3f}")
    
    return scaled_data, metadata


def _create_scaling_metadata(original_width: float, target_width: float, 
                           scale_factor: float, translation_offset: float,
                           original_x: np.ndarray, scaled_x: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Create scaling metadata dictionary."""
    metadata = {
        "original_width": float(original_width),
        "target_width": float(target_width),
        "scale_factor": float(scale_factor),
        "translation_offset": float(translation_offset),
        "original_x_range": [float(np.min(original_x)), float(np.max(original_x))],
        "version": DEFAULT_SCALING_VERSION
    }
    
    if scaled_x is not None:
        metadata["scaled_x_range"] = [float(np.min(scaled_x)), float(np.max(scaled_x))]
    
    return metadata


def normalize_data(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data with shape (n, d)
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Tuple of (normalized_data, normalization_parameters)
    """
    if method == 'minmax':
        return _normalize_minmax(data)
    elif method == 'zscore':
        return _normalize_zscore(data)
    elif method == 'robust':
        return _normalize_robust(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _normalize_minmax(data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Min-max normalization to [0, 1] range."""
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_range = data_max - data_min
    
    # Avoid division by zero
    data_range = np.where(data_range == 0, 1, data_range)
    
    normalized = (data - data_min) / data_range
    
    params = {
        'method': 'minmax',
        'min_values': data_min.tolist(),
        'max_values': data_max.tolist(),
        'range_values': data_range.tolist()
    }
    
    return normalized, params


def _normalize_zscore(data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Z-score normalization (standardization)."""
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    
    # Avoid division by zero
    data_std = np.where(data_std == 0, 1, data_std)
    
    normalized = (data - data_mean) / data_std
    
    params = {
        'method': 'zscore',
        'mean_values': data_mean.tolist(),
        'std_values': data_std.tolist()
    }
    
    return normalized, params


def _normalize_robust(data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Robust normalization using median and IQR."""
    data_median = np.median(data, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    iqr = q75 - q25
    
    # Avoid division by zero
    iqr = np.where(iqr == 0, 1, iqr)
    
    normalized = (data - data_median) / iqr
    
    params = {
        'method': 'robust',
        'median_values': data_median.tolist(),
        'q25_values': q25.tolist(),
        'q75_values': q75.tolist(),
        'iqr_values': iqr.tolist()
    }
    
    return normalized, params


def denormalize_data(normalized_data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Reverse normalization using stored parameters.
    
    Args:
        normalized_data: Normalized data
        params: Parameters from normalization
        
    Returns:
        Original scale data
    """
    method = params['method']
    
    if method == 'minmax':
        min_vals = np.array(params['min_values'])
        range_vals = np.array(params['range_values'])
        return normalized_data * range_vals + min_vals
    
    elif method == 'zscore':
        mean_vals = np.array(params['mean_values'])
        std_vals = np.array(params['std_values'])
        return normalized_data * std_vals + mean_vals
    
    elif method == 'robust':
        median_vals = np.array(params['median_values'])
        iqr_vals = np.array(params['iqr_values'])
        return normalized_data * iqr_vals + median_vals
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def convert_units(data: np.ndarray, from_unit: str, to_unit: str, 
                 columns: Optional[Union[int, list]] = None) -> np.ndarray:
    """
    Convert units for specified columns.
    
    Args:
        data: Input data
        from_unit: Source unit ('m', 'mm', 'cm')
        to_unit: Target unit ('m', 'mm', 'cm')
        columns: Column indices to convert (None for all)
        
    Returns:
        Data with converted units
    """
    conversion_factors = {
        ('m', 'mm'): M_TO_MM,
        ('mm', 'm'): MM_TO_M,
        ('m', 'cm'): 100.0,
        ('cm', 'm'): 0.01,
        ('mm', 'cm'): 0.1,
        ('cm', 'mm'): 10.0
    }
    
    if from_unit == to_unit:
        return data.copy()
    
    factor = conversion_factors.get((from_unit, to_unit))
    if factor is None:
        raise ValueError(f"Unsupported unit conversion: {from_unit} to {to_unit}")
    
    result = data.copy()
    
    if columns is None:
        result *= factor
    else:
        if isinstance(columns, int):
            columns = [columns]
        for col in columns:
            result[:, col] *= factor
    
    logger.info(f"Converted units from {from_unit} to {to_unit} (factor: {factor})")
    
    return result


def scale_to_range(data: np.ndarray, target_min: float, target_max: float,
                  axis: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Scale data to specified range.
    
    Args:
        data: Input data
        target_min: Target minimum value
        target_max: Target maximum value
        axis: Axis along which to scale (None for all)
        
    Returns:
        Tuple of (scaled_data, scaling_parameters)
    """
    if target_min >= target_max:
        raise ValueError("target_min must be less than target_max")
    
    if axis is None:
        data_min = np.min(data)
        data_max = np.max(data)
    else:
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
    
    data_range = data_max - data_min
    target_range = target_max - target_min
    
    # Avoid division by zero
    data_range = np.where(data_range == 0, 1, data_range)
    
    scaled = (data - data_min) / data_range * target_range + target_min
    
    params = {
        'original_min': float(np.min(data_min)),
        'original_max': float(np.max(data_max)),
        'target_min': target_min,
        'target_max': target_max,
        'axis': axis
    }
    
    return scaled, params


def apply_offset_and_scale(data: np.ndarray, offset: Union[float, np.ndarray], 
                          scale: Union[float, np.ndarray]) -> np.ndarray:
    """
    Apply offset and scale transformation: result = (data + offset) * scale.
    
    Args:
        data: Input data
        offset: Offset to add
        scale: Scale factor to multiply
        
    Returns:
        Transformed data
    """
    return (data + offset) * scale


def remove_offset(data: np.ndarray, method: str = 'min') -> Tuple[np.ndarray, Union[float, np.ndarray]]:
    """
    Remove offset from data.
    
    Args:
        data: Input data
        method: Method to calculate offset ('min', 'mean', 'median')
        
    Returns:
        Tuple of (offset_removed_data, offset_value)
    """
    if method == 'min':
        offset = np.min(data, axis=0)
    elif method == 'mean':
        offset = np.mean(data, axis=0)
    elif method == 'median':
        offset = np.median(data, axis=0)
    else:
        raise ValueError(f"Unknown offset method: {method}")
    
    return data - offset, offset


def calculate_scaling_statistics(original_data: np.ndarray, scaled_data: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics comparing original and scaled data.
    
    Args:
        original_data: Original data
        scaled_data: Scaled data
        
    Returns:
        Dictionary with comparison statistics
    """
    return {
        'original_shape': original_data.shape,
        'scaled_shape': scaled_data.shape,
        'original_range': {
            'min': float(np.min(original_data)),
            'max': float(np.max(original_data)),
            'mean': float(np.mean(original_data)),
            'std': float(np.std(original_data))
        },
        'scaled_range': {
            'min': float(np.min(scaled_data)),
            'max': float(np.max(scaled_data)),
            'mean': float(np.mean(scaled_data)),
            'std': float(np.std(scaled_data))
        },
        'scale_factor': float(np.std(scaled_data) / np.std(original_data)) if np.std(original_data) != 0 else 1.0
    }
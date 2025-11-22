from typing import List, Union, Tuple, Optional
from scipy import signal
import numpy as np



def apply_butterworth_filter(data: np.ndarray, dt: float, fc: float) -> np.ndarray:
    """
    Apply a zero-phase Butterworth filter to the data.
    
    Args:
        data (np.ndarray): Input signal
        dt (float): Time/space step size
        fc (float): Cutoff frequency
        
    Returns:
        np.ndarray: Filtered signal
    """
    # Calculate sample frequency
    fs = 1.0 / dt
    
    # Normalized cutoff frequency
    Wn = 2.0 * fc / fs
    Wn = min(0.99, max(0.01, Wn))  # Ensure it's within valid range
    
    # Design Butterworth filter (4th order)
    b, a = signal.butter(4, Wn, 'low')
    
    # Apply zero-phase filtering (filtfilt applies the filter forward 
    # and backward, resulting in zero phase distortion)
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def apply_rut_low_pass_filter(
    rut_data: np.ndarray, 
    cutoff_frequency: float,
    padding_method: str = "reflection_extrapolation",
    padding_size_percent: float = 25.0
) -> np.ndarray:
    """
    Apply low pass filter to rut shape data with enhanced padding to mitigate Gibbs phenomenon.
    
    This is an advanced implementation specifically designed for rut shape analysis,
    with sophisticated edge padding to minimize filter artifacts.
    
    Args:
        rut_data (np.ndarray): Input rut data with shape (n, 2) where columns are [x, y]
        cutoff_frequency (float): Filter cutoff frequency
        padding_method (str): Method for padding edges. Options:
            - "reflection": Simple reflection padding
            - "extrapolation": Linear extrapolation based on edge slopes
            - "reflection_extrapolation": Blend of reflection and extrapolation (recommended)
            - "constant": Extends with constant edge values
        padding_size_percent (float): Padding size as percentage of data length
            
    Returns:
        np.ndarray: Filtered rut data with shape (n, 2)
    """
    if cutoff_frequency is None:
        return rut_data
    
    # Calculate lane width
    lane_width = (rut_data[-1, 0] - rut_data[0, 0]) / 1000  # Convert to meters
    
    # Calculate dx (distance between points)
    dx = lane_width / (len(rut_data) - 1)
    
    # Extract y values
    y_data = rut_data[:, 1]
    n_data = len(y_data)
    
    # Determine padding size (increase for better results)
    pad_size = max(int(n_data * padding_size_percent / 100), 30)
    
    # Apply padding based on selected method
    if padding_method == "reflection":
        # Simple reflection padding
        left_pad = y_data[1:pad_size+1][::-1]
        right_pad = y_data[-pad_size-1:-1][::-1]
        padded_y = np.concatenate([left_pad, y_data, right_pad])
        
    elif padding_method == "extrapolation":
        # Linear extrapolation based on edge slopes
        left_slope = (y_data[1] - y_data[0]) / dx
        right_slope = (y_data[-1] - y_data[-2]) / dx
        
        left_x = np.arange(-pad_size, 0) * dx
        right_x = np.arange(1, pad_size + 1) * dx
        
        left_pad = y_data[0] + left_slope * left_x
        right_pad = y_data[-1] + right_slope * right_x
        
        padded_y = np.concatenate([left_pad, y_data, right_pad])
        
    elif padding_method == "reflection_extrapolation":
        # Blended approach (reflection + extrapolation)
        # Reflection padding
        left_mirror = y_data[1:pad_size+1][::-1]
        right_mirror = y_data[-pad_size-1:-1][::-1]
        
        # Extrapolation padding
        left_slope = (y_data[1] - y_data[0]) / dx
        right_slope = (y_data[-1] - y_data[-2]) / dx
        
        left_x = np.arange(-pad_size, 0) * dx
        right_x = np.arange(1, pad_size + 1) * dx
        
        left_extrap = y_data[0] + left_slope * left_x
        right_extrap = y_data[-1] + right_slope * right_x
        
        # Create weights for blending (more weight to reflection near data, 
        # more weight to extrapolation at the far edges)
        weights = np.linspace(0, 1, pad_size)
        
        left_pad = weights * left_mirror + (1 - weights) * left_extrap
        right_pad = weights[::-1] * right_mirror + (1 - weights[::-1]) * right_extrap
        
        padded_y = np.concatenate([left_pad, y_data, right_pad])
        
    elif padding_method == "constant":
        # Constant value padding
        left_pad = np.full(pad_size, y_data[0])
        right_pad = np.full(pad_size, y_data[-1])
        padded_y = np.concatenate([left_pad, y_data, right_pad])
    
    else:
        raise ValueError(f"Unknown padding method: {padding_method}")
    
    # Apply a window function to smooth transitions at the edges
    window = np.ones(len(padded_y))
    edge_taper = int(pad_size * 0.3)  # Taper length as a fraction of padding
    
    # Apply half-cosine window at edges
    for i in range(edge_taper):
        factor = 0.5 * (1 - np.cos(np.pi * i / edge_taper))
        window[i] = factor  # Left edge
        window[-(i+1)] = factor  # Right edge
    
    # Apply window to reduce edge effects
    windowed_y = padded_y * window
    
    # Apply the filter using SciPy's zero-phase filter (better than FFT-based)
    filtered_padded_y = apply_butterworth_filter(windowed_y, dx, cutoff_frequency)
    
    # Extract only the original part (removing padding)
    filtered_y = filtered_padded_y[pad_size:pad_size+n_data]
    
    # Create x values in mm
    filtered_x = np.linspace(rut_data[0, 0], rut_data[-1, 0], n_data)
    
    return np.column_stack((filtered_x, filtered_y))
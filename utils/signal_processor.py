# signal_processor.py
from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd

class SignalProcessor:
    """Handles signal processing operations like filtering"""
    
    def __init__(self, cutoff_freq=2.0, sampling_rate=100):
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self._setup_filter()

    def _setup_filter(self):
        """Sets up the Butterworth filter parameters"""
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = butter(4, normalized_cutoff, btype='low', analog=False)

    def apply_filter(self, data):
        """Applies low-pass filter to the input data"""
        if len(data) < 3:  # Minimum required points for filtering
            return data
            
        try:
            filtered_data = filtfilt(self.b, self.a, data)
            return filtered_data
        except Exception as e:
            print(f"Filtering error: {str(e)}")
            return data
        
        
class OutlierFilter:
    """Handles removal of outliers and extreme values from curve data"""
    
    def __init__(self, window_size=10, tolerance_factor=2.0, cluster_threshold=3, need_smooth_data = False):
        self.window_size = window_size
        self.tolerance_factor = tolerance_factor
        self.cluster_threshold = cluster_threshold
        self.smooth_data_switch = need_smooth_data


    def filter_extremes(self, _data):
        """
        Main filtering method that combines multiple approaches to remove extremes.
        
        Args:
            data: numpy array of shape (n, 2) containing x, y coordinates
            
        Returns:
            numpy array of filtered data with same structure
        """
        data = _data.copy()
        
        try:
            # 確保數據格式正確
            if data.shape[1] != 2:
                raise ValueError("Input data must have shape (n, 2)")

            filtered_data = self._remove_local_outliers(data)
            # 平滑處理（可選）
            if self.smooth_data_switch:
                filtered_data = self._smooth_data(filtered_data)
            
            return filtered_data
            
        except Exception as e:
            print(f"Error in filtering extremes: {str(e)}")
            return data
        
        


    # def _remove_local_outliers(self, data):
    #     """Remove global outliers using IQR method"""
    #     y = data[:, 1]
    #     q1 = np.percentile(y, 25)
    #     q3 = np.percentile(y, 75)
    #     iqr = q3 - q1
    #     lower_bound = q1 - self.iqr_scale * iqr
    #     upper_bound = q3 + self.iqr_scale * iqr
        
    #     mask = (y >= lower_bound) & (y <= upper_bound)
    #     return data[mask]

    def _remove_local_outliers(self, data):
        """
        Remove outliers by analyzing local segments of the curve.
        Uses multiple statistical measures to ensure high precision.
        """
        x, y = data[:, 0], data[:, 1]
        valid_indices = np.ones(len(y), dtype=bool)
        half_window = self.window_size // 2

        for i in range(len(y)):
            # 獲取局部窗口範圍
            start_idx = max(0, i - half_window)
            end_idx = min(len(y), i + half_window + 1)
            
            # 取得當前窗口的值
            window_x = np.linspace(start_idx, end_idx, end_idx-start_idx)
            window_y = y[start_idx:end_idx]
            
            if len(window_y) < 3:  # 確保有足夠的數據點做分析
                continue
                
            # 1. 計算窗口的主要統計量
            window_median = np.median(window_y)
            q1 = np.percentile(window_y, 25)
            q3 = np.percentile(window_y, 75)
            iqr = q3 - q1
            
            # 2. 計算該點與趨勢的偏差
            if i > 0 and i < len(y) - 1:
                # 使用最小二乘法計算局部線性回歸
                # 計算局部變化率
                coeffs = np.polyfit(window_x, window_y, 1)
                local_line = np.polyval(coeffs, i) #並返回多項式在該點的值。
                
                deviation_from_regression  = abs(y[i] - local_line)
                
                # 計算回歸線的預測區間
                regression_std = np.std(window_y - np.polyval(coeffs, window_x))
                regression_threshold = regression_std * self.tolerance_factor
                
                # 設定動態閾值
                threshold = min(
                    self.tolerance_factor * iqr,
                    max(abs(coeffs[0]) * 2, iqr)
                )
                
                # 3. 綜合多個條件判斷是否為離群值
                is_outlier = (
                    (deviation_from_regression  > regression_threshold) and       # 基於趨勢的判斷
                    (abs(y[i] - q1) > iqr * 1.5 or abs(y[i] - q3) > iqr * 1.5) 
                ) # 基於四分位數的判斷
                
                
                if is_outlier:
                    valid_indices[i] = False

        # 4. 對被標記為離群值的點進行插值處理
        filtered_data = data[valid_indices]
        
        # if len(filtered_data) < len(data):
        #     # 使用三次樣條插值填補移除的點
        #     from scipy.interpolate import CubicSpline
            
        #     cs = CubicSpline(filtered_data[:, 0], filtered_data[:, 1])
        #     y_interpolated = cs(data[:, 0])
            
        #     return np.column_stack((data[:, 0], y_interpolated))
        
        return filtered_data

    def _smooth_data(self, data, smoothing_factor=0.8):
        """
        Optional smoothing using Savitzky-Golay filter
        """
        from scipy.signal import savgol_filter
        
        window = min(self.window_size * 2 + 1, len(data) - 1)
        if window % 2 == 0:
            window -= 1
            
        if window > 2:
            smoothed_y = savgol_filter(
                data[:, 1],
                window_length=window,
                polyorder=2
            )
            return np.column_stack((data[:, 0], smoothed_y))
        return data

    # def _interpolate_gaps(self, filtered_data, original_x):
    #     """
    #     插值處理被過濾掉的點，確保輸出數據與輸入數據有相同的x座標
    #     """
    #     from scipy.interpolate import interp1d
        
    #     # 創建插值函數
    #     f = interp1d(filtered_data[:, 0], filtered_data[:, 1],
    #                  kind='linear', bounds_error=False, fill_value='extrapolate')
        
    #     # 對原始x座標進行插值
    #     interpolated_y = f(original_x)
        
    #     return np.column_stack((original_x, interpolated_y))
    
  
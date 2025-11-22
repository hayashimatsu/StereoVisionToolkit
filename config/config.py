import json
from typing import Dict, Any
import os
import shutil
import string

class Config:
    def __init__(self, config_path: str):
        self.config_data = self._load_config(config_path)
        self._check_folder(self.config_data["save_path_temp"], is_temp= True)
        self._check_folder(self.config_data["save_path_result"])
        self._validate_resize_config()
        self._init_extreme_value_defaults()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as config_file:
            config_data = json.load(config_file)
        
        # Process string formatting for paths that contain {case_name}
        self._process_string_formatting(config_data)
        return config_data
    
    def _process_string_formatting(self, config_data: Dict[str, Any]) -> None:
        """Process string formatting in config values, replacing {case_name} with actual value."""
        case_name = config_data.get("case_name", "")

        
        for key, value in config_data.items():
            if isinstance(value, str) and "{case_name}" in value:
                try:
                    config_data[key] = value.format(case_name=case_name)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Could not format value for key '{key}': {e}")
                    # Keep original value if formatting fails
                    pass
    
    def _validate_resize_config(self) -> None:
        """Validate resize configuration parameters and set defaults if needed."""
        # Set default values for resize parameters if not present
        resize_defaults = {
            "resize_target_width": None,
            "resize_target_height": None,
            "resize_scale": None,
            "resize_max_pixels": 3840 * 2160  # Default memory-safe threshold
        }
        
        for key, default_value in resize_defaults.items():
            if key not in self.config_data:
                self.config_data[key] = default_value
        
        # Validate resize configuration
        target_width = self.config_data.get("resize_target_width")
        target_height = self.config_data.get("resize_target_height")
        resize_scale = self.config_data.get("resize_scale")
        
        # Check for conflicting resize configurations
        resize_configs = [
            target_width is not None and target_height is not None,
            resize_scale is not None
        ]
        
        if sum(resize_configs) > 1:
            print("Warning: Multiple resize configurations detected. Priority order:")
            print("1. Target dimensions (resize_target_width/height)")
            print("2. Scale factor (resize_scale)")
            print("3. Auto-resize (default)")
        
        # Validate target dimensions
        if target_width is not None or target_height is not None:
            if target_width is None or target_height is None:
                raise ValueError("Both resize_target_width and resize_target_height must be specified together")
            if target_width <= 0 or target_height <= 0:
                raise ValueError("Target dimensions must be positive integers")
        
        # Validate scale factor
        if resize_scale is not None:
            if not isinstance(resize_scale, (int, float)) or resize_scale <= 0:
                raise ValueError("resize_scale must be a positive number")
            if resize_scale > 2.0:
                print(f"Warning: Large resize scale ({resize_scale}) may increase memory usage")
        
        # Validate max pixels threshold
        max_pixels = self.config_data.get("resize_max_pixels")
        if max_pixels is not None and (not isinstance(max_pixels, int) or max_pixels <= 0):
            raise ValueError("resize_max_pixels must be a positive integer")

    def _init_extreme_value_defaults(self) -> None:
        """Initialize default parameters for extreme-value filtering.

        These serve as centralized defaults for the blockwise robust detector
        used in rut shape extreme filtering. Config values (if present) take
        precedence over these defaults.
        """
        defaults = {
            # Blockwise detection
            "block_size_factor": 0.2,       # L = ceil(N * factor)
            "min_block_size": 25,          # minimum block length
            "k_y_block": 6.0,              # robust z-score threshold
            "y_ratio_threshold": 10.0,     # magnitude ratio guard
            "allowed_step_y_max": 60.0,   # step preservation threshold (vs block median)
            # Multiscale and robustness
            "use_multiscale_block_detection": "True",
            "multiscale_factor": 1.5,
            "use_leave_one_out": "True",
            # Repair behavior
            "enforce_x_monotonicity": "True",
            "repair_clamp_to_endpoints": "True",
            # Diagnostics
            "export_outlier_mask": "False"
        }

        # Expose on the instance for downstream consumers
        self.config_data.setdefault("extreme_value_default", defaults)

        # Also ensure top-level keys exist if not provided, using defaults above
        for k, v in defaults.items():
            self.config_data.setdefault(k, v)

    def get_extreme_value_defaults(self) -> Dict[str, Any]:
        """Return a copy of extreme value default parameters."""
        return dict(self.config_data.get("extreme_value_default", {}))
    
    def _check_folder(self, folder_name, is_temp=False):
        counter = 1
        new_path = f"result/{folder_name}"
        # check the folder is exist or not 
        if is_temp:
            if not os.path.exists(new_path):
                os.makedirs(new_path, exist_ok=True)
            else:
                # empty the folder and create the new one
                shutil.rmtree(new_path)
                os.makedirs(new_path)
            self.config_data["save_path_temp"]=new_path
        else:
            while os.path.exists(new_path):
                new_path = f"result/{folder_name}({counter})"
                counter += 1
            os.makedirs(new_path)
            self.config_data["save_path_result"]=new_path
    
    def get_resize_config_summary(self) -> str:
        """Get a summary of the current resize configuration."""
        target_width = self.config_data.get("resize_target_width")
        target_height = self.config_data.get("resize_target_height")
        resize_scale = self.config_data.get("resize_scale")
        max_pixels = self.config_data.get("resize_max_pixels")
        
        if target_width is not None and target_height is not None:
            return f"Target dimensions: {target_width}x{target_height}"
        elif resize_scale is not None:
            return f"Scale factor: {resize_scale}"
        else:
            return f"Auto-resize (threshold: {max_pixels} pixels)"
    
    def is_resize_enabled(self) -> bool:
        """Check if any resize configuration is enabled."""
        target_width = self.config_data.get("resize_target_width")
        target_height = self.config_data.get("resize_target_height")
        resize_scale = self.config_data.get("resize_scale")
        
        return (target_width is not None and target_height is not None) or resize_scale is not None
                
    def __getattr__(self, name: str) -> Any:
        if name in self.config_data:
            return self.config_data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

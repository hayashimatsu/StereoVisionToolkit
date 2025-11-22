# 📘 **StereoVisionToolkit — Road Rut Depth Measurement (mm accuracy)**

A complete stereo-vision processing pipeline for **millimeter-level rut depth estimation** using smartphone stereo images or any calibrated stereo camera pair.
This project includes a fully modular architecture, robust rectification for **non-synchronized stereo cameras**, accurate 3D reconstruction, and multi-stage rut-shape extraction.

# 🚀 Overview

This toolkit implements an end-to-end geometric vision pipeline:

1. **Stereo Rectification (robust for non-synchronized cameras)**
2. **SGBM-based disparity computation (auto-tuned parameters)**
3. **3D reconstruction using corrected Q matrix**
4. **Rut profile extraction (slope correction, filtering, baseline alignment)**
5. **Final rut depth measurement in millimeters**

The system is designed to be:

* Engineering-accurate
* Modular and extensible
* Suitable for research, road-inspection prototyping, or 3D reconstruction tasks

---

# 🖼️ Example Input/Output

### **Input Images**
> ⚠️ Note  
> The photo includes a 1/60‑second time lag: the right frame is captured slightly later than the left.

|                                 Left camera                                 |                                 Right camera                                |
| :-------------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
| ![Left](document/image_demo/left_001.jpg) | ![Right](document/image_demo/right_001.jpg) |



### **Rectified Output**
> 💡 Tip  
> Use a yellow line to indicate the section selected for rut‑shape calculation.

![Rectified](document/image_demo/rectified_marked_001.jpg)


### **Final Rut Depth Result**

![Rut Profile](document/image_demo/rut_depth_analysis_001.png)

> 📊 Comparison  
> The result is compared with the LiDAR data. Notice that the bottom position is almost aligned.

![Rut Profile](document/answer/graph.jpg)

---

# 🎯 Key Features

### ✔ **1. Robust Stereo Rectification for Non-Synchronized Cameras**

Standard `cv2.stereoRectify()` assumes synchronized stereo inputs.
Real smartphone captures often violate this assumption due to:

* Time lag between left/right images
* Moving objects on the road
* Camera motion
* Vertical/horizontal parallax
* FOV mismatch

This project implements an enhanced rectification pipeline:

#### Improvements:

* Auto-calculation of minimal bounding box to prevent FOV loss
* Correction of rectification parameters (`alpha`, scaling, ROI)
* Recalculation of projection matrices (P1/P2) with shifted principal points
* Regeneration of Q matrix for metric-accurate reconstruction
* Guaranteed full-frame rectification even with time-lagged pairs

These corrections enable stable disparity estimation and accurate 3D reconstruction.

### ✔ **2. Auto-Tuned SGBM Disparity**

Automatically determines `numDisparities` and SGBM parameters based on:

* baseline
* focal length (pixels)
* expected depth range
* target accuracy

Provides:

* dense disparities
* sub-pixel refinement
* noise suppression for road surfaces

### ✔ **3. Metric-Accurate 3D Reconstruction**

Using the corrected Q matrix, the system produces:

* millimeter-level world coordinates
* ground-plane alignment (XYZ rotation)
* consistent metrics regardless of input resolution

### ✔ **4. Multi-Stage Rut Profile Extraction**

Includes:

* Outlier removal (MAD-based)
* Slope correction
* Baseline normalization
* Optional low-pass filtering
* Final rut depth using geometric intersection

All intermediate results can be saved for debugging or research.

---
# 🔄 **Processing Pipeline**
```
Left/Right Images
Parameter (K1.csv, K2.csv, d1.csv, d2.csv, R.csv, T.csv, left_<case>.json)
        │
        ▼
[1] Rectification
    • Undistortion + normalization
    • Rotation to rectify epipolar lines
    • Auto-resized bounding box
    • Adjusted P1/P2 and regenerated Q
        │
        ▼
[2] Disparity Estimation (SGBM)
    • Horizontal matching on rectified pair
    • Auto-calculated disparity range
    • Sub-pixel refinement
        │
        ▼
[3] 3D Reconstruction
    • Reproject disparity → (X,Y,Z) using corrected Q
    • Convert camera coords → road coords
    • Produce metric-accurate depth/point cloud
        │
        ▼
[4] Rut Profile Extraction
    • Sample 3D points along seed-defined line
    • Remove outliers & correct slope
    • Normalize height & smooth profile
    • Compute rut depth (mm)
        │
        ▼
Final Output (Rut Depth, mm)

```

---

# 📚 Additional Documentation (Theory & Details)

If you want deeper explanation of algorithms and implementation:

### **📘 Program Deep Dive**

* `document/PROJECT_DEEP_DIVE.md`
  Detailed system description, module hierarchy, and full algorithmic explanations.

### **📘 Stereo Camera Theory**

* `document/ステレオカメラを用いたわだちぼれ量の算出_第二章_2次元座標から3次元座標への変換.docx`
* `document/ステレオカメラを用いたわだちぼれ量の算出_第三章_ステレオ画像の平行化処理.docx`

These explain:

* 2D → 3D coordinate transformation
* Rectification geometry

---

# 📂 **Repository Structure**

```
StereoVisionToolkit/
├── main.py                          # Entry point, orchestrates the full pipeline

├── config/
│   ├── config.py                    # Configuration loader and validator
│   ├── config_rut_shape.json        # Main rut shape configuration
│   └── config_rut_shape1.json       # Alternative configuration
├── src_rut_shape/
│   ├── rut_shape.py                 # High-level rut extraction pipeline
│   ├── rectify_refactored.py        # Stage 1: Stereo rectification (improved)
│   ├── disparity_refactored.py      # Stage 2: Disparity calculation (SGBM)
│   ├── depth.py                     # Stage 3: 3D reconstruction
│   ├── height_refactored.py         # Stage 4: Rut shape extraction
│   ├── base/
│   │   ├── file_manager.py          # File I/O operations
│   │   └── processor.py             # Template for pipeline processors
│   ├── rectification/
│   │   ├── engine.py                # Core rectification engine
│   │   ├── matrix_calculator.py     # P1/P2/Q matrix correction
│   │   └── file_manager.py          # Rectification file I/O
│   ├── disparity/
│   │   ├── sgbm_engine.py           # SGBM computation engine
│   │   ├── parameter_calculator.py  # Auto-parameter tuning
│   │   └── disparity_processor.py   # Post-processing (sub-pixel, filtering)
│   └── height/
│       ├── processors.py            # Profile filtering, slope correction
│       ├── rut_calculator.py        # Final rut depth estimation
│       ├── image_loader.py          # Image and data loader
│       ├── coordinate_processor.py  # Coordinate frame alignment (XYZ rotation)
│       └── file_manager.py          # File operations for height stage
├── utils/
│   ├── point_processor.py           # Geometric utilities
│   ├── image_processing.py          # Image manipulation helpers
│   ├── low_pass_filter.py           # Signal filtering
│   ├── data_scaling.py              # Coordinate scaling helpers
│   ├── rut_visualization.py         # Rut plotting utilities
│   ├── visualizer.py                # Misc visualization
│   ├── stereo_math.py               # Stereo geometry calculations
│   ├── file_operations.py           # File I/O
│   └── logger_config.py             # Logging configuration
└── document/
    ├── README.md                    # User guide (this file)
    ├── PROJECT_DEEP_DIVE.md         # Technical deep dive
    └── TECHNOLOGY_TRANSFER.md       # Implementation documentation
```

## 📦 Data Conventions

### **Images**

```
data/<dataset>/<case_name>/set_*/<pair_name>/
    left_<pair>.jpg
    right_<pair>.jpg
```

### **Parameters**

Stored under:

```
parameter/<dataset>/<case_name>
```

Required files:

* `K1.csv`, `d1.csv`
* `K2.csv`, `d2.csv`
* `R.csv`, `T.csv`
* `disparityToDepthMap.csv`
* `left_<pair>.json`
  → includes `rut_1`, `rut_2` seed endpoints in the original image

Here is the input example:
```json
{
  "case_name"         : "001",
  "image_set_folder"  : "data/2024_1106/{case_name}",
  "parameter_path"    : "parameter/2024_1106/{case_name}"
}
```
---

# ▶ How to Run

```
python main.py --config config/config_rut_shape.json
```
All required parameters are defined in the JSON file.

Inputs:

* left/right images
* calibration parameters (K1, K2, d1, d2, R, T)
* seed points for rut-line interpolation

Outputs:

* disparity map
* 3D world coordinates
* rut profile
* final rut depth (mm)

---

# 🧪 Applications

* Road surface inspection
* Infrastructure monitoring
* Stereo depth estimation research
* Smartphone-based 3D measurement
* Geometry-based computer vision experimentation

---

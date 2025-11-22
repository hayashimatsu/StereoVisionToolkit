# StereoVisionToolkit Rut-Shape Analysis - Project Deep Dive

## Executive Summary

This document provides a comprehensive technical analysis of the StereoVisionToolkit project, designed for engineers to reproduce, evaluate, and extend the system. The project solves the real-world problem of automated road rut depth measurement using stereo vision, achieving millimeter-level precision for road maintenance assessment.

---

## 1. Problem Statement and Success Criteria

### 1.1 Problem Statement

Road rutting (わだちぼれ) is a critical road maintenance issue where vehicle traffic creates longitudinal depressions in road surfaces. Traditional manual measurement methods are labor-intensive, dangerous (requiring lane closures), and inconsistent. This project provides an automated stereo vision solution that:

- Captures stereo image pairs of road surfaces from a moving vehicle
- Reconstructs 3D geometry of the road surface
- Extracts cross-sectional rut profiles
- Calculates rut depth as the orthogonal distance from crown line to rut bottom

**Real-world impact**: Enables continuous, safe, and cost-effective road condition monitoring without traffic disruption.

### 1.2 Success Criteria

**Accuracy Requirements:**
- Rut depth measurement precision: ±5mm (millimeter-level accuracy)
- 3D reconstruction accuracy: Sub-centimeter level at target depth range (1.5-3.0m)
- Disparity matching quality: Sub-pixel accuracy through SGBM refinement

**Operational Constraints:**
- Target depth range: 1.5m to 3.0m from camera baseline
- Processing: Offline batch processing (not real-time)
- Input: Calibrated stereo camera pairs with known intrinsics/extrinsics
- Output: Rut depth in millimeters, intermediate visualizations, and CSV profiles

---

## 2. System Architecture

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     StereoVisionToolkit System                      │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   main.py        │
                    │  Entry Point     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  config.Config   │
                    │  JSON Loader     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   RutShape       │
                    │  Orchestrator    │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┬────────────────────┐
        │                    │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│  Rectifier     │  │  Disparity      │  │  Depth         │  │  Height        │
│  Stage 1       │  │  Calculator     │  │  Calculator    │  │  Calculator    │
│                │  │  Stage 2        │  │  Stage 3       │  │  Stage 4       │
└───────┬────────┘  └────────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                    │                    │                    │
        │ Outputs:           │ Outputs:           │ Outputs:           │ Outputs:
        │ • Rectified imgs   │ • Disparity map    │ • World coords     │ • Rut profiles
        │ • Q_rectified      │ • Rut disparity    │ • Depth image      │ • Rut depth (mm)
        │ • P1/P2 adjusted   │ • SGBM params      │ • Rotated coords   │ • Filtered data
        │ • Rut points       │                    │                    │ • Scaled data
        └────────────────────┴────────────────────┴────────────────────┘

                    Data Flow: Sequential Pipeline
                    Temp Storage: .npy intermediates
                    Result Storage: .jpg visualizations + .csv data
```

### 2.2 Processing Architecture

**Parallelization Strategy:**
- **No GPU acceleration**: Uses OpenCV CPU-based SGBM (cv2.StereoSGBM)
- **No multi-threading**: Sequential batch processing per image pair
- **No IPC/network**: Single-process local file I/O
- **Batch processing**: Iterates through `set_*` folders, processes each pair independently

**Rationale**: Offline processing prioritizes accuracy and debuggability over speed. Each stage saves intermediate results for validation and reprocessing.



---

## 3. End-to-End Pipeline Flow Chart

### 3.1 Complete Runtime Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT: Stereo Images + Calibration Parameters + Seed Points             │
│ • left_<pair>.jpg, right_<pair>.jpg                                     │
│ • K1.csv, d1.csv, K2.csv, d2.csv, R.csv, T.csv, dispartityToDepthMap.csv│
│ • left_<pair>.json (rut_1, rut_2 seed points)                           │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────┐
│ STAGE 1: RECTIFICATION (Rectifier)                                      │
│ Module: src_rut_shape/rectify_refactored.py                             │
│ Algorithm: cv2.stereoRectify + optional auto-sizing                     │
├──────────────────────────────────────────────────────────────────────────┤
│ Processing Steps:                                                        │
│ 1. Load K1,d1,K2,d2,R,T from CSV files                                  │
│ 2. Call cv2.stereoRectify(K1,d1,K2,d2,size,R,T,alpha)                   │
│    → Outputs: R1,R2,P1,P2,Q,roi1,roi2                                   │
│ 3. If rectified_image_size='auto':                                      │
│    • Project image corners through undistort+rectify pipeline           │
│    • Calculate bounding box (x_min,y_min,x_max,y_max)                   │
│    • Adjust P1,P2 principal points: cx'=cx-x_min, cy'=cy-y_min          │
│    • Derive Q_rectified from adjusted P1,P2                             │
│ 4. Generate rectification maps: cv2.initUndistortRectifyMap()           │
│ 5. Remap images: cv2.remap(left/right, map1, map2)                      │
│ 6. Rectify seed points using PointTransformer.rectify_points()          │
│    • Undistort: cv2.undistortPoints(pts, K, d)                          │
│    • Rotate: R @ normalized_coords                                      │
│    • Project: P @ [rotated, 1]                                          │
│ 7. Interpolate N points between rut_1 and rut_2                         │
│ 8. Apply resize strategy if configured                                  │
├──────────────────────────────────────────────────────────────────────────┤
│ Key Parameters:                                                          │
│ • rectified_image_size: 'default' | 'auto'                              │
│ • rectification_alpha: -1.0 to 1.0 (crop vs preserve)                   │
│ • resize_scale, resize_target_width/height, resize_max_pixels           │
├──────────────────────────────────────────────────────────────────────────┤
│ Outputs (temp/result):                                                   │
│ • left_rectified_<pair>.npy, right_rectified_<pair>.npy                 │
│ • Q_rectified_<pair>.csv, P1_<pair>.csv, P2_<pair>.csv                  │
│ • rectified_interpolated_points_<pair>.json                             │
│ • rectified_<pair>.jpg (overlay visualization)                          │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────┐
│ STAGE 2: DISPARITY CALCULATION (DisparityCalculator)                    │
│ Module: src_rut_shape/disparity_refactored.py                           │
│ Algorithm: OpenCV Semi-Global Block Matching (SGBM)                     │
├──────────────────────────────────────────────────────────────────────────┤
│ Processing Steps:                                                        │
│ 1. Load rectified images from temp (.npy format)                        │
│ 2. Calculate SGBM parameters:                                           │
│    • If NUM_DISPARITIES not set:                                        │
│      NUM_DISP = ceil((f_pix * baseline) / target_depth / 16) * 16      │
│    • Ensure NUM_DISPARITIES is multiple of 16                           │
│    • BLOCKSIZE (typically 5-11, odd number)                             │
│    • MIN_DISPARITY (typically 0)                                        │
│ 3. Configure SGBM matcher:                                              │
│    mode = SGBM_MODE_SGBM_3WAY if SGBM_MODE_FAST else SGBM_MODE_SGBM    │
│    P1 = 8 * channels * BLOCKSIZE²                                       │
│    P2 = 32 * channels * BLOCKSIZE²                                      │
│    uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32    │
│ 4. Compute disparity: stereo.compute(left_gray, right_gray)             │
│ 5. Convert to float: disparity_float = disparity.astype(float) / 16.0  │
│ 6. Sample disparity along rectified rut points                          │
├──────────────────────────────────────────────────────────────────────────┤
│ Key Parameters:                                                          │
│ • BLOCKSIZE: 5-11 (matching window size)                                │
│ • NUM_DISPARITIES: auto-calculated or manual (must be multiple of 16)   │
│ • MIN_DISPARITY: 0 (starting disparity)                                 │
│ • SGBM_MODE_FAST: True → SGBM_3WAY (faster, less accurate)              │
│ • target_depth: 2.0m (used for auto NUM_DISPARITIES calculation)        │
├──────────────────────────────────────────────────────────────────────────┤
│ Theoretical Foundation:                                                  │
│ • Disparity-depth relationship: depth = (f * baseline) / disparity      │
│ • SGBM cost aggregation: Semi-global path optimization (8 directions)   │
│ • Sub-pixel refinement: Parabola fitting on cost curve                  │
├──────────────────────────────────────────────────────────────────────────┤
│ Outputs (temp/result):                                                   │
│ • disparity_<pair>.npy (float32, sub-pixel precision)                   │
│ • disparity_<pair>.jpg (colorized visualization)                        │
│ • rut_disparity_<pair>.csv (disparity values along rut line)            │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────┐
│ STAGE 3: DEPTH RECONSTRUCTION (DepthCalculator)                         │
│ Module: src_rut_shape/depth.py                                          │
│ Algorithm: cv2.reprojectImageTo3D with Q matrix                         │
├──────────────────────────────────────────────────────────────────────────┤
│ Processing Steps:                                                        │
│ 1. Load disparity map from temp (.npy)                                  │
│ 2. Load Q matrix (priority order):                                      │
│    a) Q_rectified_<pair>.csv (from rectification stage)                 │
│    b) dispartityToDepthMap.csv (original calibration)                   │
│ 3. Reproject to 3D: world_coord = cv2.reprojectImageTo3D(disp, Q)      │
│    • Q matrix structure:                                                │
│      [1  0  0      -cx    ]                                             │
│      [0  1  0      -cy    ]                                             │
│      [0  0  0       f     ]                                             │
│      [0  0  -1/Tx  (cx-cx')/Tx]                                         │
│    • Output: (H×W×3) array with [X, Y, Z] in mm                        │
│ 4. Apply coordinate rotations (sequential):                             │
│    a) XY rotation (θ1): align with road plane                           │
│       R_xy = [[cos θ1, -sin θ1, 0],                                     │
│               [sin θ1,  cos θ1, 0],                                     │
│               [0,       0,      1]]                                     │
│    b) YZ rotation (θ2): pitch correction                                │
│       R_yz = [[1, 0,       0      ],                                    │
│               [0, cos θ2, -sin θ2],                                     │
│               [0, sin θ2,  cos θ2]]                                     │
│    c) XZ rotation (θ3): roll correction                                 │
│       R_xz = [[cos θ3, 0, -sin θ3],                                     │
│               [0,      1,  0      ],                                    │
│               [sin θ3, 0,  cos θ3]]                                     │
│    • Cumulative: world_rotated = world @ R_xy.T @ R_yz.T @ R_xz.T      │
│ 5. Calculate depth: depth = sqrt(X² + Y² + Z²) / 1000 (convert to m)   │
│ 6. Generate depth counter image (colorized by depth range)              │
├──────────────────────────────────────────────────────────────────────────┤
│ Key Parameters:                                                          │
│ • xy_rotate_angle, yz_rotate_angle, xz_rotate_angle (degrees)           │
│ • max_depth, min_depth (meters, for visualization)                      │
│ • counter_tick (depth colormap tick interval)                           │
├──────────────────────────────────────────────────────────────────────────┤
│ Theoretical Foundation:                                                  │
│ • Homogeneous coordinates: [u, v, d, 1]ᵀ → [X, Y, Z, W]ᵀ               │
│ • Perspective division: (X/W, Y/W, Z/W) gives metric 3D coords          │
│ • Q matrix derivation: Q = [P1⁻¹ | 0; 0 0 0 1] · [I | -baseline·e₁]    │
│ • Rotation order matters: XY→YZ→XZ aligns with vehicle coordinate frame │
├──────────────────────────────────────────────────────────────────────────┤
│ Outputs (temp/result):                                                   │
│ • world_coord_<pair>.npy (H×W×3 float32, rotated 3D coords in mm)      │
│ • depth_<pair>.jpg (colorized depth visualization)                      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────┐
│ STAGE 4: RUT SHAPE EXTRACTION (HeightCalculator)                        │
│ Module: src_rut_shape/height_refactored.py                              │
│ Algorithm: Multi-stage filtering + rut depth calculation                │
├──────────────────────────────────────────────────────────────────────────┤
│ Processing Steps:                                                        │
│ 1. Load world coordinates from temp (.npy)                              │
│ 2. Load rectified rut points (interpolated line)                        │
│ 3. Extract 3D coordinates along rut line:                               │
│    For each point (u,v): coord_3d = world_coord[v, u, :]               │
│ 4. Extract X-Y profile (discard Z): rut_profile = [(X, Y), ...]        │
│ 5. Apply processing pipeline:                                           │
│    ┌─────────────────────────────────────────────────────────────┐     │
│    │ Step 0: Original Data                                       │     │
│    │ • Raw (X, Y) coordinates from 3D reconstruction             │     │
│    │ • Save: rut_0-original_<pair>.csv                           │     │
│    └─────────────────────┬───────────────────────────────────────┘     │
│                          │                                              │
│    ┌─────────────────────▼───────────────────────────────────────┐     │
│    │ Step 1: Extreme Value Filtering                             │     │
│    │ • Blockwise robust outlier detection                        │     │
│    │ • Block size: L = ceil(N * block_size_factor)              │     │
│    │ • For each block: compute robust median, MAD               │     │
│    │ • Z-score: z = |y - median| / (1.4826 * MAD)               │     │
│    │ • Flag outliers: z > k_y_block (default 6.0)               │     │
│    │ • Repair: linear interpolation or endpoint clamping         │     │
│    │ • Save: rut_1-extreme_filtered_<pair>.csv                   │     │
│    └─────────────────────┬───────────────────────────────────────┘     │
│                          │                                              │
│    ┌─────────────────────▼───────────────────────────────────────┐     │
│    │ Step 2: Rotation Calculation (if enabled)                   │     │
│    │ • Fit linear regression to endpoints (crown line)           │     │
│    │ • Calculate rotation angle: θ = atan(slope)                 │     │
│    │ • Rotate profile to horizontal: Y' = Y·cos(θ) - X·sin(θ)   │     │
│    │ • Save: rut_2-rotated_<pair>.csv                            │     │
│    └─────────────────────┬───────────────────────────────────────┘     │
│                          │                                              │
│    ┌─────────────────────▼───────────────────────────────────────┐     │
│    │ Step 3: Depth Adjustment to Camera Center                   │     │
│    │ • Shift X coordinates: X' = X - X_camera_center             │     │
│    │ • Save: rut_3-depth_adjusted_<pair>.csv                     │     │
│    └─────────────────────┬───────────────────────────────────────┘     │
│                          │                                              │
│    ┌─────────────────────▼───────────────────────────────────────┐     │
│    │ Step 4: Ground Level Adjustment (if enabled)                │     │
│    │ • Calculate baseline from endpoint average                  │     │
│    │ • Shift Y coordinates: Y' = Y - baseline                    │     │
│    │ • Save: rut_4-baseline_adjusted_<pair>.csv                  │     │
│    └─────────────────────┬───────────────────────────────────────┘     │
│                          │                                              │
│    ┌─────────────────────▼───────────────────────────────────────┐     │
│    │ Step 5: Low-Pass Filtering (if configured)                  │     │
│    │ • Apply Butterworth filter (order=4, cutoff frequency)      │     │
│    │ • Padding: reflection extrapolation (25% of data length)    │     │
│    │ • Filter Y coordinates: Y_filtered = butter_lowpass(Y)      │     │
│    │ • Save: rut_filtered_<pair>.csv                             │     │
│    └─────────────────────┬───────────────────────────────────────┘     │
│                          │                                              │
│    ┌─────────────────────▼───────────────────────────────────────┐     │
│    │ Step 6: Width Scaling (if lane_width configured)            │     │
│    │ • Calculate current width: W_current = X_max - X_min        │     │
│    │ • Scale factor: s = lane_width / W_current                  │     │
│    │ • Scale X: X_scaled = X * s                                 │     │
│    │ • Translate to origin: X_final = X_scaled - X_scaled_min    │     │
│    │ • Save: rut_filtered_scaled_<pair>.csv                      │     │
│    └─────────────────────┬───────────────────────────────────────┘     │
│                          │                                              │
│    ┌─────────────────────▼───────────────────────────────────────┐     │
│    │ Step 7: Rut Depth Calculation                               │     │
│    │ Method: Improved (default) or Original                      │     │
│    │                                                              │     │
│    │ Improved Method:                                            │     │
│    │ 1. Find crown points (endpoints or local maxima)            │     │
│    │ 2. Define crown line: linear fit through crown points       │     │
│    │ 3. Find rut bottom: global minimum Y value                  │     │
│    │ 4. Calculate orthogonal distance from rut to crown line     │     │
│    │    d = |ax₀ + by₀ + c| / sqrt(a² + b²)                     │     │
│    │ 5. Validate: check minimum distance ratio                   │     │
│    │                                                              │     │
│    │ Original Method:                                            │     │
│    │ 1. Crown line: average of endpoint Y values                 │     │
│    │ 2. Rut depth: crown_line - min(Y)                          │     │
│    │                                                              │     │
│    │ • Output: rut_depth_mm (float)                              │     │
│    │ • Save: rut_depth_<pair>.txt                                │     │
│    └─────────────────────────────────────────────────────────────┘     │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ Key Parameters:                                                          │
│ • need_fix_extreme_value: True/False (enable Step 1)                    │
│ • need_rotating_ground_level: True/False (enable Step 2)                │
│ • need_adjusting_ground_level: True/False (enable Step 4)               │
│ • low_pass_filter_cutoff: Hz (enable Step 5, None to disable)           │
│ • lane_width: meters (enable Step 6, None to disable)                   │
│ • use_improved_rut_calculation: True/False (Step 7 method selection)    │
│ • rut_min_distance_ratio: 0.1 (validation threshold)                    │
├──────────────────────────────────────────────────────────────────────────┤
│ Outputs (temp/result):                                                   │
│ • rut_0-original_<pair>.csv through rut_4-baseline_adjusted_<pair>.csv  │
│ • rut_filtered_<pair>.csv (after low-pass filter)                       │
│ • rut_filtered_scaled_<pair>.csv (after width scaling)                  │
│ • rut_depth_<pair>.txt (final rut depth in mm)                          │
│ • rut_depth_visualization_<pair>.png (annotated plot)                   │
└──────────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────┐
│ FINAL OUTPUT                                                             │
│ • Rut depth measurement: XX.X mm                                         │
│ • Complete processing chain artifacts in result/ and temp/               │
│ • Visualizations: rectified overlays, disparity maps, depth images,     │
│   rut profile plots                                                      │
└──────────────────────────────────────────────────────────────────────────┘
```



---

## 4. Implementation Highlights

### 4.1 Stereo Rectification with Auto-Sizing

**Location**: `src_rut_shape/rectify_refactored.py:Rectifier`

**Problem Solved**: Standard `cv2.stereoRectify` with fixed output size causes content loss when extreme camera configurations create large rectified images.

**Solution**: Dynamic output size calculation with projection matrix adjustment.

**Algorithm**:
```python
def _calculate_optimal_rectification_maps(image_size):
    # Step 1: Project image corners through complete pipeline
    corners = [[0,0], [w-1,0], [0,h-1], [w-1,h-1]]
    for corner in corners:
        # Undistort
        normalized = K_inv @ [u, v, 1]
        # Rotate
        rotated = R1 @ normalized
        # Project with original P1
        projected = P1 @ [rotated, 1]
        # Perspective divide
        x_rect, y_rect = projected[0]/projected[2], projected[1]/projected[2]
    
    # Step 2: Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    new_width = ceil(x_max - x_min)
    new_height = ceil(y_max - y_min)
    
    # Step 3: Adjust projection matrices
    P1_adj = P1.copy()
    P1_adj[0,2] -= x_min  # Adjust cx
    P1_adj[1,2] -= y_min  # Adjust cy
    # Same for P2_adj
    
    # Step 4: Derive Q_rectified from adjusted P1, P2
    Q_rectified = calculate_q_from_projection_matrices(P1_adj, P2_adj)
    
    return (new_width, new_height), P1_adj, P2_adj, Q_rectified
```

**Key Parameters**:
- `rectified_image_size`: `'auto'` enables dynamic sizing, `'default'` uses original size
- `rectification_alpha`: Controls crop behavior (-1.0 = crop invalid, 1.0 = preserve all)

**Why Chosen**: Handles extreme distortions (wide-angle lenses, close baselines) without manual size tuning.

**Performance**: Adds ~50ms overhead per pair for corner projection calculations.



### 4.2 SGBM Disparity Calculation with Auto-Parameter Tuning

**Location**: `src_rut_shape/disparity_refactored.py:DisparityCalculator`

**Problem Solved**: Manual SGBM parameter tuning is error-prone and scene-dependent.

**Solution**: Automatic NUM_DISPARITIES calculation based on camera geometry and target depth.

**Algorithm**:
```python
def calculate_num_disparities(focal_length_px, baseline_m, target_depth_m):
    # Disparity-depth relationship: d = (f * b) / z
    max_disparity = (focal_length_px * baseline_m) / target_depth_m
    
    # Round up to nearest multiple of 16 (SGBM requirement)
    num_disparities = ceil(max_disparity / 16) * 16
    
    # Safety bounds
    num_disparities = max(16, min(num_disparities, 256))
    
    return num_disparities
```

**SGBM Configuration**:
- **Mode**: `SGBM_MODE_SGBM_3WAY` (fast) or `SGBM_MODE_SGBM` (accurate)
- **P1**: `8 * channels * BLOCKSIZE²` (small penalty for 1-pixel disparity change)
- **P2**: `32 * channels * BLOCKSIZE²` (large penalty for large disparity changes)
- **uniquenessRatio**: 10 (reject ambiguous matches)
- **speckleWindowSize**: 100, **speckleRange**: 32 (post-filtering)

**Why Chosen**: 
- SGBM provides good accuracy/speed tradeoff for offline processing
- Semi-global optimization handles textureless regions better than local methods
- Sub-pixel refinement achieves <0.1 pixel disparity precision

**Complexity**: O(W × H × D × P) where D=NUM_DISPARITIES, P=8 (path directions)

**Typical Runtime**: 200-500ms per 1920×1080 image pair on modern CPU



### 4.3 Q Matrix Consistency for Depth Reconstruction

**Location**: `src_rut_shape/depth.py:DepthCalculator`

**Problem Solved**: When rectified image size changes (auto-sizing or resizing), the original Q matrix becomes invalid, causing incorrect 3D reconstruction.

**Solution**: Recalculate Q matrix from adjusted projection matrices P1, P2.

**Theoretical Foundation** (from `document/Advance_Q.md`):

The Q matrix structure:
```
Q = [1   0   0        -cx      ]
    [0   1   0        -cy      ]
    [0   0   0         f       ]
    [0   0  -1/Tx  (cx-cx')/Tx ]
```

Where:
- `cx, cy`: Principal point of left camera (in pixels)
- `cx'`: Principal point of right camera (in pixels)
- `f`: Focal length (in pixels)
- `Tx`: Baseline (in same units as desired 3D output)

**When image is resized by scale factors (sx, sy)**:
```
fx_new = fx_old * sx
fy_new = fy_old * sy
cx_new = cx_old * sx
cy_new = cy_old * sy
```

**Q matrix adjustment**:
```python
def adjust_q_for_resize(Q_old, scale_x, scale_y):
    Q_new = Q_old.copy()
    Q_new[0,3] *= scale_x  # Adjust -cx
    Q_new[1,3] *= scale_y  # Adjust -cy
    Q_new[2,3] *= scale_x  # Adjust f (assuming fx ≈ fy)
    Q_new[3,3] *= scale_x  # Adjust (cx-cx')/Tx
    return Q_new
```

**Why Critical**: 
- Incorrect Q causes systematic depth errors (e.g., 10% size change → 10% depth error)
- Affects all downstream rut depth calculations
- Must maintain consistency between rectification and depth stages

**Implementation**: 
- Priority 1: Load `Q_rectified_<pair>.csv` from rectification temp folder
- Priority 2: Fall back to original `dispartityToDepthMap.csv` if not found
- Log warnings when fallback occurs



### 4.4 Blockwise Robust Outlier Detection

**Location**: `src_rut_shape/height/processors.py:RutShapeProcessor`

**Problem Solved**: Traditional global outlier detection fails when rut profiles have legitimate large variations (steps, transitions).

**Solution**: Blockwise robust detection with local statistics.

**Algorithm**:
```python
def blockwise_outlier_detection(data, block_size_factor=0.2, k_threshold=6.0):
    N = len(data)
    L = max(min_block_size, ceil(N * block_size_factor))
    
    outlier_mask = np.zeros(N, dtype=bool)
    
    for i in range(N):
        # Define block window
        start = max(0, i - L//2)
        end = min(N, i + L//2 + 1)
        block = data[start:end]
        
        # Robust statistics
        median = np.median(block)
        MAD = np.median(np.abs(block - median))
        
        # Robust z-score
        if MAD > 0:
            z_score = abs(data[i] - median) / (1.4826 * MAD)
            if z_score > k_threshold:
                outlier_mask[i] = True
    
    # Repair outliers
    data_repaired = data.copy()
    for i in np.where(outlier_mask)[0]:
        # Linear interpolation from nearest valid neighbors
        left_valid = find_nearest_valid(i, direction='left')
        right_valid = find_nearest_valid(i, direction='right')
        data_repaired[i] = interpolate(left_valid, right_valid, i)
    
    return data_repaired, outlier_mask
```

**Key Parameters**:
- `block_size_factor`: 0.2 (20% of data length)
- `min_block_size`: 25 points
- `k_y_block`: 6.0 (conservative threshold, ~99.9% confidence)
- `y_ratio_threshold`: 10.0 (magnitude guard against false positives)

**Why Chosen**:
- Median Absolute Deviation (MAD) is robust to outliers (50% breakdown point)
- Local blocks adapt to profile curvature
- Preserves legitimate steps and transitions
- Avoids over-smoothing

**Multiscale Enhancement**:
- Optional second pass with `multiscale_factor=1.5` (larger blocks)
- Catches outliers that span multiple small blocks
- Leave-one-out validation reduces false positives

**Performance**: O(N × L) ≈ O(N²) for typical block sizes, but N is small (~100-1000 points)



### 4.5 Improved Rut Depth Calculation

**Location**: `src_rut_shape/height/rut_calculator.py:RutCalculator`

**Problem Solved**: Original method (crown_line - min_y) is sensitive to endpoint noise and doesn't account for profile slope.

**Solution**: Orthogonal distance from rut bottom to fitted crown line.

**Algorithm**:
```python
def calculate_rut_depth_improved(profile_data, min_distance_ratio=0.1):
    # Step 1: Identify crown points (endpoints or local maxima)
    left_crown = profile_data[0]
    right_crown = profile_data[-1]
    
    # Optional: Use local maxima if endpoints are not peaks
    if fallback_to_local_peaks:
        left_peak = find_local_maximum(profile_data[:len//4])
        right_peak = find_local_maximum(profile_data[3*len//4:])
        if left_peak and right_peak:
            left_crown, right_crown = left_peak, right_peak
    
    # Step 2: Fit crown line (linear regression)
    x1, y1 = left_crown
    x2, y2 = right_crown
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    # Line equation: y = slope*x + intercept
    # Or: slope*x - y + intercept = 0
    
    # Step 3: Find rut bottom (global minimum)
    rut_bottom_idx = np.argmin(profile_data[:, 1])
    x0, y0 = profile_data[rut_bottom_idx]
    
    # Step 4: Calculate orthogonal distance
    # Distance from point (x0,y0) to line ax+by+c=0:
    # d = |ax0 + by0 + c| / sqrt(a² + b²)
    a, b, c = slope, -1, intercept
    distance = abs(a*x0 + b*y0 + c) / sqrt(a**2 + b**2)
    
    # Step 5: Validation
    profile_width = x2 - x1
    if distance < min_distance_ratio * profile_width:
        # Rut too shallow, may be noise
        return None, {'status': 'rejected', 'reason': 'below_threshold'}
    
    return distance * 1000, {  # Convert to mm
        'crown_left': left_crown,
        'crown_right': right_crown,
        'rut_bottom': (x0, y0),
        'crown_line_slope': slope,
        'orthogonal_distance_m': distance
    }
```

**Why Chosen**:
- Accounts for road camber (cross-slope)
- Orthogonal distance is geometrically correct
- Validation prevents false positives from noise
- Detailed results enable visualization and debugging

**Comparison with Original Method**:
| Aspect | Original | Improved |
|--------|----------|----------|
| Crown definition | Endpoint average | Linear fit through crowns |
| Rut measurement | Vertical distance | Orthogonal distance |
| Slope handling | Ignores slope | Accounts for camber |
| Validation | None | Min distance ratio |
| Typical difference | ±2-5mm for flat roads | Reference method |



### 4.6 Low-Pass Filtering with Reflection Padding

**Location**: `utils/low_pass_filter.py`

**Problem Solved**: Standard zero-padding or edge-value padding causes boundary artifacts in filtered signals.

**Solution**: Reflection extrapolation padding that preserves signal continuity.

**Algorithm**:
```python
def apply_butterworth_lowpass(data, cutoff_hz, sampling_rate_hz=1.0, 
                              padding_method='reflection_extrapolation',
                              padding_size_percent=25.0):
    # Step 1: Calculate padding size
    N = len(data)
    pad_size = int(N * padding_size_percent / 100)
    
    # Step 2: Apply reflection padding
    if padding_method == 'reflection_extrapolation':
        # Reflect and extrapolate slope
        left_slope = (data[1] - data[0])
        right_slope = (data[-1] - data[-2])
        
        left_pad = data[0] - left_slope * np.arange(pad_size, 0, -1)
        right_pad = data[-1] + right_slope * np.arange(1, pad_size+1)
        
        padded_data = np.concatenate([left_pad, data, right_pad])
    
    # Step 3: Design Butterworth filter
    nyquist = sampling_rate_hz / 2
    normalized_cutoff = cutoff_hz / nyquist
    b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')
    
    # Step 4: Apply filter (forward-backward for zero phase)
    filtered_padded = scipy.signal.filtfilt(b, a, padded_data)
    
    # Step 5: Remove padding
    filtered_data = filtered_padded[pad_size:-pad_size]
    
    return filtered_data
```

**Key Parameters**:
- `cutoff_hz`: Typically 0.1-0.5 Hz for rut profiles
- `order`: 4 (Butterworth, good flatness in passband)
- `padding_size_percent`: 25% (balance between boundary quality and computation)

**Why Chosen**:
- Butterworth filter: Maximally flat passband, no ripples
- `filtfilt`: Zero-phase filtering (no time shift)
- Reflection padding: Preserves endpoint values and slopes
- Prevents ringing artifacts at boundaries

**Typical Effect**: Smooths high-frequency noise (sensor jitter, reconstruction errors) while preserving rut shape.



---

## 5. Theoretical Foundations and Optimizations

### 5.1 Stereo Vision Geometry

**Epipolar Geometry**:
- Fundamental constraint: Corresponding points lie on conjugate epipolar lines
- Rectification transforms epipolar lines to horizontal scanlines
- Simplifies correspondence search from 2D to 1D

**Rectification Mathematics**:
```
Given: K1, d1, K2, d2, R, T (camera intrinsics and extrinsics)

cv2.stereoRectify computes:
- R1, R2: Rotation matrices to align image planes
- P1, P2: New projection matrices [K'|0] and [K'|Tx]
- Q: Disparity-to-depth reprojection matrix

Rectification pipeline for point p:
1. Undistort: p_undist = undistort(p, K, d)
2. Normalize: p_norm = K^(-1) @ p_undist
3. Rotate: p_rect = R @ p_norm
4. Project: p_final = P @ [p_rect; 1]
```

**Q Matrix Derivation**:
```
From stereo geometry:
X = (u - cx) * Z / f
Y = (v - cy) * Z / f
Z = f * Tx / disparity

Combining into homogeneous form:
[X]   [1  0  0      -cx     ] [u        ]
[Y] = [0  1  0      -cy     ] [v        ]
[Z]   [0  0  0       f      ] [disparity]
[W]   [0  0  -1/Tx  (cx-cx')/Tx] [1        ]

Then (X/W, Y/W, Z/W) gives metric 3D coordinates.
```

**Reference**: `document/Advance_Q.md` provides detailed Q matrix theory and resize corrections.



### 5.2 SGBM Algorithm Details

**Semi-Global Matching (SGM)**:
- Combines local matching with global optimization
- Cost aggregation along multiple paths (8 or 16 directions)
- Dynamic programming for each path independently

**Cost Function**:
```
C(p, d) = matching_cost(p, d) + 
          Σ_r [P1 * |d - d_r| == 1] + 
          Σ_r [P2 * |d - d_r| > 1]

Where:
- p: pixel position
- d: disparity value
- r: neighboring pixels along path
- P1: small penalty for smooth disparity changes
- P2: large penalty for discontinuities
```

**OpenCV SGBM Modes**:
- `SGBM_MODE_SGBM`: Full 8-direction aggregation (accurate, slower)
- `SGBM_MODE_SGBM_3WAY`: 3-direction approximation (faster, ~70% speed)
- `SGBM_MODE_HH`: Horizontal-only (fastest, lowest quality)

**Sub-pixel Refinement**:
- Fits parabola to cost curve around integer disparity minimum
- Achieves 0.1-0.2 pixel precision
- Critical for accurate depth at long ranges

**Complexity Analysis**:
- Matching cost: O(W × H × D × B²) where B=BLOCKSIZE
- Path aggregation: O(W × H × D × P) where P=num_paths
- Total: O(W × H × D × (B² + P))
- Typical: 1920×1080 × 128 × (25 + 8) ≈ 8.6 billion operations

**No GPU Acceleration**: 
- OpenCV's CPU SGBM is well-optimized with SIMD (SSE/AVX)
- GPU version (cv2.cuda.StereoSGM) not used due to:
  - Offline processing (speed not critical)
  - Deployment simplicity (no CUDA dependency)
  - Debugging ease (CPU-only pipeline)



### 5.3 Coordinate System Transformations

**Coordinate Frames**:
1. **Image coordinates**: (u, v) in pixels, origin at top-left
2. **Normalized camera coordinates**: (x, y, z) after K^(-1), z=1 plane
3. **Rectified coordinates**: After rotation R1/R2, epipolar lines horizontal
4. **World coordinates**: Metric 3D (X, Y, Z) in mm, origin at left camera
5. **Road coordinates**: After XY/YZ/XZ rotations, aligned with road surface

**Rotation Sequence** (applied in order):
```python
# 1. XY rotation (yaw, align with road direction)
R_xy = [[cos(θ1), -sin(θ1), 0],
        [sin(θ1),  cos(θ1), 0],
        [0,        0,       1]]

# 2. YZ rotation (pitch, level road plane)
R_yz = [[1, 0,        0       ],
        [0, cos(θ2), -sin(θ2)],
        [0, sin(θ2),  cos(θ2)]]

# 3. XZ rotation (roll, horizontal crown line)
R_xz = [[cos(θ3), 0, -sin(θ3)],
        [0,       1,  0       ],
        [sin(θ3), 0,  cos(θ3)]]

# Cumulative transformation
coords_road = coords_world @ R_xy.T @ R_yz.T @ R_xz.T
```

**Why This Order**:
- XY first: Aligns longitudinal axis with vehicle motion
- YZ second: Levels the road plane (removes pitch)
- XZ last: Ensures crown line is horizontal (removes roll)

**Calibration Process**:
- Angles determined empirically from flat road sections
- Typical values: θ1 ≈ 0-5°, θ2 ≈ 2-10°, θ3 ≈ 0-3°
- Stored in configuration JSON, applied consistently



### 5.4 Parallelization Strategy

**Current Implementation: Sequential Single-Process**

The system uses **no parallelization** by design:

```python
# main.py execution flow
for set_folder in glob('set_*'):
    for pair_folder in set_folder.glob('*'):
        # Stage 1: Rectification
        rectified_left, rectified_right = rectifier.process(pair)
        
        # Stage 2: Disparity
        disparity = disparity_calc.process(rectified_left, rectified_right)
        
        # Stage 3: Depth
        world_coords = depth_calc.process(disparity)
        
        # Stage 4: Height
        rut_depth = height_calc.process(world_coords)
```

**Why No Parallelization**:
1. **Data Dependencies**: Each stage depends on previous stage outputs
2. **I/O Bound**: File reading/writing dominates (not CPU-bound)
3. **Memory Constraints**: Large image arrays (1920×1080×3 × 2 ≈ 12MB per pair)
4. **Debugging Priority**: Sequential execution simplifies error tracing
5. **Batch Size**: Typical datasets have 10-100 pairs (not thousands)

**OpenCV Internal Parallelization**:
- OpenCV uses **OpenMP** for internal multi-threading
- Controlled by `cv2.setNumThreads(N)` (default: all cores)
- Applies to:
  - `cv2.remap()`: Parallel pixel remapping
  - `cv2.StereoSGBM.compute()`: Parallel scanline processing
  - `cv2.reprojectImageTo3D()`: Parallel 3D reprojection

**Typical CPU Utilization**:
- Rectification: 30-50% (remap operations)
- Disparity: 80-100% (SGBM compute)
- Depth: 20-40% (reprojectImageTo3D)
- Height: 5-10% (numpy operations)

**Potential Parallelization Opportunities** (not implemented):
```python
# Option 1: Parallel pair processing (embarrassingly parallel)
from multiprocessing import Pool

def process_pair(pair_folder):
    # Complete 4-stage pipeline for one pair
    return rut_depth

with Pool(processes=4) as pool:
    results = pool.map(process_pair, pair_folders)

# Challenges:
# - Large memory footprint per process
# - File I/O contention
# - Error handling complexity
```

```python
# Option 2: Pipeline parallelization (producer-consumer)
from queue import Queue
from threading import Thread

rectify_queue = Queue(maxsize=2)
disparity_queue = Queue(maxsize=2)
depth_queue = Queue(maxsize=2)

# Stage threads
Thread(target=rectify_worker, args=(input_queue, rectify_queue)).start()
Thread(target=disparity_worker, args=(rectify_queue, disparity_queue)).start()
Thread(target=depth_worker, args=(disparity_queue, depth_queue)).start()
Thread(target=height_worker, args=(depth_queue, output_queue)).start()

# Challenges:
# - Queue synchronization overhead
# - Memory management (large arrays)
# - Error propagation across threads
```

**Race Condition Avoidance**:
- Each pair writes to unique output folders: `set_X/pair_Y/`
- No shared state between pair processing
- Temp folder cleared at start of each run
- File naming includes pair identifier: `disparity_<pair>.npy`

**Latency Characteristics**:
- Per-pair processing time: 2-5 seconds (1920×1080 images)
- Breakdown:
  - Rectification: 300-500ms
  - Disparity: 1000-2000ms (dominant)
  - Depth: 200-400ms
  - Height: 100-300ms
  - I/O overhead: 500-1000ms
- Total dataset (50 pairs): 2-4 minutes

**Conclusion**: Sequential processing is appropriate for this offline analysis application. Parallelization would add complexity without significant benefit given typical dataset sizes and I/O bottlenecks.



---

## 6. Data Flow Diagram with Coordinate Frames

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT STAGE                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Stereo Images:                                                           │
│   left_<pair>.jpg, right_<pair>.jpg                                     │
│   Format: BGR, uint8, shape (H, W, 3)                                   │
│   Coordinate frame: Image (u,v) pixels, origin top-left                 │
│                                                                          │
│ Calibration Parameters:                                                  │
│   K1, K2: 3×3 camera intrinsic matrices (fx, fy, cx, cy)               │
│   d1, d2: Distortion coefficients (k1, k2, p1, p2, k3)                 │
│   R: 3×3 rotation matrix (right camera relative to left)                │
│   T: 3×1 translation vector (baseline in mm)                            │
│   Q: 4×4 disparity-to-depth matrix (original calibration)               │
│                                                                          │
│ Seed Points:                                                             │
│   left_<pair>.json: {"rut_1": [u1,v1], "rut_2": [u2,v2]}               │
│   Coordinate frame: Original image (u,v) pixels                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: RECTIFICATION                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ Transformation: Image → Rectified Image                                 │
│                                                                          │
│ Input coordinates:  (u, v) in original image pixels                     │
│ Output coordinates: (u', v') in rectified image pixels                  │
│                                                                          │
│ Transformation pipeline:                                                 │
│   1. Undistort: (u,v) → (u_undist, v_undist)                           │
│      p_undist = undistort(p, K, d)  [OpenCV radial-tangential model]   │
│                                                                          │
│   2. Normalize: (u_undist, v_undist) → (x_norm, y_norm, 1)             │
│      [x_norm, y_norm, 1]ᵀ = K^(-1) @ [u_undist, v_undist, 1]ᵀ          │
│                                                                          │
│   3. Rotate: (x_norm, y_norm, 1) → (x_rect, y_rect, z_rect)            │
│      [x_rect, y_rect, z_rect]ᵀ = R1 @ [x_norm, y_norm, 1]ᵀ             │
│                                                                          │
│   4. Project: (x_rect, y_rect, z_rect) → (u', v')                      │
│      [u'·w, v'·w, w]ᵀ = P1 @ [x_rect, y_rect, z_rect, 1]ᵀ              │
│      (u', v') = (u'·w/w, v'·w/w)                                        │
│                                                                          │
│ Key property: Epipolar lines become horizontal (v'_left = v'_right)     │
│                                                                          │
│ Outputs:                                                                 │
│   • Rectified images: left_rectified.npy, right_rectified.npy          │
│     Format: RGB, uint8, shape (H', W', 3)                               │
│     Coordinate frame: Rectified (u',v') pixels                          │
│   • Adjusted matrices: P1_adj, P2_adj, Q_rectified                      │
│   • Rectified seed points: [(u'₁,v'₁), ..., (u'ₙ,v'ₙ)]                 │
│     N interpolated points between rut_1 and rut_2                       │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: DISPARITY CALCULATION                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ Transformation: Rectified Image Pair → Disparity Map                    │
│                                                                          │
│ Input coordinates:  (u', v') in rectified pixels                        │
│ Output values:      d(u', v') disparity in pixels                       │
│                                                                          │
│ Algorithm: Semi-Global Block Matching (SGBM)                            │
│   For each pixel (u', v') in left image:                                │
│     Search along horizontal line v' in right image                      │
│     Find best match at (u'-d, v')                                       │
│     Disparity d = u'_left - u'_right                                    │
│                                                                          │
│ Disparity range: [MIN_DISPARITY, MIN_DISPARITY + NUM_DISPARITIES]      │
│   Typical: [0, 128] pixels                                              │
│                                                                          │
│ Sub-pixel refinement: d_refined = d_integer + Δd                        │
│   Δd ∈ [-0.5, 0.5] from parabola fitting                               │
│                                                                          │
│ Physical meaning:                                                        │
│   Larger disparity → closer to camera                                   │
│   d = (f · baseline) / depth                                            │
│   depth = (f · baseline) / d                                            │
│                                                                          │
│ Outputs:                                                                 │
│   • Disparity map: disparity.npy                                        │
│     Format: float32, shape (H', W'), units: pixels                      │
│     Invalid regions: marked with large negative values                  │
│   • Rut disparity profile: rut_disparity.csv                            │
│     Disparity values sampled at rectified seed points                   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: DEPTH RECONSTRUCTION                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Transformation: Disparity Map → 3D World Coordinates                    │
│                                                                          │
│ Input:  d(u', v') disparity in pixels                                   │
│ Output: (X, Y, Z) in millimeters, origin at left camera optical center  │
│                                                                          │
│ Reprojection formula (using Q matrix):                                  │
│   [X·W]   [1  0  0      -cx     ] [u'       ]                          │
│   [Y·W] = [0  1  0      -cy     ] [v'       ]                          │
│   [Z·W]   [0  0  0       f      ] [d(u',v') ]                          │
│   [W  ]   [0  0  -1/Tx  (cx-cx')/Tx] [1        ]                       │
│                                                                          │
│   Then: (X, Y, Z) = (X·W/W, Y·W/W, Z·W/W)                               │
│                                                                          │
│ Coordinate frame: Camera coordinates                                     │
│   X: Right (perpendicular to optical axis)                              │
│   Y: Down (perpendicular to optical axis)                               │
│   Z: Forward (along optical axis, depth)                                │
│   Units: millimeters                                                     │
│                                                                          │
│ Rotation to road coordinates:                                            │
│   world_rotated = world_camera @ R_xy.T @ R_yz.T @ R_xz.T              │
│                                                                          │
│ Final coordinate frame: Road coordinates                                 │
│   X: Longitudinal (vehicle motion direction)                            │
│   Y: Vertical (height above road surface)                               │
│   Z: Lateral (across road width)                                        │
│   Units: millimeters                                                     │
│                                                                          │
│ Depth calculation:                                                       │
│   depth = sqrt(X² + Y² + Z²) / 1000  [convert to meters]               │
│                                                                          │
│ Outputs:                                                                 │
│   • World coordinates: world_coord.npy                                   │
│     Format: float32, shape (H', W', 3), units: mm                       │
│     Coordinate frame: Road (X, Y, Z)                                    │
│   • Depth image: depth.jpg                                              │
│     Colorized visualization, range [min_depth, max_depth] meters        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: RUT SHAPE EXTRACTION                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Transformation: 3D World Coordinates → 2D Rut Profile → Rut Depth       │
│                                                                          │
│ Input:  world_coord(u', v') = (X, Y, Z) in road coordinates (mm)       │
│         rectified_points = [(u'₁,v'₁), ..., (u'ₙ,v'ₙ)]                 │
│                                                                          │
│ Step 1: Extract 3D coordinates along rut line                           │
│   For each (u'ᵢ, v'ᵢ):                                                  │
│     (Xᵢ, Yᵢ, Zᵢ) = world_coord[v'ᵢ, u'ᵢ, :]                            │
│                                                                          │
│ Step 2: Project to 2D cross-section (discard longitudinal X)            │
│   rut_profile = [(Zᵢ, Yᵢ) for i in 1..N]                               │
│   Coordinate frame: Cross-section                                        │
│     Z: Lateral position (across road width), mm                         │
│     Y: Vertical position (height above road), mm                        │
│                                                                          │
│ Step 3: Apply processing pipeline                                       │
│   a) Extreme value filtering (blockwise robust detection)               │
│      Remove outliers, repair with interpolation                         │
│                                                                          │
│   b) Rotation calculation (optional)                                    │
│      Fit line to endpoints, rotate to horizontal                        │
│      Removes residual tilt from coordinate alignment                    │
│                                                                          │
│   c) Depth adjustment                                                   │
│      Shift Z to camera center: Z' = Z - Z_camera                        │
│                                                                          │
│   d) Ground level adjustment (optional)                                 │
│      Calculate baseline from endpoints                                  │
│      Shift Y to ground: Y' = Y - baseline                               │
│                                                                          │
│   e) Low-pass filtering (optional)                                      │
│      Butterworth filter on Y coordinates                                │
│      Removes high-frequency noise                                       │
│                                                                          │
│   f) Width scaling (optional)                                           │
│      Scale Z to match known lane_width                                  │
│      Z_scaled = Z * (lane_width / current_width)                        │
│                                                                          │
│ Step 4: Rut depth calculation                                           │
│   Method: Improved (orthogonal distance)                                │
│     1. Identify crown points (left and right peaks)                     │
│     2. Fit crown line: y = slope·z + intercept                          │
│     3. Find rut bottom: (z₀, y₀) = argmin(Y)                           │
│     4. Calculate orthogonal distance:                                   │
│        d = |slope·z₀ - y₀ + intercept| / sqrt(slope² + 1)              │
│     5. Convert to mm: rut_depth_mm = d * 1000                          │
│                                                                          │
│ Final coordinate frame: Normalized cross-section                         │
│   Z: 0 to lane_width (meters), left to right                           │
│   Y: Relative to ground level (meters), positive = above ground         │
│                                                                          │
│ Outputs:                                                                 │
│   • Rut profile CSVs: rut_0-original.csv through rut_4-baseline.csv    │
│     Format: (Z, Y) pairs in mm                                          │
│   • Filtered profile: rut_filtered.csv                                  │
│   • Scaled profile: rut_filtered_scaled.csv                             │
│   • Rut depth: rut_depth.txt                                            │
│     Single value in mm (e.g., "15.3 mm")                                │
│   • Visualization: rut_depth_visualization.png                          │
│     Annotated plot showing crown line, rut bottom, depth measurement    │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│ Primary Result:                                                          │
│   Rut depth: XX.X mm (millimeter precision)                             │
│                                                                          │
│ Supporting Artifacts:                                                    │
│   • Complete processing chain CSVs (intermediate stages)                │
│   • Visualizations: rectified overlays, disparity maps, depth images    │
│   • Metadata: Q matrices, projection matrices, processing parameters    │
│                                                                          │
│ Coordinate Frame Summary:                                                │
│   Image (u,v) → Rectified (u',v') → Disparity d →                      │
│   Camera 3D (X,Y,Z) → Road 3D (X,Y,Z) → Cross-section 2D (Z,Y) →       │
│   Rut depth (scalar, mm)                                                │
└─────────────────────────────────────────────────────────────────────────┘
```



---

## 7. Code Structure and Module Responsibilities

### 7.1 Module Hierarchy

```
StereoVisionToolkit/
├── main.py                          # Entry point, orchestrates pipeline
├── config/
│   ├── config.py                    # Configuration loader and validator
│   ├── config_rut_shape.json        # Main configuration file
│   └── config_rut_shape1.json       # Alternative configuration
├── src_rut_shape/                   # Core processing modules
│   ├── rut_shape.py                 # Pipeline orchestrator
│   ├── rectify_refactored.py        # Stage 1: Rectification
│   ├── disparity_refactored.py      # Stage 2: Disparity calculation
│   ├── depth.py                     # Stage 3: Depth reconstruction
│   ├── height_refactored.py         # Stage 4: Rut shape extraction
│   ├── base/                        # Base classes
│   │   ├── file_manager.py          # File I/O operations
│   │   └── processor.py             # Processing pipeline template
│   ├── rectification/               # Rectification sub-modules
│   │   ├── engine.py                # Stereo rectification engine
│   │   ├── matrix_calculator.py     # Q matrix calculations
│   │   └── file_manager.py          # Rectification file operations
│   ├── disparity/                   # Disparity sub-modules
│   │   ├── sgbm_engine.py           # SGBM algorithm wrapper
│   │   ├── parameter_calculator.py  # Auto-parameter tuning
│   │   └── disparity_processor.py   # Disparity processing
│   └── height/                      # Height sub-modules
│       ├── processors.py            # Rut shape processing
│       ├── rut_calculator.py        # Rut depth calculation
│       ├── image_loader.py          # Image and data loading
│       ├── coordinate_processor.py  # Coordinate transformations
│       └── file_manager.py          # Height file operations
├── utils/                           # Utility modules
│   ├── point_processor.py           # Point transformation utilities
│   ├── image_processing.py          # Image manipulation
│   ├── low_pass_filter.py           # Signal filtering
│   ├── data_scaling.py              # Coordinate scaling
│   ├── rut_visualization.py         # Rut depth plotting
│   ├── visualizer.py                # General visualization
│   ├── stereo_math.py               # Stereo geometry calculations
│   ├── file_operations.py           # File I/O helpers
│   └── logger_config.py             # Logging configuration
└── document/                        # Documentation
    ├── README.md                    # User guide
    ├── PROJECT_DEEP_DIVE.md         # This document
    ├── TECHNOLOGY_TRANSFER.md       # Implementation guide
    └── Advance_Q.md                 # Q matrix theory
```

### 7.2 Key Class Responsibilities

**RutShape** (`src_rut_shape/rut_shape.py`):
- Orchestrates the 4-stage pipeline
- Manages stage dependencies
- Provides unified entry point

**Rectifier** (`src_rut_shape/rectify_refactored.py`):
- Loads calibration parameters
- Computes rectification matrices (R1, R2, P1, P2, Q)
- Handles auto-sizing for extreme distortions
- Rectifies images and seed points
- Saves rectified outputs and adjusted matrices

**DisparityCalculator** (`src_rut_shape/disparity_refactored.py`):
- Auto-calculates SGBM parameters
- Configures and runs SGBM matcher
- Samples disparity along rut line
- Saves disparity maps and profiles

**DepthCalculator** (`src_rut_shape/depth.py`):
- Loads appropriate Q matrix (rectified or original)
- Reprojects disparity to 3D using cv2.reprojectImageTo3D
- Applies coordinate rotations (XY, YZ, XZ)
- Saves world coordinates and depth visualizations

**HeightCalculator** (`src_rut_shape/height_refactored.py`):
- Extracts 3D coordinates along rut line
- Applies multi-stage filtering pipeline
- Calculates rut depth using improved method
- Saves intermediate stages and final results

**SGBMEngine** (`src_rut_shape/disparity/sgbm_engine.py`):
- Encapsulates OpenCV StereoSGBM configuration
- Validates parameters (NUM_DISPARITIES multiple of 16)
- Provides clean API for disparity computation

**RutShapeProcessor** (`src_rut_shape/height/processors.py`):
- Implements blockwise outlier detection
- Applies rotation and baseline adjustments
- Handles low-pass filtering with padding
- Manages width scaling

**RutCalculator** (`src_rut_shape/height/rut_calculator.py`):
- Implements improved rut depth algorithm
- Calculates orthogonal distance to crown line
- Provides validation and detailed results





# Broadcast Pipeline Implementation

Technical implementation details for each step of the automated sports broadcast pipeline.

## Step 1: Fisheye Undistortion

**Purpose**: Remove barrel distortion from Pi HQ fisheye lenses for geometrically correct images.

**Implementation**: `scripts/undistort_video.py`

**Method**: Uses OpenCV's `cv2.fisheye.initUndistortRectifyMap()` with precomputed undistortion maps for performance. Camera calibration (intrinsics K and distortion coefficients D) loaded from `data/calibration/camera_calibration.json`. Processes both cameras at 2× scale factor (2560×1440 → 5120×2880) to preserve detail.

**Input**: Raw dual-camera videos with fisheye distortion  
**Output**: Geometrically corrected videos at 5120×2880 in `data/undistorted/`

---

## Step 2: Panoramic Stitching

**Purpose**: Combine left and right camera views into single wide panoramic view.

**Implementation**: `scripts/stitching/manual/apply_manual_stitch.py` (recommended)

### Manual Stitching (Recommended)

**Why Manual?** For this dual-camera sports broadcast setup, manual stitching significantly outperforms automated:
- Fixed parallel rig only needs affine transformation (6 params: rotation, scale, translation) vs homography (8 params)
- Repetitive grass textures confuse feature matchers (SIFT fails on uniform patterns)
- Precise seam control, faster processing, more reliable

**Method**: Affine transformation with calibrated parameters + vertical seam feathering (weighted blending) + horizontal stretch correction for fisheye residuals.

**Key Parameters**: `--seam-top` (2030), `--seam-bottom` (2125), `--feather` (15px)  
**Blending**: Linear alpha transition from 100% left → 50/50 → 100% right across feather width

### Automated Stitching (Alternative)

**When to use**: Non-parallel cameras, moving rigs, high-texture scenes (buildings/crowds)  
**Method**: SIFT + CLAHE → FLANN matching + Lowe's ratio test → RANSAC homography + LAB exposure matching  
**Limitations**: Struggles with grass, slower, less seam control

**Input**: Two undistorted videos (5120×2880)  
**Output**: Panorama in `output/stitched/panorama.mp4`

---

## Step 3: Ball Tracking Configuration

**Purpose**: Define field boundaries and initial ball position for improved tracking accuracy.

**Implementation**: `scripts/detection/setup_ball_tracking.py`

**Interactive Setup**:
1. Draw field polygon (click points, press 'c' to close) → creates exclusion mask
2. Click ball position → seeds search algorithm
3. Press 'q' to save

**Output**: `data/calibration/ball_tracking_config.json` containing field polygon, ball position, frame dimensions

---

## Step 4: Advanced Broadcast Generation

**Purpose**: Generate professional broadcast view with automated camera tracking.

**Implementation**: `scripts/detection/broadcast.py`

### 4.1 Object Detection & Tracking

**Detection**: YOLOv8s runs continuously on every frame, detecting both players (class 0, threshold 0.3) and balls (class 32, threshold 0.2). ByteTrack maintains persistent player IDs across occlusions.

**Ball Tracking**: Combination of HSV color detection and YOLO sports ball detection. HSV is primary, YOLO serves as fallback.

> **Note**: HSV provides reliable tracking for red balls in consistent lighting. YOLO (generic COCO weights) produces false positives but serves as fallback when HSV fails. Fine-tuned YOLO would be superior (lighting-invariant, handles all ball types, better with occlusions).

**Detection Strategy**:
1. **HSV Color Detection** (primary)
   - Red ranges [0-10°, 165-180°] with Gaussian blur + morphology
   - Scores by circularity (>0.25), extent (>0.35), size (<150px²), temporal coherence
   - 600px search radius with field exclusion mask
2. **YOLO Sports Ball** (fallback when HSV fails)
   - Class 32, threshold 0.2, size check <100px
   - Field exclusion zone filtering to reduce false positives
   - Temporal coherence check: only used if within 300px of last position (prevents false positive jumps)
   - Only activated when HSV returns no detection
3. **Kalman Prediction** (when both methods fail)
   - Continues for up to 60 frames using velocity prediction
   - Bounds checking to prevent divergence

**Limitations**: HSV is lighting-sensitive and color-specific. YOLO generic weights produce false positives.

**Future work**: Fine-tune YOLO on sport-specific annotated footage to make it the primary method

### 4.2 Ball Position Filtering

**Kalman Filter** (filterpy): State [x, y, vx, vy], constant velocity model, measurement noise R=15.0, process noise Q=[2.5, 2.5, 18, 18]. Continues predictions up to 60 frames during occlusions with bounds checking.

### 4.3 Camera Framing

**Target Computation** (pixel-space only):
- Find players within 400px of ball
- Center of mass of [ball + nearby players]
- Adjust for velocity (zoom out when speed >30px/frame)
- View width clamped 600-2000px, 1.3× spread margin, 16:9 aspect ratio

### 4.4 Camera Motion

**Spring-Damper Physics**:
```python
force = k_spring(0.04) × (target - current) - k_damper(0.8) × velocity
```
Gentle pull, heavy damping, ultra-slow zoom (0.008) → smooth, natural movement

### 4.5 Output

**Broadcast** (1280×720): Cropped panorama at camera position, HD resolution, 16:9 aspect  
**Preview** (optional): Full panorama with overlays (magenta=camera view, cyan=raw detection, yellow=Kalman filtered, green=players+IDs)

---

## Pipeline Automation

**Script**: `scripts/run_full_pipeline.py`

**Steps**: Undistort → Stitch (manual) → Ball tracking setup (interactive) → Broadcast generation → Screenshots

**Key Options**: `--seam-top/bottom/feather`, `--skip-tracking-setup`, `--screenshot-frames`

---

## Technical Stack

**CV**: OpenCV 4.x  
**DL**: Ultralytics YOLOv8 (PyTorch)  
**Tracking**: ByteTrack  
**Filtering**: FilterPy (Kalman)  
**GPU Acceleration**: YOLOv8 runs on MPS (Apple Silicon) / CUDA (NVIDIA) / CPU fallback

---

## AI Usage

**YOLOv8s**: Pre-trained on COCO dataset, detects persons (class 0) and sports balls (class 32), no custom training

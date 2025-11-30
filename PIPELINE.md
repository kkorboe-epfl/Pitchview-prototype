# Broadcast Pipeline Implementation

This document explains how each step of the automated sports broadcast pipeline is implemented.

## Step 1: Fisheye Undistortion

**Purpose**: Remove barrel distortion from Pi HQ camera fisheye lenses to create geometrically correct images.

**Implementation**: `scripts/undistort_video.py`

**Method**:
- Uses OpenCV's `cv2.fisheye.undistortImage()` with pre-calibrated camera matrices
- Calibration parameters stored in `data/calibration/custom_calibration.json`
- Contains camera intrinsics (focal length, principal point) and distortion coefficients
- Processes both left and right camera feeds frame-by-frame

**Input**: Raw dual-camera videos with fisheye distortion  
**Output**: Geometrically corrected videos in `data/undistorted/`

**Technical Details**:
- Camera matrix K defines intrinsic parameters
- Distortion coefficients model radial and tangential distortion
- Balance parameter controls field-of-view vs. black borders trade-off

---

## Step 2: Panoramic Stitching

**Purpose**: Combine left and right camera views into a single wide panoramic view.

**Implementation**: `scripts/stitching/apply_manual_stitch.py`

**Method**:
- Concatenates left and right frames horizontally
- Applies vertical seam blending at the overlap region
- Uses feathering (weighted blending) to create smooth transition
- Crops black borders for clean output

**Key Parameters**:
- `--seam-top`: Y-coordinate where blending starts (default: 2030)
- `--seam-bottom`: Y-coordinate where blending ends (default: 2125)
- `--feather`: Blending width in pixels (default: 15)

**Blending Algorithm**:
1. At seam center: 50% left frame, 50% right frame
2. At edges: Gradual alpha transition using linear interpolation
3. Formula: `alpha = (x - seam_left) / feather_width`

**Input**: Two undistorted videos  
**Output**: Single panoramic video (typically 3800×500) in `output/stitched/`

---

## Step 3: Ball Tracking Configuration

**Purpose**: Define field boundaries and initial ball position to improve tracking accuracy.

**Implementation**: `scripts/detection/setup_ball_tracking.py`

**Method**: Interactive OpenCV window with mouse callbacks

**User Actions**:
1. **Draw field polygon**: Click to add points, press 'c' to close
2. **Mark ball position**: Single click on the ball
3. **Save**: Press 'q' to write configuration

**Output**: `data/calibration/ball_tracking_config.json`
```json
{
  "field_polygon": [[x1,y1], [x2,y2], ...],
  "ball_position": [x, y],
  "frame_dimensions": [width, height]
}
```

**Purpose of Data**:
- Field polygon creates exclusion mask (ignores corner flags, out-of-bounds markers)
- Initial ball position seeds the search algorithm for first detection

---

## Step 4: Advanced Broadcast Generation

**Purpose**: Generate professional broadcast view with automated camera tracking.

**Implementation**: `scripts/detection/broadcast.py`

### 4.1 Object Detection & Tracking

**Player Detection**: YOLOv8 + ByteTrack
- Model: `yolov8s.pt` (small, balanced speed/accuracy)
- Detects class 0 (person) with confidence threshold 0.3
- ByteTrack maintains persistent player IDs across frames
- Prevents ID switches during occlusions

**Ball Detection**: Dual-method approach
1. **Primary: HSV Color Detection**
   - Red color ranges in HSV space: [0-10°, 165-180°]
   - Applies Gaussian blur and morphological operations
   - Scores candidates by size, circularity, and proximity to last position
   - Uses exclusion mask from field polygon

2. **Secondary: YOLO Sports Ball**
   - Class 32 with threshold 0.2
   - Less reliable but provides fallback

### 4.2 Ball Position Filtering

**Kalman Filter** (filterpy library)
- **State vector**: [x, y, vx, vy] in pixel coordinates
- **State transition**: Constant velocity model with dt=1 frame
- **Measurement noise** (R): 15.0 (trusts detections)
- **Process noise** (Q): [2.5, 2.5, 18, 18] (higher for velocity to track acceleration)

**Prediction Continuation**:
- Continues tracking up to 60 frames without detection
- Uses Kalman predictions during occlusions
- Bounds checking prevents divergence (clips to frame + 200px margin)
- Resets filter after extended loss

### 4.3 Camera Framing Logic

**Target Computation** (pixel-space only):
1. Find players within 400px of ball
2. Compute center of mass of [ball + nearby players]
3. Calculate bounding box spread
4. Adjust for ball velocity (zoom out when speed > 30px/frame)
5. Clamp view width: 600-2000 pixels

**Smart Framing**:
- No homography needed (pure pixel calculations)
- View margin: 1.3× the spread of targets
- Aspect ratio: 16:9 maintained throughout

### 4.4 Camera Motion

**Spring-Damper Physics System**
```python
spring_force = k_spring × (target - current)
damping_force = k_damper × velocity
acceleration = spring_force - damping_force
```

**Parameters**:
- Spring constant: 0.04 (gentle pull toward target)
- Damper constant: 0.8 (heavy resistance to motion)
- Zoom smoothing: 0.008 (ultra-slow transitions)

**Result**: Smooth, natural camera movement without jarring transitions

### 4.5 Output Generation

**Broadcast View** (1280×720):
- Crops panorama based on camera position
- Resizes to HD resolution
- Maintains 16:9 aspect ratio

**Preview Video** (optional):
- Full panorama with overlays
- Green box: Camera view rectangle
- Cyan circle: Raw ball detection
- Blue circle: Kalman-filtered position
- Red boxes: Player tracking boxes with IDs
- Magenta line: Ball velocity vector

---

## Technical Stack

**Computer Vision**: OpenCV 4.x
**Deep Learning**: Ultralytics YOLOv8 with PyTorch
**Tracking**: ByteTrack (built into YOLOv8)
**Filtering**: FilterPy (Kalman filter implementation)
**Hardware Acceleration**: MPS (Apple Silicon) / CUDA (NVIDIA) / CPU fallback

---

## AI Usage in This Project

This section documents where and how AI technologies were employed in the development of this broadcast system.

### Pre-trained AI Models

**YOLOv8 (You Only Look Once v8)**
- **Purpose**: Real-time object detection for players and ball
- **Source**: Ultralytics pre-trained model (yolov8s.pt)
- **Training**: Trained on COCO dataset (Common Objects in Context)
- **Usage**: 
  - Detects persons (class 0) for player tracking
  - Detects sports balls (class 32) as secondary ball detection method
- **No custom training performed** - uses off-the-shelf weights

**ByteTrack**
- **Purpose**: Multi-object tracking algorithm for persistent player IDs
- **Source**: Integrated into Ultralytics YOLOv8 framework
- **Usage**: Maintains player identity across frames during occlusions
- **Implementation**: Algorithm-based, not a trainable neural network
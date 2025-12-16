# PitchView Prototype

Affordable automated sports broadcast system using dual Pi HQ cameras. Stitches two camera feeds into a panoramic view, then uses computer vision with ByteTrack multi-object tracking, Kalman filtering, and physics-based camera motion to generate professional broadcast footage.

**Hardware**: Single Raspberry Pi with two Pi HQ camera modules, providing wide-angle coverage at a fraction of traditional broadcast camera costs.

## Features

- **Panoramic Stitching** from dual-camera feeds with smooth edge blending and auto-crop
- **Broadcast System** with YOLOv8s running continuously to detect both players and balls, ByteTrack multi-object tracking, Kalman filtering, and spring-damper physics-based camera motion
- **Ball Tracking**: Combination of HSV color detection and YOLO. HSV is primary (works well for red balls in consistent lighting), YOLO serves as fallback when HSV fails with temporal coherence check to prevent false positive jumps. Both methods use exclusion zones to filter out-of-bounds detections. For production use, fine-tuning YOLO on annotated footage would provide superior accuracy across all conditions
- **Output**: 720p HD broadcast view with professional camera movement

For detailed technical implementation, see [PIPELINE.md](PIPELINE.md).

## Prerequisites (One-Time Setup)

Before running the automated pipeline, you need to complete these one-time calibration steps:

### 1. Camera Calibration (Required)
**File:** `data/calibration/camera_calibration.json`

Calibrate your fisheye cameras using OpenCV's checkerboard calibration. This determines distortion coefficients for each camera.

**How to create:** Follow [this tutorial](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0) to capture checkerboard images and generate calibration parameters.

**Note:** A sample calibration file is provided for the prototype Pi HQ cameras. Replace it with your own camera calibration.

### 2. Manual Stitch Calibration (Required)
**File:** `data/calibration/manual_stitch_calibration.json`

Manually align left and right camera views by adjusting position, rotation, and scale.

**How to create:**
```bash
python3 scripts/stitching/manual/calibrate_manual_stitch.py \
  --left data/undistorted/left.mp4 \
  --right data/undistorted/right.mp4
```

Interactive GUI with keyboard controls to align the cameras perfectly. Press 'S' to save.

**Note:** Manual stitching is recommended for this dual-camera sports broadcast setup. The automated approach (`scripts/stitching/auto/`) uses SIFT feature detection and homography estimation, which struggles with repetitive grass textures and is overkill for our parallel camera configuration. Manual stitching with simple affine transformation provides better results and faster processing for this specific use case.

---

## Quick Start

### Option 1: Full Pipeline (Automated)

**Prerequisites:** Complete all one-time setup steps above first.

Run the entire pipeline in one command - undistorts, stitches, and generates broadcast view with screenshots at each step:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python3 scripts/run_full_pipeline.py \
  --left-raw data/raw/leftflip.mp4 \
  --right-raw data/raw/rightflip.mp4 \
  --output-dir output/pipeline

This will automatically:
- Undistort both camera videos
- Stitch them into a panorama
- Set up ball tracking configuration (interactive)
- Generate broadcast view with tracking
- Save screenshots at each step for debugging

All outputs will be in `output/pipeline/` with screenshots in `output/pipeline/screenshots/`.

For technical details on each step, see [PIPELINE.md](PIPELINE.md).
```

### Option 2: Step-by-Step

### macOS / Linux

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download sample videos
cd data
./download_videos.sh
cd ..

# 4. Undistort fisheye videos
python3 scripts/undistort_video.py

# Optional: Customize paths or focal scale
# python3 scripts/undistort_video.py \
#   --left-input data/raw/leftflip.mp4 \
#   --right-input data/raw/rightflip.mp4 \
#   --left-output data/undistorted/left.mp4 \
#   --right-output data/undistorted/right.mp4 \
#   --focal-scale 0.6

# 5. Stitch dual-camera videos into panorama
python3 scripts/stitching/manual/apply_manual_stitch.py --seam-top 2030 --seam-bottom 2125 --feather 15  

# 6. Configure ball tracking (one-time setup)
python3 scripts/detection/setup_ball_tracking.py output/stitched/panorama.mp4

# 7. Generate broadcast view with tracking
python3 scripts/detection/broadcast.py \
  --video output/stitched/panorama.mp4 \
  --save-broadcast output/broadcast/game.mp4 \
  --save-preview output/broadcast/preview.mp4
```

### Windows (PowerShell)

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# data\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download sample videos
cd data
& "C:\Program Files\Git\bin\bash.exe" "./download_videos.sh"
cd ..

# 4. Undistort fisheye videos
python scripts/undistort_video.py

# Optional: Customize paths or focal scale
# python scripts/undistort_video.py `
#   --left-input data/raw/leftflip.mp4 `
#   --right-input data/raw/rightflip.mp4 `
#   --left-output data/undistorted/left.mp4 `
#   --right-output data/undistorted/right.mp4 `
#   --focal-scale 0.6

# 5. Stitch dual-camera videos into panorama
python scripts/stitching/manual/apply_manual_stitch.py --seam-top 2030 --seam-bottom 2125 --feather 15  

# 6. Configure ball tracking (one-time setup)
python scripts/detection/setup_ball_tracking.py output/stitched/panorama.mp4

# 7. Generate broadcast view with advanced tracking
python scripts/detection/broadcast.py `
  --video output/stitched/panorama.mp4 `
  --save-broadcast output/broadcast/game.mp4 `
  --save-preview output/broadcast/preview.mp4
```

## Directory Structure

```
pitchview-prototype/
├── data/
│   ├── raw/              # Input videos
│   ├── undistorted/      # Undistorted videos (auto-created)
│   └── calibration/      # Camera calibration files (auto-created)
│       ├── camera_calibration.json
│       ├── manual_stitch_calibration.json
│       └── ball_tracking_config.json
├── models/               # YOLO model files (yolov8n.pt, yolov8s.pt)
├── output/
│   ├── stitched/         # Panoramic videos (auto-created)
│   └── broadcast/        # Broadcast views (auto-created)
├── scripts/
│   ├── detection/
│   │   ├── broadcast.py
│   │   └── setup_ball_tracking.py
│   ├── stitching/
│   │   ├── manual/
│   │   │   ├── calibrate_manual_stitch.py
│   │   │   └── apply_manual_stitch.py
│   │   └── auto/
│   │       ├── stitch_export_transform.py
│   │       └── stitch_apply_transform.py
│   ├── pi_scripts/       # Scripts for Raspberry Pi live capture
│   │   └── main.py
│   ├── run_full_pipeline.py
│   └── undistort_video.py
├── PIPELINE.md           # Technical implementation details
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Sample Data

Download sample dual-camera footage for quick testing:

```bash
cd data && ./download_videos.sh && cd ..
```

**Note:** The sample videos are just for quick testing. You can use any dual-camera videos from your own setup - just ensure they're synchronized and overlapping in view.

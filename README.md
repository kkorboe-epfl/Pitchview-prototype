# PitchView Prototype

Affordable automated sports broadcast system using dual Pi HQ cameras. Stitches two camera feeds into a panoramic view, then uses advanced computer vision with ByteTrack multi-object tracking, Kalman filtering, and physics-based camera motion to generate professional broadcast footage.

**Hardware**: Single Raspberry Pi with two Pi HQ camera modules, providing wide-angle coverage at a fraction of traditional broadcast camera costs.

## Features

- **Panoramic Stitching** from dual-camera feeds with:
  - Smooth edge blending at seam
  - Auto-crop for clean output
  - Feathering for seamless dow

- **Advanced Broadcast System** with:
  - ByteTrack multi-object tracking for persistent player IDs
  - HSV-based ball detection with exclusion zones
  - Kalman filter for ball prediction (handles temporary occlusions)
  - Spring-damper physics-based camera motion (ultra-smooth)
  - Pure pixel-space calculations (no homography required)
  - Field boundary configuration to filter false detections

- **Output**: 720p HD broadcast view with professional camera movement

## Quick Start

### Option 1: Full Pipeline (Automated)

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

# Or use the live demo pipeline (uses live_demo_scripts)
python3 scripts/live_demo_scripts/run_live_demo_pipeline.py \
  --left-raw data/raw/leftflip.mp4 \
  --right-raw data/raw/rightflip.mp4 \
  --output-dir output/live_demo
```

This will automatically:
- Undistort both camera videos
- Stitch them into a panorama
- Set up ball tracking configuration (interactive)
- Generate broadcast view with advanced tracking
- Save screenshots at each step for debugging

All outputs will be in `output/pipeline/` with screenshots in `output/pipeline/screenshots/`.

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

# 5. Stitch dual-camera videos into panorama
python3 scripts/stitching/apply_manual_stitch.py --seam-top 2030 --seam-bottom 2125 --feather 15  

# 6. Configure ball tracking (one-time setup)
# This opens an interactive window to:
#   - Draw a polygon around the playing field (click multiple points, press 'c' to close)
#   - Click on the ball's initial position
#   - Press 'q' to save and exit
python3 scripts/detection/setup_ball_tracking.py output/stitched/panorama.mp4

# 7. Generate broadcast view with advanced tracking
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

# 5. Stitch dual-camera videos into panorama
python scripts/stitching/apply_manual_stitch.py --seam-top 2030 --seam-bottom 2125 --feather 15  

# 6. Configure ball tracking (one-time setup)
# This opens an interactive window to:
#   - Draw a polygon around the playing field (click multiple points, press 'c' to close)
#   - Click on the ball's initial position
#   - Press 'q' to save and exit
python scripts/detection/setup_ball_tracking.py output/stitched/panorama.mp4

# 7. Generate broadcast view with advanced tracking
python scripts/detection/broadcast.py `
  --video output/stitched/panorama.mp4 `
  --save-broadcast output/broadcast/game.mp4 `
  --save-preview output/broadcast/preview.mp4
```


pitchview-prototype/
├── data/
│   ├── raw/              # Input videos (download sample videos with download_videos.sh)
│   └── calibration/      # Camera calibration files
├── models/               # YOLO model files (yolov8n.pt, yolov8s.pt)
├── output/
│   ├── stitched/         # Panoramic videos (auto-created)
│   └── broadcast/        # Broadcast views (auto-created)
├── scripts/
│   ├── detection/        # Ball/player tracking and broadcast
│   │   ├── broadcast.py
│   │   └── setup_ball_tracking.py
│   ├── stitching/        # Panorama creation
│   │   ├── apply_manual_stitch.py
│   │   └── manual_stitch.py
│   ├── live_demo_scripts/# Live demo pipeline scripts
│   │   ├── run_live_demo_pipeline.py
│   │   ├── undistort.py
│   │   ├── stitch_export_transform.py
│   │   └── stitch_apply_transform.py
│   ├── run_full_pipeline.py
│   └── undistort_video.py
└── requirements.txt      # Python dependencies
```

## Pipeline Details

### Step 1: Panoramic Stitching

Stitches left and right camera feeds using manual seam adjustment:

```bash
python3 scripts/stitching/apply_manual_stitch.py \
  --seam-top 2030 \
  --seam-bottom 2125 \
  --feather 15
```

**Note:** Uses default input/output paths from `undistort_video.py` output.

**Options:**
- `--seam-top N` - Top y-coordinate of vertical seam (default: 2030)
- `--seam-bottom N` - Bottom y-coordinate of vertical seam (default: 2125)
- `--feather N` - Feathering width in pixels for blending (default: 15)

### Step 2: Ball Tracking Configuration (One-time)

Configure the field boundary and initial ball position to improve tracking accuracy:

```bash
python3 scripts/detection/setup_ball_tracking.py output/stitched/panorama.mp4
```

**Interactive Steps:**
1. **Draw Field Boundary**: Click around the playing field to create a polygon (8 points recommended)
   - Press 'c' to close the polygon
   - This excludes corner flags, markers, and out-of-bounds areas
2. **Mark Ball Position**: Click on the ball's initial position
3. **Save**: Press 'q' to save configuration

This creates `data/calibration/ball_tracking_config.json` containing:
- Field boundary polygon (for exclusion masking)
- Initial ball position

**Note:** Only needs to be run once per field setup. Configuration is reused for all videos from the same field.

### Step 3: Broadcast Generation

Generate professional broadcast view with advanced tracking:

```bash
python3 scripts/detection/broadcast.py \
  --video output/stitched/panorama.mp4 \
  --save-broadcast output/broadcast/game.mp4 \
  --save-preview output/broadcast/preview.mp4
```

**Features:**
- **ByteTrack**: Persistent player tracking with unique IDs
- **Kalman Filter**: Predicts ball position during occlusions (up to 60 frames)
- **Spring-Damper Camera**: Ultra-smooth physics-based motion (no jarring movements)
- **Smart Framing**: Includes nearby players (within 400px) in the frame
- **Exclusion Zones**: Ignores false detections outside the configured field boundary

**Options:**
- `--save-broadcast` - Save 1280x720 broadcast view
- `--save-preview` - Save full panorama with tracking visualizations (ball predictions, player boxes, camera view rectangle)

## Sample Data

Download sample dual-camera footage for quick testing:

```bash
cd data && ./download_videos.sh && cd ..
```

**Note:** The sample videos are just for quick testing. You can use any dual-camera videos from your own setup - just ensure they're synchronized and overlapping in view.

# PitchView Prototype

Affordable automated sports broadcast system using dual Pi HQ cameras. Stitches two camera feeds into a panoramic view, then uses computer vision to generate professional broadcast footage with automated camera tracking.

**Hardware**: Single Raspberry Pi with two Pi HQ camera modules, providing wide-angle coverage at a fraction of traditional broadcast camera costs.

## Features

- Panoramic stitching from dual-camera feeds with
  - Black border artifact cleanup
  - Smooth edge blending at seam
  - Auto-crop for clean output
- Ball tracking with HSV color detection
- Player detection with YOLOv8
- Automated broadcast camera with velocity-based smoothing
- 720p HD output

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download sample videos
cd data && ./download_videos.sh && cd ..

# or for windows powershell:
# cd data & "C:\Program Files\Git\bin\bash.exe" download_videos.sh & cd ..

# 3. Stitch dual-camera videos into panorama
python3 scripts/stitching/stitch_apply_transform.py \
  --left data/raw/20251116_103024_left.mp4 \
  --right data/raw/20251116_103024_right.mp4 \
  --calib data/calibration/rig_calibration.json \
  --output output/stitched/panorama.mp4 \
  --auto-crop

# 4. Generate broadcast view with tracking
python3 scripts/detection/broadcast_yolo.py \
  --video output/stitched/panorama.mp4 \
  --save-broadcast output/broadcast/game.mp4

```

# or for windows powershell:
```bash
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#  .\.venv\Scripts\Activate.ps1
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download sample videos
cd data
& "C:\Program Files\Git\bin\bash.exe" "./download_videos.sh"
cd ..

# 3. Stitch dual-camera videos into panorama
python scripts/stitching/stitch_apply_transform.py `
  --left data/raw/20251116_103024_left.mp4 `
  --right data/raw/20251116_103024_right.mp4 `
  --calib data/calibration/rig_calibration.json `
  --output output/stitched/panorama.mp4 `
  --auto-crop

# 4. Generate broadcast view with tracking
python scripts/detection/broadcast_yolo.py `
  --video output/stitched/panorama.mp4 `
  --save-broadcast output/broadcast/game.mp4

```
pitchview-prototype/
├── data/
│   ├── raw/              # Input videos (download with download_videos.sh)
│   └── calibration/      # Camera calibration files
├── output/
│   ├── stitched/         # Panoramic videos (auto-created)
│   └── broadcast/        # Broadcast views (auto-created)
├── scripts/
│   ├── stitching/        # Panorama creation
│   └── detection/        # Ball/player tracking and broadcast
└── requirements.txt      # Python dependencies
```

## Pipeline Details

### Step 1: Panoramic Stitching

Uses pre-computed calibration to stitch left and right camera feeds:

```bash
python3 scripts/stitching/stitch_apply_transform.py \
  --left data/raw/20251116_103024_left.mp4 \
  --right data/raw/20251116_103024_right.mp4 \
  --calib data/calibration/rig_calibration.json \
  --output output/stitched/panorama.mp4 \
  --auto-crop
```

**Options:**
- `--auto-crop` - Automatically remove black borders
- `--edge-blend N` - Blend width in pixels (default: 50)
- `--preview` - Show live preview window

### Step 2: Broadcast Generation

Tracks ball and players to generate automated broadcast view:

```bash
python3 scripts/detection/broadcast_yolo.py \
  --video output/stitched/panorama.mp4 \
  --save-broadcast output/broadcast/game.mp4 \
  --save-preview output/broadcast/preview.mp4
```

**Options:**
- `--save-broadcast` - Save broadcast view (720p, tracking window)
- `--save-preview` - Save panorama with overlay annotations

## Calibration (One-time Setup)

If using different cameras or changing camera positions, create new calibration:

```bash
python3 scripts/stitching/stitch_export_transform.py \
  --left data/raw/your_left.mp4 \
  --right data/raw/your_right.mp4 \
  --save-calib data/calibration/custom_calibration.json \
  --preview
```

Then use `--calib data/calibration/custom_calibration.json` in the stitching step.

## Sample Data

Download sample dual-camera footage for quick testing:

```bash
cd data && ./download_videos.sh && cd ..
```

**Note:** The sample videos are just for quick testing. You can use any dual-camera videos from your own setup - just ensure they're synchronized and overlapping in view.
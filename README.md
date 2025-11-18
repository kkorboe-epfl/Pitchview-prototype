# PitchView Prototype

Affordable automated sports broadcast system using dual Pi HQ cameras. Stitches two camera feeds into a panoramic view, then uses computer vision to generate professional broadcast footage with automated camera tracking.

**Hardware**: Single Raspberry Pi with two Pi HQ camera modules, providing wide-angle coverage at a fraction of traditional broadcast camera costs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download sample videos
cd data && ./download_videos.sh && cd ..

# Stitch dual-camera videos into panorama
python scripts/stitching/stitch_apply_transform.py \
  --left data/raw/20251116_103024_left.mp4 \
  --right data/raw/20251116_103024_right.mp4 \
  --calib data/calibration/rig_calibration.json \
  --output output/stitched/panorama.mp4

# Generate broadcast view
python scripts/detection/broadcast_yolo.py \
  --video output/stitched/panorama.mp4 \
  --save-broadcast output/broadcast/game.mp4
```

## Features

- Panoramic stitching from dual-camera feeds
- Ball tracking with HSV color detection
- Player detection with YOLOv8
- Automated broadcast camera with velocity-based smoothing
- 720p HD output

## Project Structure

```
pitchview-prototype/
├── data/
│   ├── raw/              # Input videos
│   └── calibration/      # Camera calibration
├── output/
│   ├── stitched/         # Panoramic videos
│   └── broadcast/        # Broadcast views
└── scripts/
    ├── stitching/        # Panorama creation
    └── detection/        # Tracking and broadcast
```

## Documentation

- [WORKFLOW.md](WORKFLOW.md) - Full pipeline details
- [data/raw/README.md](data/raw/README.md) - Sample video downloads


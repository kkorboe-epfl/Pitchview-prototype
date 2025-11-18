# PitchView Workflow

## Prerequisites

```bash
source venv/bin/activate
cd data && ./download_videos.sh && cd ..
```

## Pipeline

### 1. Stitch Panoramic Video

```bash
python scripts/stitching/stitch_apply_transform.py \
  --left data/raw/20251116_103024_left.mp4 \
  --right data/raw/20251116_103024_right.mp4 \
  --calib data/calibration/rig_calibration.json \
  --output output/stitched/panorama.mp4
```

### 2. Generate Broadcast View

```bash
python scripts/detection/broadcast_yolo.py \
  --video output/stitched/panorama.mp4 \
  --save-broadcast output/broadcast/game.mp4 \
  --save-preview output/broadcast/preview.mp4
```

## Calibration (One-time Setup)

If using different cameras, create new calibration:

```bash
python scripts/stitching/stitch_export_transform.py \
  --left data/raw/20251116_103024_left.mp4 \
  --right data/raw/20251116_103024_right.mp4 \
  --save-calib data/calibration/new_calibration.json \
  --cylindrical --preview
```

Then use `--calib data/calibration/new_calibration.json` in step 1.
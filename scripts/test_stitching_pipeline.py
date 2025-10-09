import os
from pathlib import Path
import subprocess

# === Test video setup ===
RAW_DIR = Path("data/test_videos")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Test video
VIDEO_NAME = "car001"

# === Default directories for testing ===
SPLIT_DIR = Path("data/videos_split")
CROPPED_DIR = SPLIT_DIR / "cropped"
STITCHED_DIR = Path("output/videos_stitched")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_DIR.mkdir(parents=True, exist_ok=True)
STITCHED_DIR.mkdir(parents=True, exist_ok=True)

# === Set environment variables for scripts ===
os.environ["VIDEO_NAME"] = VIDEO_NAME
os.environ["RAW_DIR"] = str(RAW_DIR)
os.environ["SPLIT_DIR"] = str(SPLIT_DIR)
os.environ["CROPPED_DIR"] = str(CROPPED_DIR)
os.environ["STITCHED_DIR"] = str(STITCHED_DIR)
os.environ["FEATURE_DETECTOR"] = "auto"
os.environ["OVERLAP_PIXELS"] = "50"
os.environ["FPS"] = "20"
os.environ["LIVE_PREVIEW"] = "true"

print(f"Running test pipeline for video: {VIDEO_NAME}")
print(f"RAW_DIR: {RAW_DIR}")
print(f"SPLIT_DIR: {SPLIT_DIR}")
print(f"CROPPED_DIR: {CROPPED_DIR}")
print(f"STITCHED_DIR: {STITCHED_DIR}")

# === Run splitting + cropping script ===
subprocess.run(["python", "scripts/preprocessing/video_splitting_with_crop.py"], check=True)

# === Run stitching script ===
subprocess.run(["python", "scripts/stitching/video_stitching.py"], check=True)

print("✅ Test pipeline completed.")
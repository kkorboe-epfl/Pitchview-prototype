import cv2
from pathlib import Path
import os
from dotenv import load_dotenv

# Load config from .env
load_dotenv()

VIDEO_NAME = os.getenv("VIDEO_NAME")
RAW_DIR = Path(os.getenv("RAW_DIR"))
CROPPED_DIR = Path(os.getenv("CROPPED_DIR"))
CROPPED_DIR.mkdir(parents=True, exist_ok=True)

# Input raw video
raw_video = RAW_DIR / f"{VIDEO_NAME}.mp4"

# Output cropped videos
left_out_path = CROPPED_DIR / f"left_cropped_{VIDEO_NAME}.mp4"
right_out_path = CROPPED_DIR / f"right_cropped_{VIDEO_NAME}.mp4"

# Overlap in pixels
overlap = 400

# Open raw video
cap = cv2.VideoCapture(str(raw_video))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {raw_video}")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

half_width = width // 2

# Compute crop coordinates
t = int(round((half_width - overlap) / 2))
left_x0, left_x1 = 0, half_width - t
right_x0, right_x1 = t, half_width

left_crop_w = left_x1 - left_x0
right_crop_w = right_x1 - right_x0

print(f"Splitting and cropping {VIDEO_NAME} with {overlap}px overlap")
print(f"Left crop: x=[{left_x0},{left_x1}) width={left_crop_w}")
print(f"Right crop: x=[{right_x0},{right_x1}) width={right_crop_w}")

# Video writers
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
outL = cv2.VideoWriter(str(left_out_path), fourcc, fps, (left_crop_w, height))
outR = cv2.VideoWriter(str(right_out_path), fourcc, fps, (right_crop_w, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Split
    left_frame_raw = frame[:, :half_width]
    right_frame_raw = frame[:, half_width:]

    # Crop
    left_frame = left_frame_raw[:, left_x0:left_x1]
    right_frame = right_frame_raw[:, right_x0:right_x1]

    outL.write(left_frame)
    outR.write(right_frame)

    frame_idx += 1
    if frame_idx % 200 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
outL.release()
outR.release()
cv2.destroyAllWindows()

print(f"✅ Done! Cropped videos saved as:\n  {left_out_path}\n  {right_out_path}")
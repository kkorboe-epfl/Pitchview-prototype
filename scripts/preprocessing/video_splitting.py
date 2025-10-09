import cv2
import os
from pathlib import Path
from dotenv import load_dotenv

# Load config from .env
load_dotenv()

VIDEO_NAME = os.getenv("VIDEO_NAME")
INPUT_DIR = Path(os.getenv("RAW_DIR"))  
OUTPUT_DIR = Path(os.getenv("SPLIT_DIR"))  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input video file
video_file = INPUT_DIR / f"{VIDEO_NAME}.mp4"

# Output videos
left_video_path = OUTPUT_DIR / f"left_{VIDEO_NAME}.mp4"
right_video_path = OUTPUT_DIR / f"right_{VIDEO_NAME}.mp4"

print("Input video:", video_file)
print("Left output video:", left_video_path)
print("Right output video:", right_video_path)

# Open the video
cap = cv2.VideoCapture(str(video_file))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video_file}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

half_width = width // 2

# Create output writers
out_left = cv2.VideoWriter(str(left_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (half_width, height))
out_right = cv2.VideoWriter(str(right_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (half_width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    left_frame = frame[:, :half_width, :]
    right_frame = frame[:, half_width:, :]

    out_left.write(left_frame)
    out_right.write(right_frame)

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out_left.release()
out_right.release()
cv2.destroyAllWindows()
print(f"✅ Done! Saved {left_video_path} and {right_video_path}")

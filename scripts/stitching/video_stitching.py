import cv2
import numpy as np
from pathlib import Path
import platform
import os
from dotenv import load_dotenv

# === LOAD CONFIG FROM .env ===
load_dotenv()

VIDEO_NAME = os.getenv("VIDEO_NAME")
INPUT_DIR = Path(os.getenv("CROPPED_DIR"))
OUTPUT_DIR = Path(os.getenv("STITCHED_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DETECTOR = os.getenv("FEATURE_DETECTOR", "auto")
OVERLAP_PIXELS = int(os.getenv("OVERLAP_PIXELS", 50))
LIVE_PREVIEW = os.getenv("LIVE_PREVIEW", "false").lower() == "true"

video_left = INPUT_DIR / f"left_cropped_{VIDEO_NAME}.mp4"
video_right = INPUT_DIR / f"right_cropped_{VIDEO_NAME}.mp4"
output_video = OUTPUT_DIR / f"stitched_{VIDEO_NAME}.mp4"

# === OPEN VIDEOS ===
left_video_capture = cv2.VideoCapture(str(video_left))
right_video_capture = cv2.VideoCapture(str(video_right))
left_frame_ok, left_frame = left_video_capture.read()
right_frame_ok, right_frame = right_video_capture.read()
if not (left_frame_ok and right_frame_ok):
    raise RuntimeError("Cannot read first frames.")

# === FEATURE DETECTOR SELECTION ===
if FEATURE_DETECTOR.lower() == "sift":
    detector = cv2.SIFT_create()
    print("Using SIFT feature detector (forced).")
elif FEATURE_DETECTOR.lower() == "orb":
    # 4000 = maximum number of keypoints ORB will retain.
    # More keypoints → better matching accuracy but slower processing.
    # Fewer keypoints → faster but may reduce stitch quality, especially on high-res frames.
    detector = cv2.ORB_create(4000)
    print("Using ORB feature detector (forced).")
else:
    # Auto: detect CPU and platform
    system = platform.system()
    machine = platform.machine()
    cpu_count = os.cpu_count()
    if system == "Linux" and ("arm" in machine or cpu_count <= 4):
        detector = cv2.ORB_create(4000)
        print("Using ORB for speed (low-power CPU detected).")
    else:
        detector = cv2.SIFT_create()
        print("Using SIFT (high-CPU system detected).")

# === COMPUTE KEYPOINTS AND HOMOGRAPHY ===
kp1, des1 = detector.detectAndCompute(left_frame, None)
kp2, des2 = detector.detectAndCompute(right_frame, None)

if isinstance(detector, cv2.ORB):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
else:
    bf = cv2.BFMatcher()
matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:200]

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
H = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)[0].astype(np.float32)

# === PANORAMA SIZE ===
hL, wL = left_frame.shape[:2]
hR, wR = right_frame.shape[:2]

corners_right = np.float32([[0,0],[wR,0],[wR,hR],[0,hR]]).reshape(-1,1,2)
warped_corners = cv2.perspectiveTransform(corners_right, H)
all_corners = np.vstack((np.float32([[0,0],[wL,0],[wL,hL],[0,hL]]).reshape(-1,1,2), warped_corners))

[xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
panorama_width = xmax - xmin
panorama_height = ymax - ymin
T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)

# === VIDEO WRITER ===
fps = int(min(left_video_capture.get(cv2.CAP_PROP_FPS), right_video_capture.get(cv2.CAP_PROP_FPS)))
out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                      (panorama_width, panorama_height))

# Reset frame pointers
left_video_capture.set(cv2.CAP_PROP_POS_FRAMES,0)
right_video_capture.set(cv2.CAP_PROP_POS_FRAMES,0)

frame_count = 0
print("Stitching video frames...")

# === PROCESS FRAMES ===
while True:
    left_frame_ok, left_frame = left_video_capture.read()
    right_frame_ok, right_frame = right_video_capture.read()
    if not (left_frame_ok and right_frame_ok):
        break

    warped_right = cv2.warpPerspective(right_frame, T @ H, (panorama_width, panorama_height))
    warped_left = cv2.warpPerspective(left_frame, T, (panorama_width, panorama_height))

    # Linear blending in overlap
    mask_left = (warped_left > 0).astype(np.float32)
    mask_right = (warped_right > 0).astype(np.float32)
    overlap = mask_left + mask_right
    overlap[overlap==0] = 1
    blended = (warped_left.astype(np.float32) + warped_right.astype(np.float32)) / overlap
    blended = np.clip(blended,0,255).astype(np.uint8)

    out.write(blended)
    frame_count +=1

    # Live preview
    if LIVE_PREVIEW:
        cv2.imshow("Stitched Preview", blended)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview closed by user.")
            cv2.destroyWindow("Stitched Preview")
            LIVE_PREVIEW = False

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

left_video_capture.release()
right_video_capture.release()
out.release()
if LIVE_PREVIEW:
    cv2.destroyAllWindows()
print(f"✅ Done! Stitched video saved as {output_video}")
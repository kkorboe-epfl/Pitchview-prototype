#!/usr/bin/env python3
"""
Calibrate two cameras (lens distortion) from two video files.

Usage:
    python calibrate_from_video.py \
        --left Footage16-11/20251116_102613_cam0.mp4 \
        --right Footage16-11/20251116_102613_cam1.mp4 \
        --cols 9 --rows 6 --square 0.025 \
        --frames 30

Output:
    cam0_lens_calib.json
    cam1_lens_calib.json
"""

import argparse
import json
import cv2
import numpy as np
import os
import sys


# ------------------------------
# Checkerboard detection helper
# ------------------------------

def detect_corners(gray, pattern_size):
    """Try to detect and refine checkerboard corners."""
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )
    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria
    )
    return refined


# ------------------------------
# Calibrate one camera
# ------------------------------

def calibrate_single_camera(video_path, pattern_size, square_size, max_frames, show=False, label="cam"):
    """Extract frames from a video and perform lens calibration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = np.linspace(0, total_frames - 1, max_frames).astype(int)

    print(f"[info] {label}: sampling {len(frame_ids)} frames out of {total_frames}")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    image_size = None
    used = 0

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, img = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        corners = detect_corners(gray, pattern_size)
        if corners is None:
            print(f"[warn] {label}: no pattern in frame {fid}")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used += 1

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            cv2.imshow(f"{label} corners", vis)
            if (cv2.waitKey(200) & 0xFF) in (27, ord('q')):
                show = False
                cv2.destroyAllWindows()

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if used < 8:
        print(f"[error] {label}: only {used} valid frames. Need at least 8.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] {label}: using {used} valid frames for calibration")

    # Perform calibration
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print(f"[info] {label}: RMS error = {rms:.4f}")
    print(f"[info] {label}: K =\n{K}")
    print(f"[info] {label}: D = {D.ravel()}")

    return {
        "K": K.tolist(),
        "D": D.tolist(),
        "image_size": [int(image_size[0]), int(image_size[1])]
    }


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Calibrate both cameras from two videos.")
    ap.add_argument("--left", required=True, help="Video file for left camera")
    ap.add_argument("--right", required=True, help="Video file for right camera")
    ap.add_argument("--cols", type=int, required=True, help="Checkerboard inner corners horizontally")
    ap.add_argument("--rows", type=int, required=True, help="Checkerboard inner corners vertically")
    ap.add_argument("--square", type=float, default=0.025, help="Square size in metres")
    ap.add_argument("--frames", type=int, default=30, help="How many frames to sample per camera")
    ap.add_argument("--show", action="store_true", help="Show corner detection as calibration runs")
    args = ap.parse_args()

    pattern_size = (args.cols, args.rows)

    print("[info] Calibrating left camera")
    left_calib = calibrate_single_camera(
        args.left, pattern_size, args.square, args.frames, args.show, label="cam0"
    )

    print("[info] Calibrating right camera")
    right_calib = calibrate_single_camera(
        args.right, pattern_size, args.square, args.frames, args.show, label="cam1"
    )

    # Save JSON files
    with open("cam0_lens_calib.json", "w", encoding="utf-8") as f:
        json.dump(left_calib, f, indent=2)
    with open("cam1_lens_calib.json", "w", encoding="utf-8") as f:
        json.dump(right_calib, f, indent=2)

    print("\n[info] Saved:")
    print("  cam0_lens_calib.json")
    print("  cam1_lens_calib.json")


if __name__ == "__main__":
    main()

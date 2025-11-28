#!/usr/bin/env python3
"""
Simple and robust panoramic calibration for dual-camera stitching.
Computes homography from feature matching and exports transform for later use.

Usage:
  python scripts/stitching/stitch_export_transform.py \
    --left data/undistorted/left.mp4 \
    --right data/undistorted/right.mp4 \
    --save-calib data/calibration/custom_calibration.json \
    --preview
"""
import argparse
import sys
import json
import numpy as np
import cv2


def open_video(path):
    """Open a video file and return VideoCapture object."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def read_frame(cap):
    """Read a frame from video capture."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def detect_features(image, max_features=10000):
    """Detect ORB features in grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use SIFT instead of ORB for better feature detection
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors


def match_features(desc1, desc2, ratio_threshold=0.7):
    """Match features using FLANN matcher with ratio test."""
    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches


def compute_homography(left_image, right_image):
    """
    Compute homography from right image to left image coordinates.
    Returns H matrix that transforms right -> left.
    """
    print("Detecting features in left image...")
    kp_left, desc_left = detect_features(left_image)
    print(f"  Found {len(kp_left)} features")
    
    print("Detecting features in right image...")
    kp_right, desc_right = detect_features(right_image)
    print(f"  Found {len(kp_right)} features")
    
    if desc_left is None or desc_right is None:
        raise RuntimeError("Failed to detect features in one or both images")
    
    print("Matching features...")
    matches = match_features(desc_right, desc_left)
    print(f"  Found {len(matches)} good matches")
    
    if len(matches) < 4:
        raise RuntimeError(f"Not enough matches found ({len(matches)}). Need at least 4.")
    
    # Extract matched point coordinates
    pts_right = np.float32([kp_right[m.queryIdx].pt for m in matches])
    pts_left = np.float32([kp_left[m.trainIdx].pt for m in matches])
    
    # Compute homography with RANSAC
    print("Computing homography...")
    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 5.0, maxIters=2000, confidence=0.995)
    
    if H is None:
        raise RuntimeError("Failed to compute homography")
    
    inliers = int(mask.sum())
    inlier_ratio = inliers / len(matches)
    print(f"  Inliers: {inliers}/{len(matches)} ({inlier_ratio*100:.1f}%)")
    
    if inliers < 4:
        raise RuntimeError(f"Not enough inliers ({inliers}). Homography may be unreliable.")
    
    if inlier_ratio < 0.2:
        print(f"  Warning: Low inlier ratio ({inlier_ratio*100:.1f}%). Results may be unstable.")
    
    return H


def calculate_canvas_size(left_shape, right_shape, H):
    """
    Calculate the canvas size needed to fit both warped images.
    Returns (width, height, x_offset, y_offset).
    """
    h_left, w_left = left_shape[:2]
    h_right, w_right = right_shape[:2]
    
    # Corners of right image
    corners_right = np.float32([
        [0, 0],
        [w_right, 0],
        [w_right, h_right],
        [0, h_right]
    ]).reshape(-1, 1, 2)
    
    # Transform right corners to left coordinate system
    corners_right_warped = cv2.perspectiveTransform(corners_right, H)
    
    # Corners of left image (already in left coordinate system)
    corners_left = np.float32([
        [0, 0],
        [w_left, 0],
        [w_left, h_left],
        [0, h_left]
    ]).reshape(-1, 1, 2)
    
    # Combine all corners
    all_corners = np.concatenate([corners_left, corners_right_warped], axis=0)
    
    # Find bounding box
    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))
    
    # Canvas size
    width = x_max - x_min
    height = y_max - y_min
    
    # Offset to shift everything into positive coordinates
    offset = (-x_min, -y_min)
    
    print(f"Canvas size: {width}x{height}")
    print(f"Offset: {offset}")
    
    return width, height, offset


def stitch_images(left, right, H, offset, canvas_size):
    """Stitch two images using the homography."""
    width, height = canvas_size
    offset_x, offset_y = offset
    
    # Translation matrix to shift everything by offset
    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Warp right image
    H_shifted = T @ H
    right_warped = cv2.warpPerspective(right, H_shifted, (width, height))
    
    # Place left image
    result = right_warped.copy()
    h_left, w_left = left.shape[:2]
    
    # Make sure left image fits within canvas
    y_end = min(offset_y + h_left, height)
    x_end = min(offset_x + w_left, width)
    h_copy = y_end - offset_y
    w_copy = x_end - offset_x
    
    # Copy left image over (simple overlay for now)
    result[offset_y:y_end, offset_x:x_end] = left[:h_copy, :w_copy]
    
    return result


def save_calibration(path, H, offset, pano_size):
    """Save calibration to JSON file."""
    data = {
        "H": H.tolist(),
        "offset": list(offset),
        "pano_size": list(pano_size),
        "used_affine": False
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved calibration to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute and export panoramic stitching calibration'
    )
    parser.add_argument('--left', required=True, help='Left video path')
    parser.add_argument('--right', required=True, help='Right video path')
    parser.add_argument('--save-calib', required=True, help='Output calibration JSON path')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    
    args = parser.parse_args()
    
    print("Opening videos...")
    cap_left = open_video(args.left)
    cap_right = open_video(args.right)
    
    print("Reading first frames...")
    frame_left = read_frame(cap_left)
    frame_right = read_frame(cap_right)
    
    if frame_left is None or frame_right is None:
        print("Error: Could not read frames from videos")
        sys.exit(1)
    
    print(f"Left frame: {frame_left.shape[1]}x{frame_left.shape[0]}")
    print(f"Right frame: {frame_right.shape[1]}x{frame_right.shape[0]}")
    
    # Compute homography
    H = compute_homography(frame_left, frame_right)
    
    # Calculate canvas size
    width, height, offset = calculate_canvas_size(
        frame_left.shape, frame_right.shape, H
    )
    
    # Save calibration
    save_calibration(args.save_calib, H, offset, (width, height))
    
    # Preview if requested
    if args.preview:
        print("\nGenerating preview...")
        panorama = stitch_images(frame_left, frame_right, H, offset, (width, height))
        
        # Scale for display if too large
        max_display_width = 1600
        if width > max_display_width:
            scale = max_display_width / width
            display_width = max_display_width
            display_height = int(height * scale)
            panorama_display = cv2.resize(panorama, (display_width, display_height))
        else:
            panorama_display = panorama
        
        cv2.imshow("Panorama Preview", panorama_display)
        print("Press any key to close preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cap_left.release()
    cap_right.release()
    
    print("\nCalibration complete!")


if __name__ == '__main__':
    main()
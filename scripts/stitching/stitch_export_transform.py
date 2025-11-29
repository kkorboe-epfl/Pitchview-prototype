import argparse
import sys
import json
import os
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
    """Detect SIFT features in grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors


def match_features(desc1, desc2, ratio_threshold=0.7):
    """Match features using FLANN matcher with ratio test."""
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
    """Stitch two images using the homography with blending."""
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
    
    # Create masks
    right_mask = cv2.warpPerspective(np.ones_like(right, dtype=np.float32), 
                                      H_shifted, (width, height))
    
    # Place left image
    result = np.zeros((height, width, 3), dtype=np.float32)
    left_mask = np.zeros((height, width, 3), dtype=np.float32)
    
    h_left, w_left = left.shape[:2]
    y_end = min(offset_y + h_left, height)
    x_end = min(offset_x + w_left, width)
    h_copy = y_end - offset_y
    w_copy = x_end - offset_x
    
    result[offset_y:y_end, offset_x:x_end] = left[:h_copy, :w_copy].astype(np.float32)
    left_mask[offset_y:y_end, offset_x:x_end] = 1.0
    
    # Simple blending where images overlap
    total_mask = left_mask + right_mask
    total_mask[total_mask == 0] = 1.0  # Avoid division by zero
    
    result = (result + right_warped.astype(np.float32)) / total_mask
    
    return result.astype(np.uint8)


def save_calibration(path, H, offset, pano_size):
    """Save calibration to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
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
        description='Stitch left and right undistorted videos into panorama'
    )
    parser.add_argument('--left', default='data/undistorted/left.mp4', 
                        help='Left video path (default: data/undistorted/left.mp4)')
    parser.add_argument('--right', default='data/undistorted/right.mp4',
                        help='Right video path (default: data/undistorted/right.mp4)')
    parser.add_argument('--output', default='data/stitched/panorama.mp4',
                        help='Output video path (default: data/stitched/panorama.mp4)')
    parser.add_argument('--calib', default='data/calibration/stitch_calibration.json',
                        help='Calibration output path (default: data/calibration/stitch_calibration.json)')
    parser.add_argument('--preview', action='store_true', 
                        help='Show preview of stitched frame')
    parser.add_argument('--process-video', action='store_true',
                        help='Process and save entire stitched video')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PANORAMIC VIDEO STITCHING")
    print("=" * 60)
    print(f"Left video:  {args.left}")
    print(f"Right video: {args.right}")
    print(f"Output:      {args.output}")
    print(f"Calibration: {args.calib}")
    print("=" * 60)
    
    print("\nOpening videos...")
    cap_left = open_video(args.left)
    cap_right = open_video(args.right)
    
    print("Reading first frames...")
    frame_left = read_frame(cap_left)
    frame_right = read_frame(cap_right)
    
    if frame_left is None or frame_right is None:
        print("Error: Could not read frames from videos")
        sys.exit(1)
    
    print(f"Left frame:  {frame_left.shape[1]}x{frame_left.shape[0]}")
    print(f"Right frame: {frame_right.shape[1]}x{frame_right.shape[0]}")
    
    # Compute homography
    print("\n" + "=" * 60)
    H = compute_homography(frame_left, frame_right)
    print("=" * 60)
    
    # Calculate canvas size
    print("\nCalculating panorama dimensions...")
    width, height, offset = calculate_canvas_size(
        frame_left.shape, frame_right.shape, H
    )
    
    # Save calibration
    print("\nSaving calibration...")
    save_calibration(args.calib, H, offset, (width, height))
    
    # Preview if requested
    if args.preview:
        print("\n" + "=" * 60)
        print("GENERATING PREVIEW")
        print("=" * 60)
        panorama = stitch_images(frame_left, frame_right, H, offset, (width, height))
        
        # Scale for display if too large
        max_display_width = 1920
        if width > max_display_width:
            scale = max_display_width / width
            display_width = max_display_width
            display_height = int(height * scale)
            panorama_display = cv2.resize(panorama, (display_width, display_height))
            print(f"Display scaled to: {display_width}x{display_height}")
        else:
            panorama_display = panorama
        
        cv2.imshow("Panorama Preview (Press any key to close)", panorama_display)
        print("Press any key to close preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Process full video if requested
    if args.process_video:
        print("\n" + "=" * 60)
        print("PROCESSING FULL VIDEO")
        print("=" * 60)
        
        # Reset video captures
        cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fps = int(cap_left.get(cv2.CAP_PROP_FPS))
        total_frames = int(min(
            cap_left.get(cv2.CAP_PROP_FRAME_COUNT),
            cap_right.get(cv2.CAP_PROP_FRAME_COUNT)
        ))
        
        print(f"Output: {width}x{height} @ {fps}fps")
        print(f"Total frames: {total_frames}")
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            left = read_frame(cap_left)
            right = read_frame(cap_right)
            
            if left is None or right is None:
                break
            
            # Stitch frames
            panorama = stitch_images(left, right, H, offset, (width, height))
            out.write(panorama)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)", end='\r')
        
        print(f"\nCompleted! Processed {frame_count} frames")
        print(f"Saved to: {args.output}")
        
        out.release()
    
    cap_left.release()
    cap_right.release()
    
    print("\n" + "=" * 60)
    print("STITCHING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
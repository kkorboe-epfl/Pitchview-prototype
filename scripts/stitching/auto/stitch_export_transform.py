#!/usr/bin/env python3
"""
SIFT-based homography calibration for dual-camera stitching.
Uses SIFT feature detection, FLANN matching, and RANSAC homography estimation.

Usage:
  python scripts/stitching/auto/stitch_export_transform.py \
    --left data/undistorted/left.mp4 \
    --right data/undistorted/right.mp4 \
    --save-calib data/calibration/auto_calibration.json \
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


def detect_sift_features(image, max_features=10000, roi=None):
    """
    Detect SIFT features in image.
    
    Args:
        image: Input BGR image
        max_features: Maximum number of features to detect
        roi: Region of interest as (x, y, w, h) or None for full image
    
    Returns:
        keypoints, descriptors
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better feature detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Create ROI mask if specified
    mask = None
    if roi is not None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        x, y, w, h = roi
        mask[y:y+h, x:x+w] = 255
    
    # Detect SIFT features
    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.03,  # Lower = more features
        edgeThreshold=10,        # Lower = more edge features
        sigma=1.6
    )
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    
    return keypoints, descriptors


def match_features_flann(desc1, desc2, ratio_threshold=0.75):
    """
    Match features using FLANN-based matcher with Lowe's ratio test.
    
    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from second image
        ratio_threshold: Ratio test threshold (lower = stricter)
    
    Returns:
        List of good DMatch objects
    """
    # FLANN parameters for SIFT/SURF descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find 2 nearest neighbors
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches


def estimate_homography_ransac(kp1, kp2, matches, ransac_threshold=5.0):
    """
    Estimate homography using RANSAC.
    
    Args:
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: List of DMatch objects
        ransac_threshold: RANSAC reprojection threshold in pixels
    
    Returns:
        H: 3x3 homography matrix (or None if failed)
        inlier_mask: Binary mask of inliers
        inlier_count: Number of inliers
    """
    if len(matches) < 4:
        return None, None, 0
    
    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography with RANSAC
    H, mask = cv2.findHomography(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=5000,
        confidence=0.995
    )
    
    if H is None or mask is None:
        return None, None, 0
    
    inlier_count = int(mask.sum())
    
    return H, mask, inlier_count


def visualize_matches(img1, kp1, img2, kp2, matches, mask=None, max_display_width=1920):
    """Create visualization of feature matches."""
    if mask is not None:
        # Only show inliers
        matches_to_draw = [m for i, m in enumerate(matches) if mask[i]]
        title = f"RANSAC Inliers: {len(matches_to_draw)}/{len(matches)}"
    else:
        # Show all matches
        matches_to_draw = matches
        title = f"All Matches: {len(matches)}"
    
    # Limit number of matches to draw for clarity
    if len(matches_to_draw) > 100:
        matches_to_draw = matches_to_draw[::len(matches_to_draw)//100]
    
    # Draw matches
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Add title
    cv2.putText(match_img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Scale if too large
    h, w = match_img.shape[:2]
    if w > max_display_width:
        scale = max_display_width / w
        new_w = max_display_width
        new_h = int(h * scale)
        match_img = cv2.resize(match_img, (new_w, new_h))
    
    cv2.imshow("Feature Matches (press any key)", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_homography(img_left, img_right, overlap_pct=0.5, show_matches=False, 
                       max_features=15000, ratio_threshold=0.75, ransac_threshold=5.0):
    """
    Compute homography from right image to left image using SIFT features.
    
    Args:
        img_left: Left camera image
        img_right: Right camera image
        overlap_pct: Expected overlap percentage (0.0-1.0)
        show_matches: Whether to show match visualization
        max_features: Maximum SIFT features to detect
        ratio_threshold: Lowe's ratio test threshold
        ransac_threshold: RANSAC reprojection threshold
    
    Returns:
        H: 3x3 homography matrix transforming right image to left coordinate system
    """
    h, w = img_left.shape[:2]
    
    # Define overlap ROIs for better feature matching
    overlap_width = int(w * overlap_pct)
    
    # Right side of left image
    roi_left = (w - overlap_width, 0, overlap_width, h)
    # Left side of right image  
    roi_right = (0, 0, overlap_width, h)
    
    print(f"\nDetecting SIFT features (max {max_features})...")
    print(f"  Overlap ROI width: {overlap_width}px ({int(overlap_pct*100)}% of image)")
    
    # Detect features in overlap regions
    kp_left, desc_left = detect_sift_features(img_left, max_features, roi_left)
    kp_right, desc_right = detect_sift_features(img_right, max_features, roi_right)
    
    print(f"  Left features: {len(kp_left)}")
    print(f"  Right features: {len(kp_right)}")
    
    if len(kp_left) < 10 or len(kp_right) < 10:
        raise RuntimeError(
            f"Insufficient features detected (left: {len(kp_left)}, right: {len(kp_right)}). "
            "Try increasing --max-features or ensure overlap region has visible texture."
        )
    
    # Match features using FLANN
    print(f"\nMatching features (ratio threshold: {ratio_threshold})...")
    matches = match_features_flann(desc_right, desc_left, ratio_threshold)
    
    print(f"  Good matches: {len(matches)}")
    
    if len(matches) < 10:
        raise RuntimeError(
            f"Too few feature matches ({len(matches)}). "
            f"Try increasing --ratio-threshold or --overlap-pct"
        )
    
    # Estimate homography with RANSAC
    print(f"\nEstimating homography (RANSAC threshold: {ransac_threshold}px)...")
    H, mask, inlier_count = estimate_homography_ransac(
        kp_right, kp_left, matches, ransac_threshold
    )
    
    if H is None:
        raise RuntimeError("Homography estimation failed")
    
    inlier_pct = (inlier_count / len(matches)) * 100
    print(f"  Inliers: {inlier_count}/{len(matches)} ({inlier_pct:.1f}%)")
    
    if inlier_count < 10:
        raise RuntimeError(
            f"Too few inliers ({inlier_count}). The homography may be unreliable. "
            "Try adjusting --ransac-threshold or check camera overlap."
        )
    
    if inlier_pct < 30:
        print(f"  WARNING: Low inlier percentage ({inlier_pct:.1f}%). Results may be poor.")
    
    # Visualize matches if requested
    if show_matches:
        visualize_matches(img_right, kp_right, img_left, kp_left, matches, mask)
    
    return H


def calculate_canvas_size(left_shape, right_shape, H):
    """Calculate canvas size needed for panorama."""
    h_left, w_left = left_shape[:2]
    h_right, w_right = right_shape[:2]
    
    # Transform right image corners
    corners_right = np.float32([
        [0, 0], [w_right, 0],
        [w_right, h_right], [0, h_right]
    ]).reshape(-1, 1, 2)
    corners_right_transformed = cv2.perspectiveTransform(corners_right, H)
    
    # Left image corners (in left coordinate system)
    corners_left = np.float32([
        [0, 0], [w_left, 0],
        [w_left, h_left], [0, h_left]
    ]).reshape(-1, 1, 2)
    
    # Combine all corners
    all_corners = np.concatenate([corners_left, corners_right_transformed], axis=0)
    
    # Find bounding box
    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))
    
    # Canvas size
    width = x_max - x_min
    height = y_max - y_min
    
    # Offset to handle negative coordinates
    offset = (-x_min, -y_min)
    
    return width, height, offset


def stitch_images(left_img, right_img, H, offset, pano_size):
    """Stitch two images using homography and alpha blending."""
    width, height = pano_size
    offset_x, offset_y = offset
    
    # Create offset homography for right image
    H_offset = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64) @ H
    
    # Warp right image
    right_warped = cv2.warpPerspective(right_img, H_offset, (width, height))
    
    # Place left image on canvas with offset
    left_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left_canvas[offset_y:offset_y+left_img.shape[0], 
                offset_x:offset_x+left_img.shape[1]] = left_img
    
    # Create masks
    mask_left = cv2.cvtColor(left_canvas, cv2.COLOR_BGR2GRAY) > 0
    mask_right = cv2.cvtColor(right_warped, cv2.COLOR_BGR2GRAY) > 0
    mask_overlap = mask_left & mask_right
    
    # Alpha blend in overlap region
    result = np.zeros_like(left_canvas, dtype=np.float32)
    
    if mask_overlap.sum() > 0:
        # Compute distance transforms for blending weights
        dist_left = cv2.distanceTransform(mask_left.astype(np.uint8), cv2.DIST_L2, 5)
        dist_right = cv2.distanceTransform(mask_right.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Normalize distances in overlap
        total_dist = dist_left + dist_right
        total_dist[total_dist == 0] = 1.0
        
        alpha = dist_left / total_dist
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        
        # Blend in overlap
        overlap_mask_3ch = np.dstack([mask_overlap, mask_overlap, mask_overlap])
        result[mask_overlap] = (
            left_canvas[overlap_mask_3ch].astype(np.float32) * alpha_3ch[overlap_mask_3ch] +
            right_warped[overlap_mask_3ch].astype(np.float32) * (1 - alpha_3ch[overlap_mask_3ch])
        )
        
        # Non-overlap regions
        mask_left_only = mask_left & ~mask_overlap
        mask_right_only = mask_right & ~mask_overlap
        result[mask_left_only] = left_canvas[mask_left_only].astype(np.float32)
        result[mask_right_only] = right_warped[mask_right_only].astype(np.float32)
    else:
        # No overlap - just combine
        result[mask_left] = left_canvas[mask_left].astype(np.float32)
        result[mask_right] = right_warped[mask_right].astype(np.float32)
    
    return result.astype(np.uint8)


def save_calibration(path, H, offset, pano_size):
    """Save calibration to JSON file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data = {
        "H": H.tolist(),
        "offset": list(offset),
        "pano_size": list(pano_size),
        "method": "SIFT+FLANN+RANSAC"
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved calibration to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Automatic stitching calibration using SIFT+FLANN+RANSAC',
        epilog='Requires overlapping field of view with visible texture/features.'
    )
    parser.add_argument('--left', required=True, help='Path to left camera video')
    parser.add_argument('--right', required=True, help='Path to right camera video')
    parser.add_argument('--save-calib', required=True, help='Output calibration JSON path')
    parser.add_argument('--preview', action='store_true', help='Show stitched preview')
    parser.add_argument('--show-matches', action='store_true', help='Show feature match visualization')
    parser.add_argument('--overlap-pct', type=float, default=0.3,
                       help='Expected overlap as fraction of width (0.0-1.0), default 0.3')
    parser.add_argument('--max-features', type=int, default=15000,
                       help='Maximum SIFT features to detect, default 15000')
    parser.add_argument('--ratio-threshold', type=float, default=0.75,
                       help='Lowe\'s ratio test threshold (0.0-1.0), default 0.75')
    parser.add_argument('--ransac-threshold', type=float, default=5.0,
                       help='RANSAC reprojection threshold in pixels, default 5.0')
    
    args = parser.parse_args()
    
    if not 0 < args.overlap_pct <= 1.0:
        print("Error: --overlap-pct must be between 0 and 1")
        sys.exit(1)
    
    print("=" * 60)
    print("SIFT-based Automatic Stitching Calibration")
    print("=" * 60)
    
    # Open videos
    print("\nOpening videos...")
    cap_left = open_video(args.left)
    cap_right = open_video(args.right)
    
    # Read first frames
    print("Reading first frames...")
    frame_left = read_frame(cap_left)
    frame_right = read_frame(cap_right)
    
    if frame_left is None or frame_right is None:
        print("Error: Could not read frames from videos")
        sys.exit(1)
    
    print(f"Left frame: {frame_left.shape[1]}x{frame_left.shape[0]}")
    print(f"Right frame: {frame_right.shape[1]}x{frame_right.shape[0]}")
    
    # Compute homography
    try:
        H = compute_homography(
            frame_left, frame_right,
            overlap_pct=args.overlap_pct,
            show_matches=args.show_matches,
            max_features=args.max_features,
            ratio_threshold=args.ratio_threshold,
            ransac_threshold=args.ransac_threshold
        )
    except RuntimeError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        print("\nTroubleshooting:")
        print("1. Verify cameras have overlapping view of a textured scene")
        print("2. Try adjusting --overlap-pct (increase if more overlap)")
        print("3. Try increasing --max-features (e.g., 20000)")
        print("4. Try increasing --ratio-threshold (e.g., 0.8)")
        print("5. Use --show-matches to visualize feature detection")
        print("6. Consider manual calibration if scene lacks features")
        sys.exit(1)
    
    # Calculate canvas size
    width, height, offset = calculate_canvas_size(
        frame_left.shape, frame_right.shape, H
    )
    print(f"\nPanorama size: {width}x{height}")
    print(f"Offset: {offset}")
    
    # Save calibration
    save_calibration(args.save_calib, H, offset, (width, height))
    
    # Generate preview if requested
    if args.preview:
        print("\nGenerating preview...")
        panorama = stitch_images(frame_left, frame_right, H, offset, (width, height))
        
        # Scale for display
        max_width = 1600
        if width > max_width:
            scale = max_width / width
            display_size = (max_width, int(height * scale))
            panorama = cv2.resize(panorama, display_size)
        
        cv2.imshow("Panorama Preview (press any key to close)", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cap_left.release()
    cap_right.release()
    
    print("\n" + "="*60)
    print("Calibration complete!")
    print("="*60)
    print("\nNext: Run stitch_apply_transform.py to stitch full videos")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Full pipeline: Undistort -> Stitch -> Broadcast
Saves screenshots at each step for debugging and visualization.

Usage:
  python scripts/run_full_pipeline.py \
    --left-raw data/raw/leftflip.mp4 \
    --right-raw data/raw/rightflip.mp4 \
    --calib data/calibration/custom_calibration.json \
    --output-dir output/pipeline
"""
import argparse
import sys
import os
import subprocess
import cv2
from pathlib import Path


def save_screenshot(video_path, output_path, frame_number=0):
    """Save a screenshot from a video file."""
    cap = cv2.VideoCapture(video_path)
    
    # Seek to specific frame
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"  Saved screenshot: {output_path}")
    else:
        print(f"  Warning: Could not read frame from {video_path}")
    
    cap.release()


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Run full pipeline: undistort -> stitch -> broadcast'
    )
    parser.add_argument('--left-raw', required=True, help='Left raw video path')
    parser.add_argument('--right-raw', required=True, help='Right raw video path')
    parser.add_argument('--calib', required=True, help='Stitching calibration JSON')
    parser.add_argument('--output-dir', default='output/pipeline', help='Output directory')
    parser.add_argument('--seam-x', type=int, default=3900, help='Seam position for stitching')
    parser.add_argument('--sync-offset', type=int, default=-1, help='Frame sync offset')
    parser.add_argument('--screenshot-frames', type=int, nargs='+', default=[100, 300, 500, 700, 900],
                       help='Frame numbers for screenshots (multiple frames for broadcast/preview)')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    undistorted_dir = output_dir / "undistorted"
    undistorted_dir.mkdir(exist_ok=True)
    
    stitched_dir = output_dir / "stitched"
    stitched_dir.mkdir(exist_ok=True)
    
    broadcast_dir = output_dir / "broadcast"
    broadcast_dir.mkdir(exist_ok=True)
    
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    
    # Define paths
    left_undistorted = undistorted_dir / "left.mp4"
    right_undistorted = undistorted_dir / "right.mp4"
    stitched_video = stitched_dir / "panorama.mp4"
    broadcast_video = broadcast_dir / "game.mp4"
    preview_video = broadcast_dir / "preview.mp4"
    
    print(f"\n{'='*60}")
    print("FULL PIPELINE STARTED")
    print(f"{'='*60}")
    print(f"Left raw: {args.left_raw}")
    print(f"Right raw: {args.right_raw}")
    print(f"Calibration: {args.calib}")
    print(f"Output directory: {output_dir}")
    
    # Step 0: Save original raw screenshots
    print(f"\n{'='*60}")
    print("STEP 0: Saving original raw screenshots")
    print(f"{'='*60}")
    # Just use first frame for raw/undistorted/stitched
    first_frame = args.screenshot_frames[0]
    save_screenshot(args.left_raw, screenshots_dir / "01_raw_left.jpg", first_frame)
    save_screenshot(args.right_raw, screenshots_dir / "02_raw_right.jpg", first_frame)
    
    # Step 1: Undistort left camera
    run_command(
        [
            'python3', 'scripts/undistort_video.py',
            '--camera', 'left',
            '--input', args.left_raw,
            '--output', str(left_undistorted)
        ],
        "Undistort left camera"
    )
    save_screenshot(str(left_undistorted), screenshots_dir / "03_undistorted_left.jpg", first_frame)
    
    # Step 2: Undistort right camera
    run_command(
        [
            'python3', 'scripts/undistort_video.py',
            '--camera', 'right',
            '--input', args.right_raw,
            '--output', str(right_undistorted)
        ],
        "Undistort right camera"
    )
    save_screenshot(str(right_undistorted), screenshots_dir / "04_undistorted_right.jpg", first_frame)
    
    # Step 3: Stitch panorama
    run_command(
        [
            'python3', 'scripts/stitching/stitch_apply_transform.py',
            '--left', str(left_undistorted),
            '--right', str(right_undistorted),
            '--calib', args.calib,
            '--output', str(stitched_video),
            '--seam-x', str(args.seam_x),
            '--sync-offset', str(args.sync_offset)
        ],
        "Stitch panorama"
    )
    save_screenshot(str(stitched_video), screenshots_dir / "05_stitched_panorama.jpg", first_frame)
    
    # Step 4: Generate broadcast view
    run_command(
        [
            'python3', 'scripts/detection/broadcast_yolo.py',
            '--video', str(stitched_video),
            '--save-broadcast', str(broadcast_video),
            '--save-preview', str(preview_video)
        ],
        "Generate broadcast view with tracking"
    )
    
    # Save multiple screenshots for broadcast and preview to show tracking quality
    print(f"\n{'='*60}")
    print("Saving broadcast/preview screenshots at multiple frames")
    print(f"{'='*60}")
    for i, frame_num in enumerate(args.screenshot_frames, 1):
        save_screenshot(str(preview_video), screenshots_dir / f"06_broadcast_preview_{i}_frame{frame_num}.jpg", frame_num)
        save_screenshot(str(broadcast_video), screenshots_dir / f"07_broadcast_output_{i}_frame{frame_num}.jpg", frame_num)
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Undistorted videos: {undistorted_dir}")
    print(f"  Stitched panorama: {stitched_video}")
    print(f"  Broadcast view: {broadcast_video}")
    print(f"  Broadcast preview: {preview_video}")
    print(f"  Screenshots: {screenshots_dir}")
    print(f"\nScreenshots saved:")
    print(f"  1. {screenshots_dir / '01_raw_left.jpg'}")
    print(f"  2. {screenshots_dir / '02_raw_right.jpg'}")
    print(f"  3. {screenshots_dir / '03_undistorted_left.jpg'}")
    print(f"  4. {screenshots_dir / '04_undistorted_right.jpg'}")
    print(f"  5. {screenshots_dir / '05_stitched_panorama.jpg'}")
    print(f"  6. Broadcast preview screenshots (frames {args.screenshot_frames}):")
    for i, frame_num in enumerate(args.screenshot_frames, 1):
        print(f"     - {screenshots_dir / f'06_broadcast_preview_{i}_frame{frame_num}.jpg'}")
    print(f"  7. Broadcast output screenshots (frames {args.screenshot_frames}):")
    for i, frame_num in enumerate(args.screenshot_frames, 1):
        print(f"     - {screenshots_dir / f'07_broadcast_output_{i}_frame{frame_num}.jpg'}")


if __name__ == '__main__':
    main()

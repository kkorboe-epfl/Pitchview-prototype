#!/usr/bin/env python3
"""
Live Demo Pipeline: Undistort -> Export Stitch Transform -> Apply Stitch -> Setup Tracking -> Broadcast
Uses the simplified scripts in live_demo_scripts folder.

Usage:
  python scripts/live_demo_scripts/run_live_demo_pipeline.py \
    --left-raw data/raw/leftflip.mp4 \
    --right-raw data/raw/rightflip.mp4 \
    --output-dir output/live_demo
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
        description='Run live demo pipeline: undistort -> export stitch -> apply stitch -> broadcast'
    )
    parser.add_argument('--left-raw', required=True, help='Left raw video path')
    parser.add_argument('--right-raw', required=True, help='Right raw video path')
    parser.add_argument('--output-dir', default='output/live_demo', help='Output directory')
    parser.add_argument('--calib', default='data/calibration/custom_calibration.json',
                       help='Calibration/transform JSON file (default: data/calibration/custom_calibration.json)')
    parser.add_argument('--skip-export-transform', action='store_true',
                       help='Skip stitch transform export (use existing calibration file)')
    parser.add_argument('--skip-tracking-setup', action='store_true',
                       help='Skip interactive ball tracking setup (use existing config)')
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
    calib_json = Path(args.calib)
    stitched_video = stitched_dir / "panorama.mp4"
    broadcast_video = broadcast_dir / "game.mp4"
    preview_video = broadcast_dir / "preview.mp4"
    
    print(f"\n{'='*60}")
    print("LIVE DEMO PIPELINE STARTED")
    print(f"{'='*60}")
    print(f"Left raw: {args.left_raw}")
    print(f"Right raw: {args.right_raw}")
    print(f"Output directory: {output_dir}")
    print(f"Calibration file: {calib_json}")
    
    # Step 0: Save original raw screenshots
    print(f"\n{'='*60}")
    print("STEP 0: Saving original raw screenshots")
    print(f"{'='*60}")
    first_frame = args.screenshot_frames[0]
    save_screenshot(args.left_raw, screenshots_dir / "01_raw_left.jpg", first_frame)
    save_screenshot(args.right_raw, screenshots_dir / "02_raw_right.jpg", first_frame)
    
    # Step 1: Undistort left camera
    run_command(
        [
            'python3', 'scripts/live_demo_scripts/undistort.py',
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
            'python3', 'scripts/live_demo_scripts/undistort.py',
            '--camera', 'right',
            '--input', args.right_raw,
            '--output', str(right_undistorted)
        ],
        "Undistort right camera"
    )
    save_screenshot(str(right_undistorted), screenshots_dir / "04_undistorted_right.jpg", first_frame)
    
    # Step 3: Export stitch transform (if not skipping)
    if not args.skip_export_transform:
        export_output = stitched_dir / "transform.json"
        run_command(
            [
                'python3', 'scripts/live_demo_scripts/stitch_export_transform.py',
                '--left', str(left_undistorted),
                '--right', str(right_undistorted),
                '--save-calib', str(export_output)
            ],
            "Export stitch transformation matrix"
        )
        print(f"Note: Using calibration file {calib_json} for stitching (exported to {export_output} for reference)")
    else:
        print(f"\n{'='*60}")
        print("STEP 3: Skipping stitch transform export")
        print(f"{'='*60}")
    
    # Check calibration file exists
    if not calib_json.exists():
        print(f"Error: Calibration file {calib_json} does not exist!")
        sys.exit(1)
    
    # Step 4: Apply stitch transform to create panorama
    run_command(
        [
            'python3', 'scripts/live_demo_scripts/stitch_apply_transform.py',
            '--left', str(left_undistorted),
            '--right', str(right_undistorted),
            '--calib', str(calib_json),
            '--output', str(stitched_video),
            '--seam-x', '3945',
            '--left-alpha', '1.0',
            '--edge-blend', '100'
        ],
        "Apply stitch transformation to create panorama"
    )
    save_screenshot(str(stitched_video), screenshots_dir / "05_stitched_panorama.jpg", first_frame)
    
    # Step 5: Setup ball tracking configuration (if not skipping)
    if not args.skip_tracking_setup:
        print(f"\n{'='*60}")
        print("STEP 5: Ball Tracking Configuration")
        print(f"{'='*60}")
        print("Interactive setup will open:")
        print("  1. Draw polygon around the playing field (click points, press 'c' to close)")
        print("  2. Click on the ball's initial position")
        print("  3. Press 'q' to save and continue")
        print()
        
        run_command(
            [
                'python3', 'scripts/detection/setup_ball_tracking.py',
                str(stitched_video)
            ],
            "Configure ball tracking (field boundary + initial ball position)"
        )
    else:
        print(f"\n{'='*60}")
        print("STEP 5: Skipping ball tracking setup (using existing config)")
        print(f"{'='*60}")
    
    # Step 6: Generate broadcast view with advanced tracking
    run_command(
        [
            'python3', 'scripts/detection/broadcast.py',
            '--video', str(stitched_video),
            '--save-broadcast', str(broadcast_video),
            '--save-preview', str(preview_video)
        ],
        "Generate broadcast view with advanced tracking"
    )
    
    # Save multiple screenshots for broadcast and preview
    print(f"\n{'='*60}")
    print("Saving broadcast/preview screenshots at multiple frames")
    print(f"{'='*60}")
    for i, frame_num in enumerate(args.screenshot_frames, 1):
        save_screenshot(str(preview_video), screenshots_dir / f"06_broadcast_preview_{i}_frame{frame_num}.jpg", frame_num)
        save_screenshot(str(broadcast_video), screenshots_dir / f"07_broadcast_output_{i}_frame{frame_num}.jpg", frame_num)
    
    # Summary
    print(f"\n{'='*60}")
    print("LIVE DEMO PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Undistorted videos: {undistorted_dir}")
    print(f"    - Left: {left_undistorted}")
    print(f"    - Right: {right_undistorted}")
    print(f"  Calibration used: {calib_json}")
    print(f"  Stitched panorama: {stitched_video}")
    print(f"  Ball tracking config: data/calibration/ball_tracking_config.json")
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

#!/usr/bin/env python3
"""
Undistort fisheye video using camera-specific calibration parameters.

Usage:
  python scripts/undistort_video.py --camera left --input video.mp4 --output undistorted.mp4
  python scripts/undistort_video.py --camera right --input video.mp4 --output undistorted.mp4
"""
import argparse
import numpy as np
import cv2
import sys

# Calibration parameters for left camera
LEFT_CAMERA = {
    'DIM': (2560, 1440),
    'K': np.array([[1377.757728301297, 0.0, 1267.1200591051727], 
                   [0.0, 1376.4516881783097, 757.9649350021892], 
                   [0.0, 0.0, 1.0]]),
    'D': np.array([[0.035718550519243566], 
                   [-0.05599207629437077], 
                   [0.10681473096471601], 
                   [-0.080901595667908]])
}

# Calibration parameters for right camera
RIGHT_CAMERA = {
    'DIM': (2560, 1440),
    'K': np.array([[1389.291034402298, 0.0, 1262.529547048051], 
                   [0.0, 1389.8736144967868, 751.059363279825], 
                   [0.0, 0.0, 1.0]]),
    'D': np.array([[0.0091828800754966], 
                   [0.03389379247052483], 
                   [-0.02339525205416778], 
                   [-0.024336675312163564]])
}


def undistort_video(input_path, output_path, camera_params):
    """Undistort a fisheye video using the provided camera parameters."""
    DIM = camera_params['DIM']
    K = camera_params['K']
    D = camera_params['D']
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Scale output dimensions to avoid cropping
    scale_factor = 2
    out_width = int(width * scale_factor)
    out_height = int(height * scale_factor)
    out_dim = (out_width, out_height)
    
    print(f"Output will be scaled to: {out_width}x{out_height}")
    
    # Scale the camera matrix to match new output dimensions
    scaled_K = K.copy()
    scaled_K[0, 2] = out_width / 2   # cx - center x
    scaled_K[1, 2] = out_height / 2  # cy - center y
    
    # Precompute undistortion maps (much faster than computing per frame)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), scaled_K, out_dim, cv2.CV_16SC2
    )
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'H264'
    out = cv2.VideoWriter(output_path, fourcc, fps, out_dim)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Apply undistortion
        undistorted = cv2.remap(
            frame, map1, map2, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT
        )
        
        # Write frame
        out.write(undistorted)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames", end='\r')
    
    print(f"\nCompleted! Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Saved undistorted video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Undistort fisheye video using camera-specific calibration'
    )
    parser.add_argument('--camera', '-c', 
                        choices=['left', 'right'], 
                        required=True,
                        help='Which camera calibration to use (left or right)')
    parser.add_argument('--input', '-i', 
                        required=True,
                        help='Input video file path')
    parser.add_argument('--output', '-o', 
                        required=True,
                        help='Output video file path')
    
    args = parser.parse_args()
    
    # Select camera parameters
    camera_params = LEFT_CAMERA if args.camera == 'left' else RIGHT_CAMERA
    print(f"Using {args.camera} camera calibration")
    
    # Undistort the video
    undistort_video(args.input, args.output, camera_params)


if __name__ == '__main__':
    main()

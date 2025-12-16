import numpy as np
import cv2
import os
import argparse
import sys
import json

def load_camera_calibration(calib_path='data/calibration/camera_calibration.json'):
    """
    Load camera calibration from JSON file.
    
    Tutorial for creating calibration: 
    https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    Note that you may need to adjust the number of calibration images and chessboard size.q
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"Camera calibration file not found: {calib_path}\n"
            f"Please create this file using OpenCV fisheye calibration."
        )
    
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    # Convert lists to numpy arrays
    left_camera = {
        'DIM': tuple(calib_data['left_camera']['DIM']),
        'K': np.array(calib_data['left_camera']['K'], dtype=np.float64),
        'D': np.array(calib_data['left_camera']['D'], dtype=np.float64)
    }
    
    right_camera = {
        'DIM': tuple(calib_data['right_camera']['DIM']),
        'K': np.array(calib_data['right_camera']['K'], dtype=np.float64),
        'D': np.array(calib_data['right_camera']['D'], dtype=np.float64)
    }
    
    return left_camera, right_camera

def undistort_video(input_path, output_path, camera_params, focal_scale=0.6):
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return False
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {input_path}")
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    K_scaled = camera_params['K'].copy()
    K_scaled[0, 0] *= focal_scale
    K_scaled[1, 1] *= focal_scale
    
    # Precompute undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_params['K'], 
        camera_params['D'], 
        np.eye(3), 
        K_scaled,  # Use scaled K for output
        camera_params['DIM'], 
        cv2.CV_16SC2
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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
    print(f"Saved to: {output_path}\n")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Undistort fisheye camera videos')
    parser.add_argument('--left-input', default='data/raw/leftflip.mp4',
                        help='Path to left camera input video')
    parser.add_argument('--right-input', default='data/raw/rightflip.mp4',
                        help='Path to right camera input video')
    parser.add_argument('--left-output', default='data/undistorted/left.mp4',
                        help='Path to save undistorted left video')
    parser.add_argument('--right-output', default='data/undistorted/right.mp4',
                        help='Path to save undistorted right video')
    parser.add_argument('--focal-scale', type=float, default=0.6,
                        help='Focal length scaling factor (lower = wider FOV, default: 0.6)')
    parser.add_argument('--calibration', default='data/calibration/camera_calibration.json',
                        help='Path to camera calibration JSON file')
    args = parser.parse_args()
    
    # Load camera calibration
    LEFT_CAMERA, RIGHT_CAMERA = load_camera_calibration(args.calibration)
    
    print("=" * 50)
    print("Processing LEFT camera")
    print("=" * 50)
    success_left = undistort_video(args.left_input, args.left_output, LEFT_CAMERA, args.focal_scale)
    
    print("=" * 50)
    print("Processing RIGHT camera")
    print("=" * 50)
    success_right = undistort_video(args.right_input, args.right_output, RIGHT_CAMERA, args.focal_scale)
    
    print("=" * 50)
    if success_left and success_right:
        print("Both videos undistorted successfully!")
        sys.exit(0)
    else:
        print("Some videos failed to process.")
        sys.exit(1)
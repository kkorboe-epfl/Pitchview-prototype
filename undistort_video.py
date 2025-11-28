import numpy as np
import cv2
import sys

# Calibration parameters from your calibration
DIM = (2560, 1440)
K = np.array([[1377.757728301297, 0.0, 1267.1200591051727], 
              [0.0, 1376.4516881783097, 757.9649350021892], 
              [0.0, 0.0, 1.0]])
D = np.array([[0.035718550519243566], 
              [-0.05599207629437077], 
              [0.10681473096471601], 
              [-0.080901595667908]])

def undistort_video(input_path, output_path):
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
    
    # Precompute undistortion maps (much faster than computing per frame)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, DIM, cv2.CV_16SC2
    )
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'H264'
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
    print(f"Saved undistorted video to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 undistort_video.py input_video.mp4 [output_video.mp4]")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "undistorted_" + input_video
    
    undistort_video(input_video, output_video)
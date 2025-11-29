import numpy as np
import cv2
import os

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
    # Open input video
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
    
    # Use scaled K matrix to see all pixels (no zoom)
    # This keeps the full field of view with black borders
    K_scaled = camera_params['K'].copy()
    K_scaled[0, 0] *= 0.6  # Scale focal length
    K_scaled[1, 1] *= 0.6
    
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
    # Input and output paths
    left_input = "data/raw/leftflip.mp4"
    right_input = "data/raw/rightflip.mp4"
    left_output = "data/undistorted/left.mp4"
    right_output = "data/undistorted/right.mp4"
    
    # Process left camera
    print("=" * 50)
    print("Processing LEFT camera")
    print("=" * 50)
    success_left = undistort_video(left_input, left_output, LEFT_CAMERA)
    
    # Process right camera
    print("=" * 50)
    print("Processing RIGHT camera")
    print("=" * 50)
    success_right = undistort_video(right_input, right_output, RIGHT_CAMERA)
    
    # Summary
    print("=" * 50)
    if success_left and success_right:
        print("Both videos undistorted successfully!")
    else:
        print("Some videos failed to process.")
import argparse
import json
import os
import numpy as np
import cv2
from tqdm import tqdm


def apply_horizontal_stretch(image, stretch_factor, from_right=False):
    """
    Apply vertical scaling at inner edge.
    stretch_factor < 1.0 shrinks height at inner edge
    stretch_factor > 1.0 expands height at inner edge
    """
    if stretch_factor == 1.0:
        return image
    
    h, w = image.shape[:2]
    
    # Create coordinate maps
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(x_coords, y_coords)
    
    # For each column, calculate the stretch factor
    for x in range(w):
        if from_right:
            # Right image: left edge (inner) is stretched, right edge stays same
            gradient = 1.0 - (x / (w - 1))
        else:
            # Left image: right edge (inner) is stretched, left edge stays same
            gradient = x / (w - 1)
        
        # Local stretch for this column
        local_stretch = 1.0 + gradient * (stretch_factor - 1.0)
        
        # Remap y coordinates
        center_y = h / 2
        map_y[:, x] = center_y + (y_coords - center_y) / local_stretch
    
    # Clamp to valid range
    map_y = np.clip(map_y, 0, h - 1)
    
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return result


def load_calibration(path):
    """Load calibration from JSON file."""
    with open(path, 'r') as f:
        calib = json.load(f)
    return calib


def get_transform_matrix(calib, right_shape):
    """Create transformation matrix from calibration."""
    h, w = right_shape[:2]
    center = (w / 2, h / 2)
    
    # Rotation and scale matrix around center
    M = cv2.getRotationMatrix2D(center, calib['rotation'], calib['scale'])
    
    # Add translation
    M[0, 2] += calib['x_offset']
    M[1, 2] += calib['y_offset']
    
    return M


def stitch_frame(frame_left, frame_right, calib, transform_matrix, seam_x_top=None, seam_x_bottom=None, feather_width=50):
    """Stitch a single frame pair with adjustable vertical seam and feathering."""
    # Apply stretch
    inner_stretch = calib.get('inner_stretch', 1.0)
    left_stretched = apply_horizontal_stretch(frame_left, inner_stretch, from_right=False)
    right_stretched = apply_horizontal_stretch(frame_right, inner_stretch, from_right=True)
    
    # Get canvas dimensions
    width = calib['canvas_width']
    height = calib['canvas_height']
    offset_x, offset_y = calib['offset']
    
    # Create canvases for left and right
    canvas_left = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_right = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place left image
    h_left, w_left = left_stretched.shape[:2]
    y1 = offset_y
    y2 = min(offset_y + h_left, height)
    x1 = offset_x
    x2 = min(offset_x + w_left, width)
    canvas_left[y1:y2, x1:x2] = left_stretched[:y2-y1, :x2-x1]
    
    # Transform and place right image
    M = transform_matrix.copy()
    M[0, 2] += offset_x
    M[1, 2] += offset_y
    
    canvas_right = cv2.warpAffine(right_stretched, M, (width, height),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
    
    # Find overlap region
    left_mask = (canvas_left > 0).any(axis=2)
    right_mask = (canvas_right > 0).any(axis=2)
    overlap_mask = left_mask & right_mask
    
    # Auto-detect seam position if not specified
    if seam_x_top is None or seam_x_bottom is None:
        # Find the middle of the overlap region
        overlap_cols = np.where(overlap_mask.any(axis=0))[0]
        if len(overlap_cols) > 0:
            default_seam = int(np.median(overlap_cols))
        else:
            default_seam = width // 2
        
        if seam_x_top is None:
            seam_x_top = default_seam
        if seam_x_bottom is None:
            seam_x_bottom = default_seam
    
    # Create blending masks with feathering
    blend_left = np.ones((height, width), dtype=np.float32)
    blend_right = np.ones((height, width), dtype=np.float32)
    
    # Apply feathering around the seam (which can be slanted)
    for y in range(height):
        # Interpolate seam position based on y coordinate
        alpha_y = y / (height - 1) if height > 1 else 0
        seam_x = seam_x_top * (1 - alpha_y) + seam_x_bottom * alpha_y
        
        for x in range(width):
            distance_from_seam = x - seam_x
            
            if distance_from_seam < -feather_width:
                # Left side of feather zone - fully left
                blend_left[y, x] = 1.0
                blend_right[y, x] = 0.0
            elif distance_from_seam > feather_width:
                # Right side of feather zone - fully right
                blend_left[y, x] = 0.0
                blend_right[y, x] = 1.0
            else:
                # In feather zone - smooth transition
                alpha = 0.5 - (distance_from_seam / (2 * feather_width))
                alpha = np.clip(alpha, 0.0, 1.0)
                blend_left[y, x] = alpha
                blend_right[y, x] = 1.0 - alpha
    
    # Apply masks - only blend where both images exist
    blend_left = blend_left[:, :, np.newaxis]
    blend_right = blend_right[:, :, np.newaxis]
    
    # Blend the images
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    
    # Where both exist, blend
    both_exist = overlap_mask[:, :, np.newaxis]
    canvas = np.where(both_exist,
                     canvas_left.astype(np.float32) * blend_left + 
                     canvas_right.astype(np.float32) * blend_right,
                     canvas)
    
    # Where only left exists
    only_left = (left_mask & ~right_mask)[:, :, np.newaxis]
    canvas = np.where(only_left, canvas_left.astype(np.float32), canvas)
    
    # Where only right exists
    only_right = (right_mask & ~left_mask)[:, :, np.newaxis]
    canvas = np.where(only_right, canvas_right.astype(np.float32), canvas)
    
    return canvas.astype(np.uint8), seam_x_top, seam_x_bottom


def interactive_seam_adjustment(cap_left, cap_right, calib, transform_matrix, feather_width):
    """Interactive mode to adjust seam line."""
    print("\n" + "=" * 60)
    print("INTERACTIVE SEAM ADJUSTMENT")
    print("=" * 60)
    print("Controls:")
    print("  A/D - Move top of seam left/right")
    print("  Z/C - Move bottom of seam left/right")
    print("  [/] - Fine/Coarse adjustment mode")
    print("  SPACE - Next frame")
    print("  BACKSPACE - Previous frame")
    print("  S - Save seam configuration")
    print("  ESC - Exit")
    print("=" * 60)
    
    # Get first frame
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        print("Error: Could not read frames")
        return None, None
    
    height = calib['canvas_height']
    
    # Auto-detect initial seam
    _, seam_x_top, seam_x_bottom = stitch_frame(frame_left, frame_right, calib, 
                                                 transform_matrix, None, None, feather_width)
    
    window_name = "Seam Adjustment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)
    
    frame_idx = 0
    total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    step_size = 5
    coarse_mode = False
    
    while True:
        # Stitch current frame
        stitched, _, _ = stitch_frame(frame_left, frame_right, calib, transform_matrix,
                                      seam_x_top, seam_x_bottom, feather_width)
        
        # Draw seam line
        display = stitched.copy()
        
        # Draw gradient seam line
        for y in range(0, height, 10):
            alpha_y = y / (height - 1) if height > 1 else 0
            seam_x = int(seam_x_top * (1 - alpha_y) + seam_x_bottom * alpha_y)
            next_y = min(y + 10, height - 1)
            next_alpha_y = next_y / (height - 1) if height > 1 else 0
            next_seam_x = int(seam_x_top * (1 - next_alpha_y) + seam_x_bottom * next_alpha_y)
            cv2.line(display, (seam_x, y), (next_seam_x, next_y), (0, 255, 0), 2)
        
        # Draw markers at top and bottom
        cv2.circle(display, (int(seam_x_top), 20), 10, (0, 255, 255), -1)
        cv2.circle(display, (int(seam_x_bottom), height - 20), 10, (255, 0, 255), -1)
        
        # Add info text
        info_text = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Seam Top: {seam_x_top:.0f}px (Yellow)",
            f"Seam Bottom: {seam_x_bottom:.0f}px (Magenta)",
            f"Feather: {feather_width}px",
            f"Mode: {'COARSE' if coarse_mode else 'FINE'}",
            "",
            "A/D: Move top | Z/C: Move bottom"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(display, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:
            continue
        
        step = step_size * 10 if coarse_mode else step_size
        needs_update = False
        
        if key == 27:  # ESC
            print("\nExiting without saving")
            cv2.destroyAllWindows()
            return None, None
        
        elif key == ord('s') or key == ord('S'):
            print(f"\nSeam configuration saved:")
            print(f"  Top: {seam_x_top:.0f}px")
            print(f"  Bottom: {seam_x_bottom:.0f}px")
            cv2.destroyAllWindows()
            return seam_x_top, seam_x_bottom
        
        elif key == ord('['):
            coarse_mode = False
            print("Fine adjustment mode")
        elif key == ord(']'):
            coarse_mode = True
            print("Coarse adjustment mode")
        
        # Adjust top of seam
        elif key == ord('a') or key == ord('A'):
            seam_x_top -= step
            print(f"Seam top: {seam_x_top:.0f}px")
        elif key == ord('d') or key == ord('D'):
            seam_x_top += step
            print(f"Seam top: {seam_x_top:.0f}px")
        
        # Adjust bottom of seam
        elif key == ord('z') or key == ord('Z'):
            seam_x_bottom -= step
            print(f"Seam bottom: {seam_x_bottom:.0f}px")
        elif key == ord('c') or key == ord('C'):
            seam_x_bottom += step
            print(f"Seam bottom: {seam_x_bottom:.0f}px")
        
        # Navigate frames
        elif key == ord(' '):  # Space - next frame
            frame_idx = min(frame_idx + 1, total_frames - 1)
            cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            print(f"Frame: {frame_idx}")
        
        elif key == 8:  # Backspace - previous frame
            frame_idx = max(frame_idx - 1, 0)
            cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            print(f"Frame: {frame_idx}")


def find_crop_bounds(frame):
    """Find the largest rectangle without black borders."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find non-black pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0, frame.shape[1], frame.shape[0]
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return x, y, w, h


def auto_crop_frame(frame, crop_bounds=None):
    """Crop frame to remove black borders."""
    if crop_bounds is None:
        x, y, w, h = find_crop_bounds(frame)
    else:
        x, y, w, h = crop_bounds
    
    return frame[y:y+h, x:x+w], (x, y, w, h)


def interactive_crop_adjustment(frame, angle, target_width, target_height):
    """Interactive mode to adjust crop center position."""
    print("\n" + "=" * 60)
    print("INTERACTIVE CROP ADJUSTMENT")
    print("=" * 60)
    print("Controls:")
    print("  Arrow Keys / W/A/S/D - Move crop center")
    print("  [/] - Fine/Coarse adjustment mode")
    print("  R - Reset to center")
    print("  ENTER - Save and continue")
    print("  ESC - Cancel")
    print("=" * 60)
    
    h, w = frame.shape[:2]
    
    # Rotate frame first
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(frame, rotation_matrix, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    
    # Start at center
    center_x = new_w // 2
    center_y = new_h // 2
    
    window_name = "Crop Position Adjustment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)
    
    step_size = 5
    coarse_mode = False
    
    while True:
        # Calculate crop bounds
        x1 = max(0, center_x - target_width // 2)
        y1 = max(0, center_y - target_height // 2)
        x2 = min(new_w, x1 + target_width)
        y2 = min(new_h, y1 + target_height)
        
        # Adjust if we're at edges
        if x2 - x1 < target_width:
            x1 = max(0, x2 - target_width)
        if y2 - y1 < target_height:
            y1 = max(0, y2 - target_height)
        
        # Create display
        display = rotated.copy()
        
        # Draw crop rectangle
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw center crosshair
        crosshair_size = 50
        cv2.line(display, (center_x - crosshair_size, center_y), 
                (center_x + crosshair_size, center_y), (0, 255, 255), 2)
        cv2.line(display, (center_x, center_y - crosshair_size), 
                (center_x, center_y + crosshair_size), (0, 255, 255), 2)
        
        # Add info text
        info_text = [
            f"Center: ({center_x}, {center_y})",
            f"Crop: {target_width}x{target_height}",
            f"Mode: {'COARSE' if coarse_mode else 'FINE'}",
            "",
            "Move with arrows/WASD",
            "ENTER to save"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(display, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Scale for display if needed
        display_h, display_w = display.shape[:2]
        if display_w > 1600:
            scale = 1600 / display_w
            display = cv2.resize(display, (1600, int(display_h * scale)))
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:
            continue
        
        step = step_size * 10 if coarse_mode else step_size
        
        if key == 27:  # ESC
            print("\nCancelled")
            cv2.destroyAllWindows()
            return None, None
        
        elif key == 13:  # ENTER
            print(f"\nCrop center saved: ({center_x}, {center_y})")
            cv2.destroyAllWindows()
            return center_x, center_y
        
        elif key == ord('r') or key == ord('R'):
            center_x = new_w // 2
            center_y = new_h // 2
            print("Reset to center")
        
        elif key == ord('['):
            coarse_mode = False
            print("Fine adjustment mode")
        elif key == ord(']'):
            coarse_mode = True
            print("Coarse adjustment mode")
        
        # Movement with WASD
        elif key == ord('w') or key == ord('W') or key == 82:  # W or Up arrow
            center_y -= step
            print(f"Center: ({center_x}, {center_y})")
        elif key == ord('s') or key == ord('S') or key == 84:  # S or Down arrow
            center_y += step
            print(f"Center: ({center_x}, {center_y})")
        elif key == ord('a') or key == ord('A') or key == 81:  # A or Left arrow
            center_x -= step
            print(f"Center: ({center_x}, {center_y})")
        elif key == ord('d') or key == ord('D') or key == 83:  # D or Right arrow
            center_x += step
            print(f"Center: ({center_x}, {center_y})")


def rotate_and_crop(frame, angle=-15, target_width=4000, target_height=1000, center_x=None, center_y=None):
    """Rotate frame anticlockwise and crop to specified dimensions at custom center."""
    h, w = frame.shape[:2]
    
    # Get rotation matrix around center
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions after rotation
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Rotate the image
    rotated = cv2.warpAffine(frame, rotation_matrix, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    
    # Use custom center or default to middle
    if center_x is None:
        center_x = new_w // 2
    if center_y is None:
        center_y = new_h // 2
    
    # Calculate crop bounds centered at custom position
    x1 = max(0, center_x - target_width // 2)
    y1 = max(0, center_y - target_height // 2)
    x2 = min(new_w, x1 + target_width)
    y2 = min(new_h, y1 + target_height)
    
    # Adjust if we're at edges
    if x2 - x1 < target_width:
        x1 = max(0, x2 - target_width)
    if y2 - y1 < target_height:
        y1 = max(0, y2 - target_height)
    
    # Crop
    cropped = rotated[y1:y2, x1:x2]
    
    # Pad if necessary to ensure exact dimensions
    if cropped.shape[0] < target_height or cropped.shape[1] < target_width:
        pad_h = max(0, target_height - cropped.shape[0])
        pad_w = max(0, target_width - cropped.shape[1])
        cropped = cv2.copyMakeBorder(cropped, 
                                     pad_h // 2, pad_h - pad_h // 2,
                                     pad_w // 2, pad_w - pad_w // 2,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return cropped


def process_videos(left_path, right_path, calib_path, output_path, preview=False, 
                  seam_x_top=None, seam_x_bottom=None, feather_width=50, interactive=False,
                  rotate_angle=-15, crop_width=4000, crop_height=1000, 
                  interactive_crop=False, crop_center_x=None, crop_center_y=None):
    """Process full videos with calibration."""
    print("\n" + "=" * 60)
    print("APPLYING STITCH CALIBRATION")
    print("=" * 60)
    
    # Load calibration
    print(f"Loading calibration: {calib_path}")
    calib = load_calibration(calib_path)
    print(f"  Position: ({calib['x_offset']}, {calib['y_offset']})")
    print(f"  Rotation: {calib['rotation']:.2f}°")
    print(f"  Scale: {calib['scale']:.3f}")
    print(f"  Inner stretch: {calib.get('inner_stretch', 1.0):.3f}")
    print(f"  Canvas: {calib['canvas_width']}x{calib['canvas_height']}")
    print(f"  Feather width: {feather_width}px")
    print(f"  Rotate: {rotate_angle}°")
    print(f"  Crop: {crop_width}x{crop_height}px")
    
    # Open videos
    print(f"\nOpening videos...")
    print(f"  Left:  {left_path}")
    print(f"  Right: {right_path}")
    
    cap_left = cv2.VideoCapture(left_path)
    cap_right = cv2.VideoCapture(right_path)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open videos")
        return
    
    # Get video properties
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Read first frame to get transform matrix
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        print("Error: Could not read frames")
        return
    
    transform_matrix = get_transform_matrix(calib, frame_right.shape)
    
    # Interactive seam adjustment mode
    if interactive:
        cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
        seam_x_top, seam_x_bottom = interactive_seam_adjustment(
            cap_left, cap_right, calib, transform_matrix, feather_width)
        
        if seam_x_top is None:
            return
    
    # Get seam position from first frame if auto
    if seam_x_top is None or seam_x_bottom is None:
        _, detected_top, detected_bottom = stitch_frame(
            frame_left, frame_right, calib, transform_matrix, 
            None, None, feather_width)
        seam_x_top = seam_x_top or detected_top
        seam_x_bottom = seam_x_bottom or detected_bottom
        print(f"\nDetected seam: top={seam_x_top:.0f}px, bottom={seam_x_bottom:.0f}px")
    else:
        print(f"\nUsing seam: top={seam_x_top:.0f}px, bottom={seam_x_bottom:.0f}px")
    
    # Interactive crop center adjustment
    if interactive_crop:
        cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        # Stitch first frame
        stitched, _, _ = stitch_frame(frame_left, frame_right, calib, transform_matrix,
                                     seam_x_top, seam_x_bottom, feather_width)
        
        crop_center_x, crop_center_y = interactive_crop_adjustment(
            stitched, rotate_angle, crop_width, crop_height)
        
        if crop_center_x is None:
            return
    
    if crop_center_x is not None and crop_center_y is not None:
        print(f"\nUsing crop center: ({crop_center_x}, {crop_center_y})")
    else:
        print(f"\nUsing default crop center (middle)")
    
    # Reset to beginning
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if preview:
        print("\n" + "=" * 60)
        print("PREVIEW MODE")
        print("=" * 60)
        print("Press SPACE to see next frame")
        print("Press ESC to exit preview")
        print("=" * 60)
        
        window_name = "Stitched Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        
        frame_count = 0
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            
            if not ret_left or not ret_right:
                print("\nReached end of video")
                break
            
            # Stitch frame
            stitched, _, _ = stitch_frame(frame_left, frame_right, calib, transform_matrix,
                                         seam_x_top, seam_x_bottom, feather_width)
            
            # Rotate and crop
            final = rotate_and_crop(stitched, rotate_angle, crop_width, crop_height,
                                   crop_center_x, crop_center_y)
            
            # Add info overlay
            display = final.copy()
            cv2.putText(display, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Size: {final.shape[1]}x{final.shape[0]}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Rotated: {rotate_angle}deg", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break
            
            frame_count += 1
        
        cv2.destroyAllWindows()
    
    else:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))
        
        print(f"\nProcessing video...")
        print(f"  Output: {output_path}")
        print(f"  Output size: {crop_width}x{crop_height}")
        
        # Process all frames
        with tqdm(total=total_frames, desc="Stitching", unit="frames") as pbar:
            while True:
                ret_left, frame_left = cap_left.read()
                ret_right, frame_right = cap_right.read()
                
                if not ret_left or not ret_right:
                    break
                
                # Stitch frame
                stitched, _, _ = stitch_frame(frame_left, frame_right, calib, transform_matrix,
                                             seam_x_top, seam_x_bottom, feather_width)
                
                # Rotate and crop
                final = rotate_and_crop(stitched, rotate_angle, crop_width, crop_height,
                                       crop_center_x, crop_center_y)
                
                # Write frame
                out.write(final)
                
                pbar.update(1)
        
        out.release()
        
        print("\n" + "=" * 60)
        print("COMPLETE!")
        print("=" * 60)
        print(f"Stitched video saved to: {output_path}")
        print(f"Output dimensions: {crop_width}x{crop_height}")
        print(f"Total frames processed: {total_frames}")
    
    cap_left.release()
    cap_right.release()


def main():
    parser = argparse.ArgumentParser(
        description='Apply stitching calibration to full videos'
    )
    parser.add_argument('--left', default='data/undistorted/left.mp4',
                        help='Left video path')
    parser.add_argument('--right', default='data/undistorted/right.mp4',
                        help='Right video path')
    parser.add_argument('--calib', default='data/calibration/manual_stitch_calibration.json',
                        help='Calibration JSON path')
    parser.add_argument('--output', default='output/stitched/panorama.mp4',
                        help='Output video path')
    parser.add_argument('--preview', action='store_true',
                        help='Preview mode (step through frames)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode to adjust seam line')
    parser.add_argument('--interactive-crop', action='store_true',
                        help='Interactive mode to adjust crop center')
    parser.add_argument('--seam-top', type=int, default=None,
                        help='Seam x-position at top')
    parser.add_argument('--seam-bottom', type=int, default=None,
                        help='Seam x-position at bottom')
    parser.add_argument('--feather', type=int, default=50,
                        help='Feathering width in pixels (default: 50)')
    parser.add_argument('--rotate', type=float, default=3,
                        help='Rotation angle in degrees (default: 3)')
    parser.add_argument('--crop-width', type=int, default=3800,
                        help='Final crop width (default: 3800)')
    parser.add_argument('--crop-height', type=int, default=500,
                        help='Final crop height (default: 600)')
    parser.add_argument('--crop-center-x', type=int, default=None,
                        help='Crop center x position')
    parser.add_argument('--crop-center-y', type=int, default=850,
                        help='Crop center y position')
    
    args = parser.parse_args()
    
    process_videos(args.left, args.right, args.calib, args.output, 
                  args.preview, args.seam_top, args.seam_bottom, args.feather, 
                  args.interactive, args.rotate, args.crop_width, args.crop_height,
                  args.interactive_crop, args.crop_center_x, args.crop_center_y)


if __name__ == '__main__':
    main()
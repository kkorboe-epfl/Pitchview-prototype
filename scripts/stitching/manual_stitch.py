import argparse
import json
import os
import numpy as np
import cv2


class ManualStitcher:
    def __init__(self, left_image, right_image, inner_stretch=1.0):
        self.left = left_image.copy()
        self.right = right_image.copy()
        
        # Initial position: side by side
        self.x_offset = left_image.shape[1]  # Start right image next to left
        self.y_offset = 0
        self.rotation = 0.0  # degrees
        self.scale = 1.0
        self.opacity = 0.5  # 0.0 = only left, 1.0 = only right
        
        # Edge stretch correction - SET THIS VALUE
        self.inner_stretch = inner_stretch
        
        # UI settings
        self.step_size = 10  # pixels
        self.rotation_step = 0.5  # degrees
        self.scale_step = 0.01
        self.opacity_step = 0.05
        
        print("\n" + "=" * 60)
        print("MANUAL STITCHING CONTROLS")
        print("=" * 60)
        print("Position:")
        print("  W/A/S/D       - Move right image")
        print("  [ / ]         - Fine/Coarse mode")
        print("\nRotation:")
        print("  Q / E         - Rotate left/right")
        print("\nScale:")
        print("  Z / X         - Scale down/up")
        print("\nOpacity:")
        print("  - / =         - Decrease/Increase right image opacity")
        print("\nOther:")
        print("  R             - Reset to initial position")
        print("  Shift+S       - Save calibration")
        print("  ESC           - Exit without saving")
        print("=" * 60)
        print(f"\nInitial position: x={self.x_offset}, y={self.y_offset}")
        print(f"Initial rotation: {self.rotation}°")
        print(f"Initial scale: {self.scale}")
        print(f"Initial opacity: {self.opacity}")
        print(f"Inner stretch (FIXED): {self.inner_stretch}")
        print()
    
    def apply_horizontal_stretch(self, image, stretch_factor, from_right=False):
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
    
    def get_transform_matrix(self):
        """Create transformation matrix for current position/rotation/scale."""
        h, w = self.right.shape[:2]
        center = (w / 2, h / 2)
        
        # Rotation and scale matrix around center
        M_rot_scale = cv2.getRotationMatrix2D(center, self.rotation, self.scale)
        
        # Add translation
        M_rot_scale[0, 2] += self.x_offset
        M_rot_scale[1, 2] += self.y_offset
        
        return M_rot_scale
    
    def calculate_canvas_size(self):
        """Calculate canvas size to fit both images."""
        # Use stretched versions for canvas calculation
        left_stretched = self.apply_horizontal_stretch(self.left, self.inner_stretch, from_right=False)
        right_stretched = self.apply_horizontal_stretch(self.right, self.inner_stretch, from_right=True)
        
        h_left, w_left = left_stretched.shape[:2]
        h_right, w_right = right_stretched.shape[:2]
        
        # Corners of left image
        corners_left = np.float32([
            [0, 0],
            [w_left, 0],
            [w_left, h_left],
            [0, h_left]
        ])
        
        # Corners of right image
        corners_right = np.float32([
            [0, 0],
            [w_right, 0],
            [w_right, h_right],
            [0, h_right]
        ])
        
        # Transform right corners
        M = self.get_transform_matrix()
        corners_right_transformed = cv2.transform(corners_right.reshape(-1, 1, 2), M).reshape(-1, 2)
        
        # Combine all corners
        all_corners = np.vstack([corners_left, corners_right_transformed])
        
        # Find bounding box
        x_min = int(np.floor(all_corners[:, 0].min()))
        y_min = int(np.floor(all_corners[:, 1].min()))
        x_max = int(np.ceil(all_corners[:, 0].max()))
        y_max = int(np.ceil(all_corners[:, 1].max()))
        
        width = x_max - x_min
        height = y_max - y_min
        offset = (-x_min, -y_min)
        
        return width, height, offset
    
    def render(self):
        """Render current view with both images."""
        width, height, offset = self.calculate_canvas_size()
        offset_x, offset_y = offset
        
        # Apply stretch to both images
        left_stretched = self.apply_horizontal_stretch(self.left, self.inner_stretch, from_right=False)
        right_stretched = self.apply_horizontal_stretch(self.right, self.inner_stretch, from_right=True)
        
        # Create canvas
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        
        # Place left image
        h_left, w_left = left_stretched.shape[:2]
        y1 = offset_y
        y2 = min(offset_y + h_left, height)
        x1 = offset_x
        x2 = min(offset_x + w_left, width)
        canvas[y1:y2, x1:x2] = left_stretched[:y2-y1, :x2-x1].astype(np.float32)
        
        # Transform and place right image
        M = self.get_transform_matrix()
        M[0, 2] += offset_x
        M[1, 2] += offset_y
        
        right_warped = cv2.warpAffine(right_stretched, M, (width, height), 
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))
        
        # Blend with opacity control
        right_mask = (right_warped > 0).any(axis=2, keepdims=True)
        
        # Where right image exists, blend based on opacity
        canvas = np.where(right_mask, 
                         canvas * (1 - self.opacity) + right_warped.astype(np.float32) * self.opacity,
                         canvas)
        
        return canvas.astype(np.uint8), (width, height, offset)
    
    def scale_for_display(self, image, max_width=1600):
        """Scale image for display if needed."""
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h)), scale
        return image, 1.0
    
    def run(self):
        """Run interactive stitching loop."""
        window_name = "Manual Stitching"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 800)
        
        saved_calib = None
        shift_pressed = False
        needs_render = True
        
        print("\n*** CLICK ON THE OPENCV WINDOW TO ACTIVATE IT ***\n")
        
        while True:
            if needs_render:
                try:
                    canvas, (width, height, offset) = self.render()
                    display, display_scale = self.scale_for_display(canvas)
                    
                    info_text = [
                        f"Position: ({self.x_offset}, {self.y_offset})",
                        f"Rotation: {self.rotation:.2f}deg",
                        f"Scale: {self.scale:.3f}",
                        f"Opacity: {self.opacity:.2f}",
                        f"Canvas: {width}x{height}",
                        f"Mode: {'Coarse' if shift_pressed else 'Fine'}",
                        "",
                        "Press Shift+S to save, ESC to exit"
                    ]
                    
                    for i, text in enumerate(info_text):
                        cv2.putText(display, text, (10, 30 + i*25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, display)
                    needs_render = False
                except Exception as e:
                    print(f"Render error: {e}")
            
            key = cv2.waitKey(10) & 0xFF
            
            if key == 255:
                continue
            
            print(f"Key: {key}")
            needs_render = True
            
            if key == 27:  # ESC
                print("\nExiting without saving.")
                break
            
            elif key == ord('r') or key == ord('R'):
                self.x_offset = self.left.shape[1]
                self.y_offset = 0
                self.rotation = 0.0
                self.scale = 1.0
                self.opacity = 0.5
                print("\nReset")
            
            elif key == ord('k'):
                shift_pressed = False
            elif key == ord('l'):
                shift_pressed = True
            
            step = self.step_size * 5 if shift_pressed else self.step_size
            rot_step = self.rotation_step * 5 if shift_pressed else self.rotation_step
            
            if key == ord('w') or key == ord('W'):
                self.y_offset -= step
            elif key == ord('a') or key == ord('A'):
                self.x_offset -= step
            elif key == ord('d') or key == ord('D'):
                self.x_offset += step
            elif key == ord('s'):
                self.y_offset += step
            elif key == ord('S'):
                saved_calib = {
                    'x_offset': self.x_offset,
                    'y_offset': self.y_offset,
                    'rotation': self.rotation,
                    'scale': self.scale,
                    'opacity': self.opacity,
                    'inner_stretch': self.inner_stretch,
                    'canvas_width': width,
                    'canvas_height': height,
                    'offset': offset
                }
                print("\nCALIBRATION SAVED!")
            elif key == ord('q') or key == ord('Q'):
                self.rotation -= rot_step
            elif key == ord('e') or key == ord('E'):
                self.rotation += rot_step
            elif key == ord('z') or key == ord('Z'):
                self.scale = max(0.1, self.scale - self.scale_step)
            elif key == ord('x') or key == ord('X'):
                self.scale = min(2.0, self.scale + self.scale_step)
            elif key == ord('-') or key == ord('_'):
                self.opacity = max(0.0, self.opacity - self.opacity_step)
            elif key == ord('=') or key == ord('+'):
                self.opacity = min(1.0, self.opacity + self.opacity_step)
        
        cv2.destroyAllWindows()
        return saved_calib

def save_calibration(path, calib_data):
    """Save calibration to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    print(f"\nCalibration saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Manually stitch left and right video frames'
    )
    parser.add_argument('--left', default='data/undistorted/left.mp4',
                        help='Left video path')
    parser.add_argument('--right', default='data/undistorted/right.mp4',
                        help='Right video path')
    parser.add_argument('--calib', default='data/calibration/manual_stitch_calibration.json',
                        help='Calibration output path')
    parser.add_argument('--stretch', type=float, default=1.0,
                        help='Inner edge stretch factor (< 1.0 = compress, > 1.0 = expand)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MANUAL VIDEO STITCHING")
    print("=" * 60)
    print(f"Left video:  {args.left}")
    print(f"Right video: {args.right}")
    print(f"Calibration: {args.calib}")
    print("=" * 60)
    
    # Open videos and read first frames
    print("\nOpening videos...")
    cap_left = cv2.VideoCapture(args.left)
    cap_right = cv2.VideoCapture(args.right)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open videos")
        return
    
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        print("Error: Could not read frames")
        return
    
    print(f"Left frame:  {frame_left.shape[1]}x{frame_left.shape[0]}")
    print(f"Right frame: {frame_right.shape[1]}x{frame_right.shape[0]}")
    
    # Run manual stitcher with stretch factor
    stitcher = ManualStitcher(frame_left, frame_right, inner_stretch=args.stretch)
    calib_data = stitcher.run()
    
    # Save calibration if user saved
    if calib_data is not None:
        save_calibration(args.calib, calib_data)
        print("\n" + "=" * 60)
        print("COMPLETE!")
        print("=" * 60)
    
    cap_left.release()
    cap_right.release()


if __name__ == '__main__':
    main()
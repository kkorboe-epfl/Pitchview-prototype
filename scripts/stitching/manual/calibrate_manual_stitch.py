import argparse
import json
import os
import numpy as np
import cv2


class ManualStitcher:
    def __init__(self, left_image, right_image, inner_stretch=1.0, initial_calib=None):
        self.left = left_image.copy()
        self.right = right_image.copy()
        
        if initial_calib:
            self.x_offset = initial_calib.get('x_offset', left_image.shape[1])
            self.y_offset = initial_calib.get('y_offset', 0)
            self.rotation = initial_calib.get('rotation', 0.0)
            self.scale = initial_calib.get('scale', 1.0)
            self.opacity = initial_calib.get('opacity', 0.5)
            self.inner_stretch = initial_calib.get('inner_stretch', inner_stretch)
            print("\nLoaded existing calibration")
        else:
            self.x_offset = left_image.shape[1]
            self.y_offset = 0
            self.rotation = 0.0
            self.scale = 1.0
            self.opacity = 0.5
            self.inner_stretch = inner_stretch
        
        self.step_size = 10
        self.rotation_step = 0.5
        self.scale_step = 0.01
        self.opacity_step = 0.05
        
        print("\n" + "=" * 60)
        print("STITCH CALIBRATION - KEYBOARD CONTROLS")
        print("=" * 60)
        print("Position:    Arrow Keys  Move right image")
        print("Rotation:    Q/E         Rotate counterclockwise/clockwise")
        print("Scale:       Z/X         Scale down/up")
        print("Opacity:     -/=         Show more left/right")
        print("Adjust:      [/]         Fine/Coarse adjustment mode")
        print("Navigation:  SPACE       Next frame")
        print("             BACKSPACE   Previous frame")
        print("Save/Exit:   S           Save calibration and exit")
        print("             R           Reset to initial position")
        print("             ESC         Exit without saving")
        print("=" * 60)
        print(f"Position: x={self.x_offset}, y={self.y_offset}")
        print(f"Rotation: {self.rotation}° | Scale: {self.scale} | Opacity: {self.opacity}")
        print(f"Inner stretch: {self.inner_stretch}")
        print("=" * 60)
    
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
    
    def run(self, video_left=None, video_right=None):
        """Run interactive stitching loop."""
        main_window = "Stitch Calibration"
        help_window = "Controls (you can close this)"
        
        cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(help_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(main_window, 1400, 800)
        cv2.resizeWindow(help_window, 450, 600)
        
        # Position windows side by side
        cv2.moveWindow(main_window, 0, 0)
        cv2.moveWindow(help_window, 1000, 0)
        
        # Create help panel
        help_panel = np.zeros((600, 450, 3), dtype=np.uint8)
        help_text = [
            "KEYBOARD CONTROLS",
            "",
            "Position:",
            "  Up/W        Move up",
            "  Down/S      Move down",
            "  Left/A      Move left",
            "  Right/D     Move right",
            "",
            "Rotation:",
            "  Q           Rotate right (CW)",
            "  E           Rotate left (CCW)",
            "",
            "Scale:",
            "  Z           Scale down",
            "  X           Scale up",
            "",
            "Opacity:",
            "  -           Show more left",
            "  =           Show more right",
            "",
            "Navigate Frames:",
            "  SPACE       Next frame",
            "  BACKSPACE   Previous frame",
            "",
            "Actions:",
            "  S           Save & exit",
            "  R           Reset position",
            "  ESC         Exit (no save)",
        ]
        
        y_pos = 25
        for i, line in enumerate(help_text):
            if line and not line.startswith(" "):
                # Header
                cv2.putText(help_panel, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
            else:
                # Regular text
                cv2.putText(help_panel, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            y_pos += 18
        
        cv2.imshow(help_window, help_panel)
        
        saved_calib = None
        coarse_mode = False
        needs_render = True
        frame_idx = 0
        total_frames = 0
        
        if video_left and video_right:
            total_frames = int(video_left.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("\nCalibration window opened. Click on main window to activate keyboard controls.")
        print("You can close the controls window if you want.\n")
        
        while True:
            if needs_render:
                canvas, (width, height, offset) = self.render()
                display, display_scale = self.scale_for_display(canvas)
                
                # Minimal overlay - just status
                info_text = [
                    f"Pos:({self.x_offset},{self.y_offset}) Rot:{self.rotation:.1f}deg Scale:{self.scale:.3f} Opacity:{self.opacity:.2f}",
                    f"Mode: {'COARSE' if coarse_mode else 'FINE'}",
                ]
                
                if total_frames > 0:
                    info_text.append(f"Frame: {frame_idx + 1}/{total_frames}")
                
                for i, text in enumerate(info_text):
                    cv2.putText(display, text, (10, 25 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow(main_window, display)
                needs_render = False
            
            key = cv2.waitKey(10)
            
            if key == -1:
                continue
            
            needs_render = True
            
            if key == 27:  # ESC
                print("\nExited without saving.")
                break
            
            elif key == ord('r') or key == ord('R'):
                self.x_offset = self.left.shape[1]
                self.y_offset = 0
                self.rotation = 0.0
                self.scale = 1.0
                self.opacity = 0.5
                print("Reset to initial position")
            
            elif key == ord('['):
                coarse_mode = False
                print("Fine adjustment mode")
            elif key == ord(']'):
                coarse_mode = True
                print("Coarse adjustment mode")
            
            step = self.step_size * 5 if coarse_mode else self.step_size
            rot_step = self.rotation_step * 5 if coarse_mode else self.rotation_step
            
            # Arrow keys use extended codes (platform-specific)
            # Up=2490368, Down=2621440, Left=2424832, Right=2555904 on some systems
            # Up=82, Down=84, Left=81, Right=83 on others
            # We check both the full key code and the masked version
            key_masked = key & 0xFF
            
            if key == 2490368 or key_masked == 82 or key == ord('w') or key == ord('W'):  # Up
                self.y_offset -= step
            elif key == 2621440 or key_masked == 84:  # Down arrow only (S is for save)
                self.y_offset += step
            elif key == ord('s') or key == ord('S'):  # S key - save and exit
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
                print("\n" + "=" * 60)
                print("CALIBRATION SAVED!")
                print("=" * 60)
                break
            elif key == 2424832 or key_masked == 81 or key == ord('a') or key == ord('A'):  # Left
                self.x_offset -= step
            elif key == 2555904 or key_masked == 83 or key == ord('d') or key == ord('D'):  # Right
                self.x_offset += step
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
            
            elif key == ord(' ') and video_left and video_right:  # Space
                frame_idx = min(frame_idx + 1, total_frames - 1)
                video_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                video_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_left, self.left = video_left.read()
                ret_right, self.right = video_right.read()
                if ret_left and ret_right:
                    print(f"Frame: {frame_idx + 1}/{total_frames}")
            
            elif key == 8 and video_left and video_right:  # Backspace
                frame_idx = max(frame_idx - 1, 0)
                video_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                video_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_left, self.left = video_left.read()
                ret_right, self.right = video_right.read()
                if ret_left and ret_right:
                    print(f"Frame: {frame_idx + 1}/{total_frames}")
        
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
        description='Interactively calibrate stitching alignment for dual-camera videos'
    )
    parser.add_argument('--left', default='data/undistorted/left.mp4',
                        help='Left camera video path (default: data/undistorted/left.mp4)')
    parser.add_argument('--right', default='data/undistorted/right.mp4',
                        help='Right camera video path (default: data/undistorted/right.mp4)')
    parser.add_argument('--output', default='data/calibration/manual_stitch_calibration.json',
                        help='Calibration output path (default: data/calibration/manual_stitch_calibration.json)')
    parser.add_argument('--stretch', type=float, default=1.0,
                        help='Inner edge stretch factor (default: 1.0)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load existing calibration to refine')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.left):
        print(f"Error: Left video not found: {args.left}")
        return
    if not os.path.exists(args.right):
        print(f"Error: Right video not found: {args.right}")
        return
    
    print("\n" + "=" * 60)
    print("STITCH CALIBRATION")
    print("=" * 60)
    print(f"Left:   {args.left}")
    print(f"Right:  {args.right}")
    print(f"Output: {args.output}")
    print(f"Stretch: {args.stretch}")
    if args.load:
        print(f"Loading: {args.load}")
    print("=" * 60)
    
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
    
    print(f"Left:  {frame_left.shape[1]}x{frame_left.shape[0]}")
    print(f"Right: {frame_right.shape[1]}x{frame_right.shape[0]}")
    
    # Load existing calibration if specified
    initial_calib = None
    if args.load and os.path.exists(args.load):
        with open(args.load, 'r') as f:
            initial_calib = json.load(f)
        print(f"Loaded calibration from: {args.load}")
    
    # Reset to first frame
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_left.read()
    cap_right.read()
    
    stitcher = ManualStitcher(frame_left, frame_right, 
                             inner_stretch=args.stretch,
                             initial_calib=initial_calib)
    calib_data = stitcher.run(cap_left, cap_right)
    
    if calib_data is not None:
        save_calibration(args.output, calib_data)
        print(f"Calibration parameters:")
        print(f"  Position: ({calib_data['x_offset']}, {calib_data['y_offset']})")
        print(f"  Rotation: {calib_data['rotation']:.2f}°")
        print(f"  Scale: {calib_data['scale']:.3f}")
        print("\nUse this calibration with apply_manual_stitch.py")
    
    cap_left.release()
    cap_right.release()


if __name__ == '__main__':
    main()
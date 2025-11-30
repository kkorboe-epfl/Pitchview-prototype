#!/usr/bin/env python3
"""
Interactive tool to manually align left and right camera frames.
Use arrow keys to move the right frame until it aligns with the left.
Then save the transform.

Usage:
  python scripts/live_demo_scripts/interactive_align.py \
    --left output/live_demo/undistorted/left.mp4 \
    --right output/live_demo/undistorted/right.mp4 \
    --save-calib data/calibration/live_calibration.json
"""

import argparse
import cv2
import numpy as np
import json
from pathlib import Path


class InteractiveAligner:
    def __init__(self, left_path, right_path, save_path):
        self.left_path = left_path
        self.right_path = right_path
        self.save_path = save_path
        
        # Load first frames
        cap_left = cv2.VideoCapture(left_path)
        cap_right = cv2.VideoCapture(right_path)
        
        ret_l, self.frame_left = cap_left.read()
        ret_r, self.frame_right = cap_right.read()
        
        cap_left.release()
        cap_right.release()
        
        if not ret_l or not ret_r:
            raise RuntimeError("Failed to read frames")
        
        # Get dimensions
        self.h_left, self.w_left = self.frame_left.shape[:2]
        self.h_right, self.w_right = self.frame_right.shape[:2]
        
        # Initial offset (start with right frame shifted to the right)
        self.offset_x = self.w_left  # Place right frame next to left
        self.offset_y = 0
        
        # Blend alpha
        self.blend_alpha = 0.5
        
        # Movement step size
        self.step_size = 10
        self.fine_step = 1
        
        print("\n" + "="*60)
        print("INTERACTIVE FRAME ALIGNMENT")
        print("="*60)
        print("Controls:")
        print("  Arrow Keys - Move right frame (hold Shift for fine control)")
        print("  [ / ] - Decrease/increase transparency")
        print("  r - Reset to default position")
        print("  s - Save calibration and exit")
        print("  q/ESC - Quit without saving")
        print("="*60 + "\n")
    
    def create_panorama(self):
        """Create panorama with current offset."""
        # Calculate canvas size
        canvas_w = max(self.w_left, self.offset_x + self.w_right)
        canvas_h = max(self.h_left, self.offset_y + self.h_right)
        
        # Create canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Place left frame at origin
        canvas[0:self.h_left, 0:self.w_left] = self.frame_left
        
        # Calculate right frame position
        x1 = self.offset_x
        y1 = self.offset_y
        x2 = self.offset_x + self.w_right
        y2 = self.offset_y + self.h_right
        
        # Clamp to canvas
        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(canvas_w, x2)
        y2_c = min(canvas_h, y2)
        
        # Calculate crop in right frame
        rx1 = x1_c - x1
        ry1 = y1_c - y1
        rx2 = rx1 + (x2_c - x1_c)
        ry2 = ry1 + (y2_c - y1_c)
        
        # Blend in overlap region
        if x2_c > x1_c and y2_c > y1_c:
            right_crop = self.frame_right[ry1:ry2, rx1:rx2]
            canvas_region = canvas[y1_c:y2_c, x1_c:x2_c]
            
            # Blend where both frames exist
            blended = cv2.addWeighted(canvas_region, 1 - self.blend_alpha, 
                                     right_crop, self.blend_alpha, 0)
            canvas[y1_c:y2_c, x1_c:x2_c] = blended
        
        return canvas
    
    def save_calibration(self):
        """Save the transform as a simple translation homography."""
        # Create homography matrix (simple translation)
        H = np.array([
            [1.0, 0.0, float(self.offset_x)],
            [0.0, 1.0, float(self.offset_y)],
            [0.0, 0.0, 1.0]
        ])
        
        # Calculate canvas size
        canvas_w = max(self.w_left, self.offset_x + self.w_right)
        canvas_h = max(self.h_left, self.offset_y + self.h_right)
        
        # Create calibration data
        calib = {
            "H": H.tolist(),
            "offset": [0, 0],  # Left frame is at origin
            "pano_size": [canvas_w, canvas_h],
            "used_affine": False,
            "manual_alignment": True,
            "description": f"Manual alignment: right frame offset by ({self.offset_x}, {self.offset_y})"
        }
        
        # Save to file
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(calib, f, indent=2)
        
        print(f"\n✓ Calibration saved to: {self.save_path}")
        print(f"  Offset: ({self.offset_x}, {self.offset_y})")
        print(f"  Canvas size: {canvas_w}x{canvas_h}")
    
    def run(self):
        """Run the interactive alignment tool."""
        cv2.namedWindow("Interactive Alignment", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Interactive Alignment", 1600, 600)
        
        while True:
            # Create current panorama
            pano = self.create_panorama()
            
            # Add info overlay
            info_text = [
                f"Offset: ({self.offset_x}, {self.offset_y})",
                f"Blend: {int(self.blend_alpha * 100)}%",
                "Press 's' to save, 'q' to quit"
            ]
            
            y_pos = 30
            for text in info_text:
                cv2.putText(pano, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_pos += 30
            
            # Scale for display
            scale = min(1.0, 1600 / pano.shape[1])
            if scale < 1.0:
                display = cv2.resize(pano, 
                                    (int(pano.shape[1] * scale), 
                                     int(pano.shape[0] * scale)))
            else:
                display = pano
            
            cv2.imshow("Interactive Alignment", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Check for shift key (fine control)
            modifiers = cv2.waitKey(1)
            is_shift = (modifiers & 0xFF) == 225  # Shift key
            step = self.fine_step if is_shift else self.step_size
            
            # Arrow keys
            if key == 81 or key == 2:  # Left arrow
                self.offset_x -= step
                print(f"Offset: ({self.offset_x}, {self.offset_y})")
            elif key == 83 or key == 3:  # Right arrow
                self.offset_x += step
                print(f"Offset: ({self.offset_x}, {self.offset_y})")
            elif key == 82 or key == 0:  # Up arrow
                self.offset_y -= step
                print(f"Offset: ({self.offset_x}, {self.offset_y})")
            elif key == 84 or key == 1:  # Down arrow
                self.offset_y += step
                print(f"Offset: ({self.offset_x}, {self.offset_y})")
            
            # Transparency control
            elif key == ord('['):
                self.blend_alpha = max(0.0, self.blend_alpha - 0.1)
                print(f"Blend: {int(self.blend_alpha * 100)}%")
            elif key == ord(']'):
                self.blend_alpha = min(1.0, self.blend_alpha + 0.1)
                print(f"Blend: {int(self.blend_alpha * 100)}%")
            
            # Reset
            elif key == ord('r'):
                self.offset_x = self.w_left
                self.offset_y = 0
                self.blend_alpha = 0.5
                print("Reset to default")
            
            # Save
            elif key == ord('s'):
                self.save_calibration()
                break
            
            # Quit
            elif key == ord('q') or key == 27:
                print("Quit without saving")
                break
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive tool to manually align camera frames'
    )
    parser.add_argument('--left', required=True, help='Left camera video')
    parser.add_argument('--right', required=True, help='Right camera video')
    parser.add_argument('--save-calib', required=True, 
                       help='Output calibration JSON file')
    
    args = parser.parse_args()
    
    aligner = InteractiveAligner(args.left, args.right, args.save_calib)
    aligner.run()


if __name__ == '__main__':
    main()

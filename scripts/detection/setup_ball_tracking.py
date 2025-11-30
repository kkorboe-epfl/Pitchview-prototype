#!/usr/bin/env python3
"""
Tool to set up ball tracking: draw playing field boundary and mark initial ball position.
Everything outside the polygon will be excluded from ball detection.
"""

import cv2
import numpy as np
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python setup_ball_tracking.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    sys.exit(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read first frame")
    sys.exit(1)

h, w = frame.shape[:2]

print("\n" + "="*60)
print("BALL TRACKING SETUP")
print("="*60)
print("STEP 1: Draw the playing field boundary")
print("  - Click to add points around the field")
print("  - Press 'c' to close the polygon")
print("  - Press 'r' to reset and redraw")
print()
print("STEP 2: Mark the initial ball position")
print("  - Click on the ball")
print()
print("Press 'q' to save and quit")
print("="*60 + "\n")

# State
mode = "draw_field"  # "draw_field" or "mark_ball"
field_polygon = []
ball_position = None
display = frame.copy()

def redraw():
    global display
    display = frame.copy()
    
    # Draw field polygon
    if len(field_polygon) > 0:
        for i, pt in enumerate(field_polygon):
            cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, tuple(field_polygon[i-1]), tuple(pt), (0, 255, 0), 2)
        
        if len(field_polygon) >= 3:
            # Show polygon outline
            cv2.polylines(display, [np.array(field_polygon)], True, (0, 255, 0), 2)
            
            # Show exclusion zone in red with transparency if polygon is closed
            if mode == "mark_ball":
                overlay = display.copy()
                
                # Create mask for inside polygon
                mask_inside = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(field_polygon, dtype=np.int32)
                cv2.fillPoly(mask_inside, [pts], 255)
                
                # Exclusion is everything outside
                mask_exclusion = cv2.bitwise_not(mask_inside)
                
                # Draw exclusion zone in red
                exclusion_overlay = np.zeros_like(frame)
                exclusion_overlay[mask_exclusion > 0] = (0, 0, 255)
                cv2.addWeighted(exclusion_overlay, 0.3, overlay, 1.0, 0, overlay)
                
                # Draw field boundary in green
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
                
                display = overlay
    
    # Draw ball position
    if ball_position is not None:
        cv2.circle(display, ball_position, 15, (0, 255, 255), 2)
        cv2.circle(display, ball_position, 3, (0, 255, 255), -1)
        cv2.putText(display, "BALL", (ball_position[0] + 20, ball_position[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw instructions
    if mode == "draw_field":
        cv2.putText(display, f"STEP 1: Draw field boundary ({len(field_polygon)} points) - Press 'c' to close", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(display, "STEP 2: Click on the ball - Press 'q' to save", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(display, "r=reset, q=save&quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global field_polygon, ball_position, display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "draw_field":
            field_polygon.append([x, y])
            redraw()
            print(f"  Point {len(field_polygon)}: ({x}, {y})")
        
        elif mode == "mark_ball":
            ball_position = (x, y)
            redraw()
            print(f"  Ball position: ({x}, {y})")

cv2.namedWindow("Ball Tracking Setup", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ball Tracking Setup", 1600, 420)
cv2.setMouseCallback("Ball Tracking Setup", mouse_callback)

redraw()

while True:
    cv2.imshow("Ball Tracking Setup", display)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c') and mode == "draw_field":  # Close polygon
        if len(field_polygon) >= 3:
            mode = "mark_ball"
            redraw()
            print(f"\nField boundary closed with {len(field_polygon)} points")
            print("Now click on the ball...")
    
    elif key == ord('r'):  # Reset
        if mode == "draw_field":
            field_polygon = []
            redraw()
            print("Field boundary reset")
        elif mode == "mark_ball":
            ball_position = None
            redraw()
            print("Ball position reset")
    
    elif key == ord('q'):  # Save and quit
        if len(field_polygon) >= 3 and ball_position is not None:
            break
        else:
            print("Error: Need both field boundary (3+ points) and ball position")
    
    elif key == 27:  # ESC
        cv2.destroyAllWindows()
        sys.exit(0)

cv2.destroyAllWindows()

# Create exclusion mask (everything outside the field polygon)
exclusion_mask = np.ones((h, w), dtype=np.uint8) * 255
pts = np.array(field_polygon, dtype=np.int32)
cv2.fillPoly(exclusion_mask, [pts], 0)  # Set inside to 0 (not excluded)

# Save configuration
config = {
    "field_polygon": field_polygon,
    "ball_position": ball_position,
    "frame_dimensions": {"width": w, "height": h}
}

with open("data/calibration/ball_tracking_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n" + "="*60)
print("CONFIGURATION SAVED")
print("="*60)
print(f"Field boundary: {len(field_polygon)} points")
print(f"Ball position: {ball_position}")
print(f"Saved to: data/calibration/ball_tracking_config.json")
print("="*60)

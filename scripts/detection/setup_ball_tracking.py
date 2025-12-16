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
print("  - Press 'c' to move to ball marking")
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

def create_instructions_panel():
    """Create an instructions panel to display alongside the video."""
    panel_width = 400
    panel_height = h
    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 40  # Dark gray background
    
    instructions = [
        ("BALL TRACKING SETUP", 30, (255, 255, 255), 0.8, 2),
        ("", 60, (255, 255, 255), 0.6, 1),
        ("STEP 1: Draw Field Boundary", 90, (0, 255, 0), 0.7, 2),
        ("  Click points around field", 120, (200, 200, 200), 0.5, 1),
        ("  Press 'c' to move to ball marking", 145, (200, 200, 200), 0.5, 1),
        ("  Press 'r' to reset", 170, (200, 200, 200), 0.5, 1),
        ("", 200, (255, 255, 255), 0.6, 1),
        ("STEP 2: Mark Ball Position", 230, (0, 255, 255), 0.7, 2),
        ("  Click on the ball", 260, (200, 200, 200), 0.5, 1),
        ("  Press 'q' to save and quit", 285, (200, 200, 200), 0.5, 1),
        ("  ESC - Exit without saving", 310, (200, 200, 200), 0.5, 1),
    ]
    
    for text, y, color, scale, thickness in instructions:
        cv2.putText(panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   scale, color, thickness, cv2.LINE_AA)
    
    # Add status section at bottom
    status_y = panel_height - 150
    cv2.line(panel, (10, status_y - 10), (panel_width - 10, status_y - 10), (100, 100, 100), 2)
    cv2.putText(panel, "Status:", (20, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 200, 100), 2, cv2.LINE_AA)
    
    # Dynamic status based on mode and progress
    if mode == "draw_field":
        status_text = f"Drawing field ({len(field_polygon)} pts)"
        status_color = (0, 255, 255)
    elif mode == "mark_ball":
        status_text = "Ready to mark ball"
        status_color = (0, 255, 0)
        if ball_position is not None:
            status_text = "Ball marked - Press 'q'"
            status_color = (0, 255, 0)
    
    cv2.putText(panel, status_text, (20, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, status_color, 2, cv2.LINE_AA)
    
    # Color legend
    legend_y = status_y + 100
    cv2.circle(panel, (30, legend_y), 8, (0, 255, 0), -1)
    cv2.putText(panel, "Field boundary", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.circle(panel, (30, legend_y + 25), 8, (0, 0, 255), -1)
    cv2.putText(panel, "Exclusion zone", (50, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.circle(panel, (30, legend_y + 50), 8, (0, 255, 255), -1)
    cv2.putText(panel, "Ball position", (50, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    return panel

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
    
    # Draw simple instructions overlay on video (at bottom to avoid overlap)
    text_y = h - 20
    if mode == "draw_field":
        cv2.putText(display, f"Draw field boundary ({len(field_polygon)} points) - Press 'c' to move to ball marking", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(display, "Click on the ball - Press 'q' to save", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
cv2.setMouseCallback("Ball Tracking Setup", mouse_callback)

# Scale window to fit screen better
combined_width = w + 400  # Video + instructions panel
scale = min(1920 / combined_width, 1.0)  # Don't scale up, only down if needed
window_width = int(combined_width * scale)
window_height = int(h * scale)
cv2.resizeWindow("Ball Tracking Setup", window_width, window_height)

redraw()

while True:
    # Create instructions panel
    instructions_panel = create_instructions_panel()
    
    # Combine video frame with instructions panel side-by-side
    combined = np.hstack([display, instructions_panel])
    
    cv2.imshow("Ball Tracking Setup", combined)
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

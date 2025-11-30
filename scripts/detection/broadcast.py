#!/usr/bin/env python3
"""
Advanced Broadcast System with:
- ByteTrack multi-object tracking
- Homography-based pitch coordinates
- Kalman filter for ball prediction
- Spring-damper camera motion
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import torch

# ---------------- ARGUMENTS ---------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="output/stitched/panorama.mp4",
                    help="Path to input panoramic video")
parser.add_argument("--save-preview", type=str, default=None,
                    help="Save the panorama preview video")
parser.add_argument("--save-broadcast", type=str, default="output/broadcast/broadcast_view.mp4",
                    help="Save the broadcast view video")
args = parser.parse_args()

# ---------------- SETTINGS ---------------- #

VIDEO_PATH = args.video
print(f"Using video: {VIDEO_PATH}")

# Detection settings
PLAYER_MODEL_PATH = "models/yolov8s.pt"
YOLO_CONF_THRESH = 0.3
BALL_CONF_THRESH = 0.2
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32

# Tracking settings (ByteTrack parameters)
TRACK_THRESH = 0.5
TRACK_BUFFER = 30
MATCH_THRESH = 0.8

# Camera settings
BROADCAST_WIDTH = 1280
BROADCAST_HEIGHT = 720
BROADCAST_ASPECT = 16.0 / 9.0

# Spring-damper camera motion (in pixel space)
SPRING_CONSTANT = 0.04  # How strongly camera pulls toward target
DAMPER_CONSTANT = 0.8   # How much camera resists movement (damping)
ZOOM_SMOOTHING = 0.008  # Ultra smooth zoom transitions

# Kalman filter settings for ball
KALMAN_PROCESS_NOISE_POS = 1.0
KALMAN_PROCESS_NOISE_VEL = 5.0  # Higher to follow acceleration better
KALMAN_MEASUREMENT_NOISE = 10.0
MAX_FRAMES_WITHOUT_DETECTION = 60  # Continue tracking with prediction when ball is lost

# View framing (in pixel space)
PLAYER_INFLUENCE_RADIUS_PX = 400  # pixels - distance to include nearby players
VIEW_MARGIN = 1.3  # Margin around targets
MIN_VIEW_WIDTH = 600   # pixels (most zoomed in)
MAX_VIEW_WIDTH = 2000  # pixels (most zoomed out)
DEFAULT_VIEW_WIDTH = 1000

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------- KALMAN FILTER FOR BALL ---------------- #

def create_ball_kalman_filter():
    """
    Kalman filter for ball in pixel coordinates.
    State: [x, y, vx, vy] in pixels and pixels/frame
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    
    # State transition
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Measurement function
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    
    # Measurement noise
    kf.R = np.eye(2) * 15.0  # Trust detections even more
    
    # Process noise - higher for velocity to follow acceleration
    kf.Q = np.diag([2.5, 2.5, 18.0, 18.0])  # Higher velocity noise to track fast movements
    
    # Initial covariance
    kf.P = np.eye(4) * 100.0
    
    return kf


# ---------------- SPRING-DAMPER CAMERA MOTION ---------------- #

class SpringDamperCamera:
    """
    Physics-based camera motion using spring-damper system.
    Works in pixel space for accurate framing.
    """
    def __init__(self, spring_k=0.04, damper_k=0.8):
        self.spring_k = spring_k
        self.damper_k = damper_k
        self.center_px = None  # Current camera center in pixels
        self.velocity = np.array([0.0, 0.0])  # Camera velocity in pixels
        self.view_width = DEFAULT_VIEW_WIDTH  # Current view width in pixels
    
    def update(self, target_center_px, target_width):
        """
        Update camera position using spring-damper physics.
        target_center_px: [x, y] in pixel coordinates
        target_width: desired view width in pixels
        """
        if self.center_px is None:
            self.center_px = np.array(target_center_px, dtype=float)
            self.view_width = target_width
            return self.center_px, self.view_width
        
        target = np.array(target_center_px, dtype=float)
        
        # Spring-damper for position
        displacement = target - self.center_px
        spring_force = self.spring_k * displacement
        damping_force = -self.damper_k * self.velocity
        acceleration = spring_force + damping_force
        
        # Update velocity and position
        self.velocity += acceleration
        self.center_px += self.velocity
        
        # Smooth view width with LERP
        self.view_width = self.view_width * (1 - ZOOM_SMOOTHING) + target_width * ZOOM_SMOOTHING
        
        return self.center_px.copy(), self.view_width
    
    def reset(self):
        """Reset camera state."""
        self.center_px = None
        self.velocity = np.array([0.0, 0.0])
        self.view_width = DEFAULT_VIEW_WIDTH


# ---------------- DETECTION & TRACKING ---------------- #

def detect_objects_with_tracking(model, frame):
    """
    Detect and track players and ball using YOLO with ByteTrack.
    Returns: (players_dict, ball_detection)
    Note: Ball detection from YOLO is often unreliable, use HSV as primary.
    """
    # Run YOLO with tracking
    results = model.track(frame, persist=True, conf=YOLO_CONF_THRESH, 
                         tracker="bytetrack.yaml", verbose=False)
    
    if len(results) == 0 or results[0].boxes is None:
        return {}, None
    
    boxes = results[0].boxes
    players = {}
    ball = None
    
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Get track ID if available
        track_id = int(box.id[0]) if box.id is not None else None
        
        if cls_id == PERSON_CLASS_ID and track_id is not None:
            players[track_id] = {
                'bbox': (x1, y1, x2, y2),
                'center_px': (cx, cy),
                'conf': conf
            }
        
        # Ball detection from YOLO - but it's unreliable, will use HSV instead
        elif cls_id == SPORTS_BALL_CLASS_ID and conf > BALL_CONF_THRESH:
            # Sanity check - ball should be reasonably small
            ball_w = x2 - x1
            ball_h = y2 - y1
            if ball_w < 100 and ball_h < 100:  # Ball shouldn't be huge
                if ball is None or conf > ball['conf']:
                    ball = {
                        'center_px': (cx, cy),
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf
                    }
    
    return players, ball


# ---------------- HSV BALL DETECTION (FALLBACK) ---------------- #

def detect_ball_hsv(frame, last_pos=None, exclusion_mask=None):
    """HSV-based red ball detection - proven working version."""
    h, w = frame.shape[:2]
    
    # If we have a last position, only search in a local region
    search_radius = 600  # pixels - larger radius to catch fast-moving ball
    search_mask = None
    
    if last_pos is not None:
        search_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(search_mask, last_pos, search_radius, 255, -1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Optimized HSV ranges for red ball
    lower_red1 = np.array([0, 60, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([165, 60, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply search region mask
    if search_mask is not None:
        mask = cv2.bitwise_and(mask, search_mask)
    
    # Apply exclusion zones
    if exclusion_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclusion_mask))
    
    # Morphological operations
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5 or area > 10000:  # More lenient for small/large balls
            continue
        
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 1.5 or r > 80:  # More lenient radius range
            continue
        
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        
        # Circularity check
        circ = 4 * np.pi * area / (peri * peri)
        if circ < 0.25:  # More lenient circularity
            continue
        
        # Extent check
        circle_area = np.pi * r * r
        extent = area / circle_area if circle_area > 0 else 0
        if extent < 0.35:  # More lenient
            continue
        
        # Aspect ratio check
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        aspect_ratio = float(w_cnt) / h_cnt if h_cnt > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # More lenient
            continue
        
        candidates.append((int(x), int(y), int(r), circ, area, extent))
    
    # Pick best candidate
    if not candidates:
        return None
    
    best_ball = None
    best_score = -1
    
    for (x, y, r, circ, area, extent) in candidates:
        # Prioritize roundness
        score = circ * 4.0 + extent * 3.0
        
        # VERY strongly prefer smaller balls (actual soccer ball vs cones/markers)
        if area < 50:  # Tiny ball (far away)
            score += 5.0
        elif area < 150:  # Small ball
            score += 4.0
        elif area < 400:  # Medium ball
            score += 1.0
        else:  # Large objects (definitely cones/markers)
            score -= 5.0
        
        # Temporal coherence with VERY strong bonus for proximity
        if last_pos:
            lx, ly = last_pos
            dist = np.hypot(x - lx, y - ly)
            # VERY strong bonus for being close to last position
            if dist < 30:
                score += 10.0  # Almost certainly the same ball
            elif dist < 80:
                score += 6.0
            elif dist < 200:
                score += 2.0
            else:
                score -= dist * 0.01  # Heavy penalty for jumps
        
        # Radius preferences - STRONGLY prefer smaller balls
        if r < 6:  # Very small (distant ball)
            score += 3.0
        elif r < 12:  # Small ball
            score += 2.0
        elif r < 20:  # Medium ball
            score += 0.5
        elif r > 35:  # Large object (cone)
            score -= 5.0
        
        if score > best_score:
            best_score = score
            best_ball = {'center_px': (x, y), 'conf': circ, 'radius': r, 'area': area}
    
    return best_ball


# ---------------- FRAMING LOGIC ---------------- #

def compute_target_view(ball_px, players_px, ball_velocity_px, frame_shape):
    """
    Compute target camera position and view width in pixel space.
    
    ball_px: (x, y) ball position in pixels
    players_px: dict of player pixel positions {track_id: (x, y)}
    ball_velocity_px: (vx, vy) ball velocity in pixels/frame
    
    Returns: (target_center_px, target_width)
    """
    h, w = frame_shape[:2]
    
    if ball_px is None:
        return np.array([w/2, h/2]), DEFAULT_VIEW_WIDTH
    
    ball_pos_px = np.array(ball_px)
    
    # Find players near ball using pixel distances
    nearby_players_px = []
    for track_id, player_px in players_px.items():
        dist_px = np.linalg.norm(np.array(player_px) - ball_pos_px)
        if dist_px < PLAYER_INFLUENCE_RADIUS_PX:
            nearby_players_px.append(player_px)
    
    # Compute center of mass in pixel space (ball + nearby players)
    all_points_px = [ball_pos_px]
    all_points_px.extend(nearby_players_px)
    
    target_center = np.mean(all_points_px, axis=0)
    
    # Compute view width based on spread in pixels
    if len(all_points_px) > 1:
        points_array = np.array(all_points_px)
        spread_x = np.ptp(points_array[:, 0])
        spread_y = np.ptp(points_array[:, 1])
        spread = max(spread_x, spread_y * BROADCAST_ASPECT)  # Account for aspect ratio
        target_width = max(spread * VIEW_MARGIN, MIN_VIEW_WIDTH)
    else:
        target_width = DEFAULT_VIEW_WIDTH
    
    # Adjust width based on ball velocity in pixels
    if ball_velocity_px is not None:
        speed_px = np.linalg.norm(ball_velocity_px)
        if speed_px > 30:  # Fast movement
            target_width *= 1.5
        elif speed_px > 15:
            target_width *= 1.2
    
    target_width = np.clip(target_width, MIN_VIEW_WIDTH, MAX_VIEW_WIDTH)
    
    # Clamp center to valid range
    half_width = target_width / 2
    half_height = target_width / (2 * BROADCAST_ASPECT)
    target_center[0] = np.clip(target_center[0], half_width, w - half_width)
    target_center[1] = np.clip(target_center[1], half_height, h - half_height)
    
    return target_center, target_width


def pixel_camera_to_crop(camera_center_px, view_width, frame_shape):
    """
    Convert pixel-based camera position to crop coordinates.
    camera_center_px: (x, y) center in pixels
    view_width: view width in pixels
    frame_shape: (height, width)
    
    Returns: (x1, y1, x2, y2) pixel crop coordinates
    """
    h, w = frame_shape[:2]
    
    half_width = view_width / 2
    half_height = view_width / (2 * BROADCAST_ASPECT)
    
    x1 = int(camera_center_px[0] - half_width)
    x2 = int(camera_center_px[0] + half_width)
    y1 = int(camera_center_px[1] - half_height)
    y2 = int(camera_center_px[1] + half_height)
    
    # Clamp to frame boundaries
    x1 = max(0, min(x1, w-1))
    x2 = max(x1+1, min(x2, w))
    y1 = max(0, min(y1, h-1))
    y2 = max(y1+1, min(y2, h))
    
    return x1, y1, x2, y2


# ---------------- MAIN PROCESSING ---------------- #

def main():
    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {w}x{h} @ {fps} fps, {total_frames} frames")
    
    # No homography needed - using pure pixel-space calculations
    
    # Load YOLO model
    model = YOLO(PLAYER_MODEL_PATH)
    model.to(device)
    
    # Load ball tracking configuration (field boundary + initial ball position)
    exclusion_mask = None
    last_ball_pos = None
    try:
        import json
        with open("data/calibration/ball_tracking_config.json", "r") as f:
            config = json.load(f)
            
            # Load field polygon and create exclusion mask (everything outside)
            if "field_polygon" in config:
                field_polygon = config["field_polygon"]
                exclusion_mask = np.ones((h, w), dtype=np.uint8) * 255
                pts = np.array(field_polygon, dtype=np.int32)
                cv2.fillPoly(exclusion_mask, [pts], 0)  # Inside field = 0 (not excluded)
                print(f"Loaded field boundary with {len(field_polygon)} points")
            
            # Load initial ball position
            if "ball_position" in config:
                last_ball_pos = tuple(config["ball_position"])
                print(f"Loaded initial ball position: {last_ball_pos}")
    except Exception as e:
        print(f"No ball tracking config found: {e}")
        print("Run setup_ball_tracking.py first to configure field boundary and ball position")
    
    # Initialize tracking
    ball_kf = create_ball_kalman_filter()
    camera = SpringDamperCamera(SPRING_CONSTANT, DAMPER_CONSTANT)
    ball_initialized = False
    frames_without_detection = 0  # Track how long ball has been lost
    
    # Output videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save_broadcast, fourcc, fps, (BROADCAST_WIDTH, BROADCAST_HEIGHT))
    
    if args.save_preview:
        preview_out = cv2.VideoWriter(args.save_preview, fourcc, fps, (w, h))
    
    frame_count = 0
    
    print("\nProcessing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track objects
        players, ball_detection_yolo = detect_objects_with_tracking(model, frame)
        
        # Always use HSV as primary ball detection (more reliable for red ball)
        ball_detection = detect_ball_hsv(frame, last_ball_pos, exclusion_mask)
        
        # Update last ball position for next frame
        if ball_detection is not None:
            last_ball_pos = ball_detection['center_px']
            # If we were in a long loss period and found the ball, reinitialize
            if frames_without_detection >= MAX_FRAMES_WITHOUT_DETECTION:
                ball_initialized = False  # Force reinitialization
            frames_without_detection = 0
        else:
            frames_without_detection += 1
        
        # Process ball with Kalman filter in pixel coordinates
        ball_px = None
        ball_velocity_px = None
        
        if ball_detection is not None:
            ball_px = ball_detection['center_px']
            
            if not ball_initialized:
                ball_kf.x = np.array([ball_px[0], ball_px[1], 0, 0])
                ball_initialized = True
            
            # Kalman update
            ball_kf.predict()
            ball_kf.update(np.array(ball_px))
            
            ball_px = ball_kf.x[:2]
            ball_velocity_px = ball_kf.x[2:4]
        elif ball_initialized and frames_without_detection < MAX_FRAMES_WITHOUT_DETECTION:
            # Predict only - keep tracking even without detection for up to N frames
            ball_kf.predict()
            ball_px = ball_kf.x[:2]
            ball_velocity_px = ball_kf.x[2:4]
            
            # Clamp predictions to reasonable bounds (with margin)
            ball_px[0] = np.clip(ball_px[0], -200, w + 200)
            ball_px[1] = np.clip(ball_px[1], -200, h + 200)
            
            # If prediction is way off screen, stop predicting and reset
            if ball_px[0] < 0 or ball_px[0] > w or ball_px[1] < 0 or ball_px[1] > h:
                frames_without_detection = MAX_FRAMES_WITHOUT_DETECTION  # Stop predicting
            else:
                # Update last_ball_pos to predicted position for search area
                last_ball_pos = (int(ball_px[0]), int(ball_px[1]))
        
        # Get player positions in pixel space
        players_px = {}
        for track_id, player in players.items():
            players_px[track_id] = player['center_px']
        
        # Compute target view in pixel space
        target_center_px, target_width = compute_target_view(
            ball_px, players_px, ball_velocity_px, frame.shape
        )
        
        # Update camera with spring-damper in pixel space
        camera_center_px, camera_width = camera.update(target_center_px, target_width)
        
        # Convert to pixel crop
        x1, y1, x2, y2 = pixel_camera_to_crop(camera_center_px, camera_width, frame.shape)
        
        # Extract and resize broadcast view
        crop = frame[y1:y2, x1:x2]
        broadcast_frame = cv2.resize(crop, (BROADCAST_WIDTH, BROADCAST_HEIGHT))
        out.write(broadcast_frame)
        
        # Preview frame
        if args.save_preview:
            preview = frame.copy()
            
            # Draw raw ball detection (cyan)
            if ball_detection:
                bx, by = ball_detection['center_px']
                cv2.circle(preview, (int(bx), int(by)), 10, (255, 255, 0), 2)
                cv2.circle(preview, (int(bx), int(by)), 2, (255, 255, 0), -1)
                # Draw detection confidence/type
                det_type = "YOLO" if 'conf' in ball_detection and ball_detection['conf'] > 0.5 else "HSV"
                cv2.putText(preview, det_type, (int(bx) + 15, int(by)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw Kalman filtered ball (yellow outline)
            if ball_px is not None:
                cv2.circle(preview, (int(ball_px[0]), int(ball_px[1])), 12, (0, 255, 255), 2)
                # Draw velocity vector
                if ball_velocity_px is not None:
                    vx, vy = ball_velocity_px
                    end_x = int(ball_px[0] + vx * 3)
                    end_y = int(ball_px[1] + vy * 3)
                    cv2.arrowedLine(preview, (int(ball_px[0]), int(ball_px[1])), 
                                   (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)
            
            # Draw players
            for track_id, player in players.items():
                px1, py1, px2, py2 = player['bbox']
                cv2.rectangle(preview, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(preview, f"ID:{track_id}", (px1, py1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Draw camera view
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            preview_out.write(preview)
        
        frame_count += 1
        if frame_count % 30 == 0:
            speed_px = np.linalg.norm(ball_velocity_px) if ball_velocity_px is not None else 0
            ball_status = "YOLO" if (ball_detection and 'conf' in ball_detection and ball_detection['conf'] > 0.5) else "HSV" if ball_detection else f"PRED({frames_without_detection})" if frames_without_detection > 0 else "LOST"
            ball_pos_str = f"({int(ball_px[0])},{int(ball_px[1])})" if ball_px is not None else "N/A"
            ball_details = f"r={int(ball_detection['radius'])},a={int(ball_detection['area'])}" if ball_detection and 'radius' in ball_detection else ""
            print(f"Frame {frame_count}/{total_frames} | Ball: {ball_status} at {ball_pos_str} {ball_details} | Speed: {speed_px:.1f}px/f")
    
    cap.release()
    out.release()
    if args.save_preview:
        preview_out.release()
    
    print(f"\nBroadcast video saved to: {args.save_broadcast}")
    if args.save_preview:
        print(f"Preview video saved to: {args.save_preview}")


if __name__ == '__main__':
    main()

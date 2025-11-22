# Sample usage:
# python scripts/detection/broadcast_yolo.py --video output/stitched/20251116_103024_stitched.mp4 --save-broadcast output/broadcast/broadcast_view.mp4 --save-preview output/broadcast/panorama_preview.mp4

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / '.env')

# ---------------- ARGUMENTS ---------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Path to input panoramic video (overrides .env)")
parser.add_argument("--save-preview", type=str, default=None,
                    help="Save the panorama preview video to this file")
parser.add_argument("--save-broadcast", type=str, default=None,
                    help="Save the broadcast view video to this file")
args = parser.parse_args()


# ---------------- SETTINGS ---------------- #

if args.video:
    VIDEO_PATH = args.video
elif os.getenv('PANO_VIDEO'):
    VIDEO_PATH = str(Path(__file__).parent.parent / os.getenv('PANO_VIDEO'))
else:
    VIDEO_PATH = str(Path(__file__).parent.parent / "footage/pano08.mp4")

print(f"Using video: {VIDEO_PATH}")
PREVIEW_WIDTH = 1600
EXCLUSION_RATIO = 0.13    # top 13%
BOTTOM_EXCLUSION_RATIO = 0.15   # bottom 15%
LEFT_EXCLUSION_RATIO = 0.05     # left band (0 = off)
RIGHT_EXCLUSION_RATIO = 0.00    # right band (0 = off)

MAX_MISSED_FRAMES = 20  # Increased to handle fast ball kicks better

PLAYER_MIN_AREA = 100      
PLAYER_MAX_AREA = 50000    
PLAYER_MIN_ASPECT = 0.5    
PLAYER_MAX_ASPECT = 5.0
PLAYER_NEAR_DIST = 600

BROADCAST_ASPECT = 16.0 / 9.0
MIN_VIEW_WIDTH = 1400  # Increased for wider view
MAX_VIEW_WIDTH = 3200  # Increased max width
MARGIN_FACTOR = 0.75  # Increased margins to show more context
MARGIN_PIXELS = 120  # More pixel margin

ENABLE_DYNAMIC_ZOOM = True
ZOOM_OUT_VELOCITY = 12  # Higher threshold for zoom activation
ZOOM_MAX_VELOCITY = 45  # Higher max for stronger kicks
ZOOM_OUT_FACTOR = 4.5  # Increased zoom out factor for more context
ZOOM_SMOOTHING = 0.008  # Much smoother zoom transitions

SMOOTHING_NORMAL = 0.006  # Smoother camera movement
SMOOTHING_FAST = 0.05  # Balanced response - not too fast to cause jitter
SMOOTHING_SIZE = 0.003  # Smoother size changes
BALL_POSITION_SMOOTHING = 0.18  # Smooth ball position but less lag
BALL_SAFE_MARGIN = 0.48  # Larger safety margin to keep ball in frame during speed changes

ENABLE_PREDICTION = True  # Enable predictive tracking to anticipate ball movement
USE_KALMAN_FILTER = True  # Use Kalman filter for smoother prediction
USE_PLAYER_CONTEXT = True  # Use nearby player position to bias prediction
PREDICTION_FRAMES = 6  # Look ahead more frames for fast ball
PREDICTION_WEIGHT = 0.6  # Balanced prediction weight
PLAYER_DIRECTION_WEIGHT = 0.25  # Reduced to prevent over-correction
VELOCITY_SMOOTHING = 0.6  # Balanced velocity smoothing
SPEED_SMOOTHING = 0.45  # More smoothing to reduce jitter from speed changes
MIN_VELOCITY = 8.0  # Only predict for faster movements

LEADING_SPACE_FACTOR = 0.0
MIN_LEADING_SPACE = 0

VELOCITY_THRESHOLD_SLOW = 8   # Threshold for slow movement
VELOCITY_THRESHOLD_FAST = 25  # Balanced threshold
SPEED_CHANGE_THRESHOLD = 20  # Higher threshold - only react to very sudden changes

PLAYER_MODEL_PATH = "yolov8s.pt"
PERSON_CLASS_ID = 0
YOLO_CONF_THRESH = 0.2
DETECTION_INTERVAL = 5

import torch
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
player_model = YOLO(PLAYER_MODEL_PATH)
player_model.to(device)


# ---------------- BALL DETECTION ---------------- #
def detect_red_candidates(
    frame,
    top_exclusion_ratio=EXCLUSION_RATIO,
    bottom_exclusion_ratio=BOTTOM_EXCLUSION_RATIO,
    left_exclusion_ratio=LEFT_EXCLUSION_RATIO,
    right_exclusion_ratio=RIGHT_EXCLUSION_RATIO,
    draw_exclusions=False,
    debug_frame=None,
):
    h, w = frame.shape[:2]

    top_excl = int(h * top_exclusion_ratio)
    bottom_excl = int(h * bottom_exclusion_ratio)
    left_excl = int(w * left_exclusion_ratio)
    right_excl = int(w * right_exclusion_ratio)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 70], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 70, 70], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # ---- apply exclusion zones on the mask ----
    if top_excl > 0:
        mask[0:top_excl, :] = 0

    if bottom_excl > 0:
        mask[h - bottom_excl:h, :] = 0

    if left_excl > 0:
        mask[:, 0:left_excl] = 0

    if right_excl > 0:
        mask[:, w - right_excl:w] = 0

    # ---- optional visualisation of exclusion zones ----
    if draw_exclusions and debug_frame is not None:
        # top band
        if top_excl > 0:
            cv2.rectangle(
                debug_frame,
                (0, 0),
                (w - 1, top_excl - 1),
                (255, 0, 255),
                2,
            )
        # bottom band
        if bottom_excl > 0:
            cv2.rectangle(
                debug_frame,
                (0, h - bottom_excl),
                (w - 1, h - 1),
                (255, 0, 255),
                2,
            )
        # left band
        if left_excl > 0:
            cv2.rectangle(
                debug_frame,
                (0, 0),
                (left_excl - 1, h - 1),
                (255, 0, 255),
                2,
            )
        # right band
        if right_excl > 0:
            cv2.rectangle(
                debug_frame,
                (w - right_excl, 0),
                (w - 1, h - 1),
                (255, 0, 255),
                2,
            )

    # ---- rest of your pipeline ----
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15 or area > 12000:  # Increased max area for fast moving/blurred ball
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 3 or r > 90:  # Increased max radius for motion blur
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        # Calculate circularity (how round the contour is)
        circ = 4 * np.pi * area / (peri * peri)
        if circ < 0.35:  # More lenient circularity for motion blur
            continue
        
        # Additional roundness check: compare contour area to bounding circle area
        circle_area = np.pi * r * r
        extent = area / circle_area if circle_area > 0 else 0
        if extent < 0.5:  # Must occupy at least 50% of bounding circle
            continue
        
        # Check aspect ratio of bounding rectangle
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        aspect_ratio = float(w_cnt) / h_cnt if h_cnt > 0 else 0
        if aspect_ratio < 0.6 or aspect_ratio > 1.7:  # Should be roughly square
            continue

        candidates.append((int(x), int(y), int(r), circ, area, extent))

    return candidates

def pick_best_candidate(cands, last_pos=None):
    if not cands:
        return None

    best, best_score = None, -1
    for (x, y, r, circ, area, extent) in cands:
        # Heavily prioritize circularity and extent (roundness)
        score = circ * 3.0 + extent * 2.0
        
        # Prefer consistent tracking
        if last_pos:
            lx, ly = last_pos
            score -= np.hypot(x - lx, y - ly) * 0.01
        
        # Slight bonus for reasonable size
        score += np.log(max(area, 1)) * 0.1

        if score > best_score:
            best_score = score
            best = (x, y, r)

    return best


# ---------------- PLAYER DETECTION (YOLO) ---------------- #

def detect_players_yolo(frame, exclusion_ratio=EXCLUSION_RATIO):
    h, w = frame.shape[:2]
    exclusion_h = int(h * exclusion_ratio)

    results = player_model(frame, conf=YOLO_CONF_THRESH, verbose=False)[0]
    players = []

    for box in results.boxes:
        if int(box.cls[0]) != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if y2 < exclusion_h:
            continue

        wb, hb = x2 - x1, y2 - y1
        if wb <= 0 or hb <= 0:
            continue

        area = wb * hb
        if not (PLAYER_MIN_AREA <= area <= PLAYER_MAX_AREA):
            continue

        asp = hb / float(wb + 1e-6)
        if not (PLAYER_MIN_ASPECT <= asp <= PLAYER_MAX_ASPECT):
            continue

        cx = x1 + wb // 2
        cy = y1 + hb // 2
        players.append({"bbox": (x1, y1, x2, y2), "centre": (cx, cy)})

    return players


# ---------------- NEARBY PLAYER & VIEW LOGIC ---------------- #

def create_ball_kalman_filter():
    """
    Create a Kalman filter for ball tracking.
    State: [x, y, vx, vy] - position and velocity in 2D
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # State transition matrix (assumes constant velocity model)
    dt = 1.0  # time step (1 frame)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Measurement function (we only measure position, not velocity)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    
    # Measurement noise (uncertainty in ball detection)
    kf.R = np.array([[10.0, 0],
                     [0, 10.0]])
    
    # Process noise (uncertainty in motion model)
    kf.Q = np.eye(4) * 0.5
    
    # Initial covariance
    kf.P = np.eye(4) * 100.0
    
    return kf


def choose_nearby(players, ball, max_count=2):
    if ball is None:
        return []
    bx, by, _ = ball
    close = []
    for p in players:
        cx, cy = p["centre"]
        d = np.hypot(cx - bx, cy - by)
        if d <= PLAYER_NEAR_DIST:
            close.append((d, p))
    close.sort(key=lambda z: z[0])
    return [p for (_, p) in close[:max_count]]


def compute_view(ball, nearby, shape, ball_velocity=None, zoom_factor=1.0):
    h, w = shape[:2]
    if ball is None:
        return np.array([0, 0, w, h], float)

    bx, by, r = ball
    
    # Apply predictive tracking
    predicted_bx, predicted_by = bx, by
    
    if ENABLE_PREDICTION and ball_velocity is not None:
        if USE_KALMAN_FILTER:
            # Kalman filter prediction is already in the state
            # We'll get this from the main loop
            vx, vy = ball_velocity
            speed = np.hypot(vx, vy)
            
            if speed > MIN_VELOCITY:
                # Use Kalman filter velocity for prediction
                pred_offset_x = vx * PREDICTION_FRAMES
                pred_offset_y = vy * PREDICTION_FRAMES
                
                # Use player context to bias prediction if enabled
                if USE_PLAYER_CONTEXT and len(nearby) > 0:
                    # Find closest player to ball
                    closest_player = None
                    min_dist = float('inf')
                    
                    for p in nearby:
                        pcx, pcy = p["centre"]
                        dist = np.hypot(pcx - bx, pcy - by)
                        if dist < min_dist:
                            min_dist = dist
                            closest_player = p
                    
                    if closest_player is not None and min_dist < 150:  # Only if very close
                        # Calculate direction from player to ball
                        pcx, pcy = closest_player["centre"]
                        player_to_ball_x = bx - pcx
                        player_to_ball_y = by - pcy
                        
                        # Normalize
                        magnitude = np.hypot(player_to_ball_x, player_to_ball_y)
                        if magnitude > 0:
                            player_to_ball_x /= magnitude
                            player_to_ball_y /= magnitude
                            
                            # Bias prediction in direction away from player
                            # This assumes ball is likely to move away from nearby player
                            bias_x = player_to_ball_x * speed * PLAYER_DIRECTION_WEIGHT
                            bias_y = player_to_ball_y * speed * PLAYER_DIRECTION_WEIGHT
                            
                            pred_offset_x += bias_x * PREDICTION_FRAMES
                            pred_offset_y += bias_y * PREDICTION_FRAMES
                
                predicted_bx = bx + pred_offset_x * PREDICTION_WEIGHT
                predicted_by = by + pred_offset_y * PREDICTION_WEIGHT
                
                predicted_bx = max(0, min(w, predicted_bx))
                predicted_by = max(0, min(h, predicted_by))
        else:
            # Original linear prediction
            vx, vy = ball_velocity
            speed = np.hypot(vx, vy)
            
            if speed > MIN_VELOCITY:
                pred_offset_x = vx * PREDICTION_FRAMES
                pred_offset_y = vy * PREDICTION_FRAMES
                
                predicted_bx = bx + pred_offset_x * PREDICTION_WEIGHT
                predicted_by = by + pred_offset_y * PREDICTION_WEIGHT
                
                predicted_bx = max(0, min(w, predicted_bx))
                predicted_by = max(0, min(h, predicted_by))
    
    # Use predicted position for framing the view
    xs = [predicted_bx]
    ys = [predicted_by]
    
    # But keep actual ball position in the calculation too for stability
    xs.append(bx)
    ys.append(by)

    for p in nearby:
        x1, y1, x2, y2 = p["bbox"]
        xs += [x1, x2]
        ys += [y1, y2]

    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))

    ww = xmax - xmin
    hh = ymax - ymin

    # Apply dynamic zoom - increase margins based on zoom factor
    mx = (ww * MARGIN_FACTOR + MARGIN_PIXELS) * zoom_factor
    my = (hh * MARGIN_FACTOR + MARGIN_PIXELS) * zoom_factor

    xmin -= mx
    xmax += mx
    ymin -= my
    ymax += my

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    # minimum and maximum width with zoom factor applied
    min_width = MIN_VIEW_WIDTH * zoom_factor
    max_width = MAX_VIEW_WIDTH
    
    current_width = xmax - xmin
    
    if current_width < min_width:
        c = (xmin + xmax) / 2
        half = min_width / 2
        xmin = max(0, c - half)
        xmax = min(w, c + half)
    elif current_width > max_width:
        c = (xmin + xmax) / 2
        half = max_width / 2
        xmin = max(0, c - half)
        xmax = min(w, c + half)

    # enforce aspect ratio
    ww = xmax - xmin
    hh = ymax - ymin
    aspect = ww / (hh + 1e-6)

    if aspect > BROADCAST_ASPECT:
        desired_h = ww / BROADCAST_ASPECT
        extra = desired_h - hh
        ymin -= extra / 2
        ymax += extra / 2
    else:
        desired_w = hh * BROADCAST_ASPECT
        extra = desired_w - ww
        xmin -= extra / 2
        xmax += extra / 2

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    return np.array([xmin, ymin, xmax, ymax], float)


def lerp_rect(a, b, t):
    return a * (1 - t) + b * t


# ---------------- MAIN LOOP ---------------- #

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video")
        return

    preview_writer = None
    broadcast_writer = None

    last_pos = None
    last_radius = None
    missed = 0
    view_rect = None
    ball_velocity = np.array([0.0, 0.0])
    prev_ball_pos = None
    smoothed_ball_pos = None  # Add smoothed ball position
    current_zoom = 1.0
    smoothed_speed = 0.0  # Add smoothed speed to prevent jitter
    prev_speed = 0.0  # Track previous speed for acceleration detection
    frame_count = 0
    cached_players = []
    
    # Initialize Kalman filter for ball tracking
    ball_kf = None
    if USE_KALMAN_FILTER:
        ball_kf = create_ball_kalman_filter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # prepare display frame early so we can draw debug overlays on it
        disp = frame.copy()

        # --- Ball detection ---
        cand = detect_red_candidates(
            frame,
            draw_exclusions=True,      # set to False to hide boxes
            debug_frame=disp,
        )
        ball = pick_best_candidate(cand, last_pos)

        if ball:
            bx, by, br = ball
            
            if USE_KALMAN_FILTER and ball_kf is not None:
                # Update Kalman filter with measurement
                measurement = np.array([float(bx), float(by)])
                
                if last_pos is None:
                    # Initialize filter state with first measurement
                    ball_kf.x = np.array([bx, by, 0.0, 0.0])
                
                # Predict step
                ball_kf.predict()
                
                # Update step
                ball_kf.update(measurement)
                
                # Get filtered position and velocity
                bx_smooth = int(ball_kf.x[0])
                by_smooth = int(ball_kf.x[1])
                vx_kalman = ball_kf.x[2]
                vy_kalman = ball_kf.x[3]
                
                # Use Kalman velocity estimate
                ball_velocity[0] = vx_kalman
                ball_velocity[1] = vy_kalman
                
                # Update ball to use Kalman filtered position
                ball = (bx_smooth, by_smooth, br)
                
            else:
                # Original smoothing approach
                # Smooth ball position to reduce jitter
                if smoothed_ball_pos is None:
                    smoothed_ball_pos = np.array([float(bx), float(by)])
                else:
                    smoothed_ball_pos[0] = smoothed_ball_pos[0] * (1 - BALL_POSITION_SMOOTHING) + bx * BALL_POSITION_SMOOTHING
                    smoothed_ball_pos[1] = smoothed_ball_pos[1] * (1 - BALL_POSITION_SMOOTHING) + by * BALL_POSITION_SMOOTHING
                
                # Use smoothed position for tracking
                bx_smooth, by_smooth = int(smoothed_ball_pos[0]), int(smoothed_ball_pos[1])
                
                # Update velocity using smoothed position
                if prev_ball_pos is not None:
                    new_vx = bx_smooth - prev_ball_pos[0]
                    new_vy = by_smooth - prev_ball_pos[1]
                    # Smooth velocity estimate
                    ball_velocity[0] = ball_velocity[0] * (1 - VELOCITY_SMOOTHING) + new_vx * VELOCITY_SMOOTHING
                    ball_velocity[1] = ball_velocity[1] * (1 - VELOCITY_SMOOTHING) + new_vy * VELOCITY_SMOOTHING
                
                prev_ball_pos = (bx_smooth, by_smooth)
                # Update ball to use smoothed position for view calculation
                ball = (bx_smooth, by_smooth, br)
            
            last_pos = (bx, by)
            last_radius = br
            missed = 0
            
        else:
            if last_pos and missed < MAX_MISSED_FRAMES:
                missed += 1
                bx, by = last_pos
                br = last_radius or 10
                
                # If using Kalman, still predict even without measurement
                if USE_KALMAN_FILTER and ball_kf is not None:
                    ball_kf.predict()
                    bx_pred = int(ball_kf.x[0])
                    by_pred = int(ball_kf.x[1])
                    ball = (bx_pred, by_pred, br)
                else:
                    ball = (bx, by, br)
            else:
                ball = None
                prev_ball_pos = None
                smoothed_ball_pos = None
                if USE_KALMAN_FILTER and ball_kf is not None:
                    ball_kf = create_ball_kalman_filter()  # Reset filter

        # --- Player detection (only every N frames) ---
        if frame_count % DETECTION_INTERVAL == 0:
            cached_players = detect_players_yolo(frame)
        
        players = cached_players
        frame_count += 1

        # --- Nearby players ---
        near = choose_nearby(players, ball)
        
        # --- Calculate dynamic zoom based on ball velocity ---
        if ENABLE_DYNAMIC_ZOOM and ball_velocity is not None:
            speed = np.hypot(ball_velocity[0], ball_velocity[1])
            
            # Smooth the speed itself to prevent jitter
            smoothed_speed = smoothed_speed * (1 - SPEED_SMOOTHING) + speed * SPEED_SMOOTHING
            
            if smoothed_speed < ZOOM_OUT_VELOCITY:
                target_zoom = 1.0  # Normal zoom
            elif smoothed_speed > ZOOM_MAX_VELOCITY:
                target_zoom = ZOOM_OUT_FACTOR  # Maximum zoom out
            else:
                # Interpolate zoom based on smoothed speed
                t = (smoothed_speed - ZOOM_OUT_VELOCITY) / (ZOOM_MAX_VELOCITY - ZOOM_OUT_VELOCITY)
                target_zoom = 1.0 + (ZOOM_OUT_FACTOR - 1.0) * t
            
            # Smooth zoom changes very aggressively
            current_zoom = current_zoom * (1 - ZOOM_SMOOTHING) + target_zoom * ZOOM_SMOOTHING
            
            # Debug: print every 30 frames
            if frame_count % 30 == 0:
                print(f"Speed: {speed:.1f} (smoothed: {smoothed_speed:.1f}) px/frame | Zoom: {current_zoom:.2f}x")
        else:
            current_zoom = 1.0

        # --- Compute target view with velocity and zoom ---
        target = compute_view(ball, near, frame.shape, ball_velocity, current_zoom)

        if view_rect is None:
            view_rect = target.copy()
        else:
            # velocity-based adaptive smoothing with smooth interpolation
            speed = np.hypot(ball_velocity[0], ball_velocity[1])
            
            # Detect sudden speed changes (acceleration)
            speed_change = abs(speed - prev_speed)
            is_accelerating = speed_change > SPEED_CHANGE_THRESHOLD
            
            # Smooth interpolation between slow and fast smoothing (no sudden jumps)
            if speed < VELOCITY_THRESHOLD_SLOW:
                base_alpha = SMOOTHING_NORMAL
            elif speed > VELOCITY_THRESHOLD_FAST:
                base_alpha = SMOOTHING_FAST
            else:
                # Smooth interpolation to prevent jitter at threshold boundaries
                t = (speed - VELOCITY_THRESHOLD_SLOW) / (VELOCITY_THRESHOLD_FAST - VELOCITY_THRESHOLD_SLOW)
                # Use smooth curve instead of linear
                t_smooth = t * t * (3.0 - 2.0 * t)  # Smoothstep function
                base_alpha = SMOOTHING_NORMAL + (SMOOTHING_FAST - SMOOTHING_NORMAL) * t_smooth
            
            # Boost responsiveness when accelerating
            if is_accelerating:
                base_alpha = min(base_alpha * 1.5, SMOOTHING_FAST * 1.2)  # Less aggressive boost
            
            alpha = base_alpha
            if ball:
                bx, by, _ = ball
                x1, y1, x2, y2 = view_rect
                vw, vh = (x2 - x1), (y2 - y1)

                mx, my = vw * BALL_SAFE_MARGIN, vh * BALL_SAFE_MARGIN
                sx1, sx2 = x1 + mx, x2 - mx
                sy1, sy2 = y1 + my, y2 - my

                if not (sx1 <= bx <= sx2 and sy1 <= by <= sy2):
                    alpha = SMOOTHING_FAST * 1.2  # Less aggressive when near edge

            # Smooth position and size separately to prevent zoom jitter
            curr_cx = (view_rect[0] + view_rect[2]) / 2
            curr_cy = (view_rect[1] + view_rect[3]) / 2
            curr_w = view_rect[2] - view_rect[0]
            curr_h = view_rect[3] - view_rect[1]
            
            target_cx = (target[0] + target[2]) / 2
            target_cy = (target[1] + target[3]) / 2
            target_w = target[2] - target[0]
            target_h = target[3] - target[1]
            
            # Smooth center position
            new_cx = curr_cx * (1 - alpha) + target_cx * alpha
            new_cy = curr_cy * (1 - alpha) + target_cy * alpha
            
            # Smooth size changes separately with different rate
            new_w = curr_w * (1 - SMOOTHING_SIZE) + target_w * SMOOTHING_SIZE
            new_h = curr_h * (1 - SMOOTHING_SIZE) + target_h * SMOOTHING_SIZE
            
            view_rect = np.array([
                new_cx - new_w/2,
                new_cy - new_h/2,
                new_cx + new_w/2,
                new_cy + new_h/2
            ])
            
            prev_speed = speed

        # clamp + int conversion
        x1, y1, x2, y2 = map(int, map(round, view_rect))
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # --- Crops ---
        broadcast_crop = frame[y1:y2, x1:x2]
        broadcast_view = cv2.resize(broadcast_crop, (1280, 720))

        # --- Preview frame ---

        if ball:
            cv2.circle(disp, (bx, by), br, (0, 255, 255), 2)
            cv2.circle(disp, (bx, by), 3, (0, 255, 0), -1)

        for p in players:
            xx1, yy1, xx2, yy2 = p["bbox"]
            col = (255, 255, 0)
            if p in near:
                col = (0, 0, 255)
            cv2.rectangle(disp, (xx1, yy1), (xx2, yy2), col, 2)

        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)

        scale = PREVIEW_WIDTH / float(w)
        preview = cv2.resize(disp, (PREVIEW_WIDTH, int(h * scale)))

        # --- Initialise writers if needed ---
        if args.save_preview and preview_writer is None:
            from pathlib import Path
            Path(args.save_preview).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            ph, pw = preview.shape[:2]
            preview_writer = cv2.VideoWriter(args.save_preview, fourcc, 30, (pw, ph))

        if args.save_broadcast and broadcast_writer is None:
            from pathlib import Path
            Path(args.save_broadcast).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            bh, bw = broadcast_view.shape[:2]
            broadcast_writer = cv2.VideoWriter(args.save_broadcast, fourcc, 30, (bw, bh))

        # --- Write frames if enabled ---
        if preview_writer:
            preview_writer.write(preview)

        if broadcast_writer:
            broadcast_writer.write(broadcast_view)

        # --- Show windows ---
        cv2.imshow("Panorama view", preview)
        cv2.imshow("Broadcast view", broadcast_view)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    if preview_writer:
        preview_writer.release()
    if broadcast_writer:
        broadcast_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

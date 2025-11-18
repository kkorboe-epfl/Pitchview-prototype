# Sample usage:
# python scripts/detection/broadcast_yolo.py --video output/stitched/20251116_103024_stitched.mp4 --save-broadcast output/broadcast/broadcast_view.mp4 --save-preview output/broadcast/panorama_preview.mp4

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

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
EXCLUSION_RATIO = 0.13
MAX_MISSED_FRAMES = 10

PLAYER_MIN_AREA = 100      
PLAYER_MAX_AREA = 50000    
PLAYER_MIN_ASPECT = 0.5    
PLAYER_MAX_ASPECT = 5.0
PLAYER_NEAR_DIST = 600

BROADCAST_ASPECT = 16.0 / 9.0
MIN_VIEW_WIDTH = 800
MAX_VIEW_WIDTH = 2400
MARGIN_FACTOR = 0.45
MARGIN_PIXELS = 60

ENABLE_DYNAMIC_ZOOM = True
ZOOM_OUT_VELOCITY = 8
ZOOM_MAX_VELOCITY = 25
ZOOM_OUT_FACTOR = 2.5
ZOOM_SMOOTHING = 0.06

SMOOTHING_NORMAL = 0.015
SMOOTHING_FAST = 0.06
SMOOTHING_SIZE = 0.015
BALL_SAFE_MARGIN = 0.25

ENABLE_PREDICTION = False
PREDICTION_FRAMES = 2
VELOCITY_SMOOTHING = 0.3
MIN_VELOCITY = 5.0

LEADING_SPACE_FACTOR = 0.0
MIN_LEADING_SPACE = 0

VELOCITY_THRESHOLD_SLOW = 10
VELOCITY_THRESHOLD_FAST = 40

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

def detect_red_candidates(frame, exclusion_ratio=EXCLUSION_RATIO):
    h, w = frame.shape[:2]
    exclusion_height = int(h * exclusion_ratio)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 70], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 70, 70], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask[0:exclusion_height, :] = 0

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 8000:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 4 or r > 70:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circ = 4 * np.pi * area / (peri * peri)
        if circ < 0.45:
            continue

        candidates.append((int(x), int(y), int(r), circ, area))

    return candidates


def pick_best_candidate(cands, last_pos=None):
    if not cands:
        return None

    best, best_score = None, -1
    for (x, y, r, circ, area) in cands:
        score = circ * 2.0
        if last_pos:
            lx, ly = last_pos
            score -= np.hypot(x - lx, y - ly) * 0.01
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
    original_bx, original_by = bx, by  # Keep original position for player framing
    
    # Apply predictive tracking if enabled and velocity is available
    # This is used for leading space, not for centering the view
    pred_offset_x, pred_offset_y = 0, 0
    if ENABLE_PREDICTION and ball_velocity is not None:
        vx, vy = ball_velocity
        speed = np.hypot(vx, vy)
        
        # Only predict if ball is moving fast enough
        if speed > MIN_VELOCITY:
            # Calculate prediction offset
            pred_offset_x = vx * PREDICTION_FRAMES
            pred_offset_y = vy * PREDICTION_FRAMES
    
    # Start with ball and nearby players at their ACTUAL positions
    xs = [bx]
    ys = [by]

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
    
    # Add leading space in direction of motion (using prediction offset)
    if ball_velocity is not None:
        vx, vy = ball_velocity
        speed = np.hypot(vx, vy)
        
        if speed > MIN_VELOCITY:
            # Calculate leading space based on velocity direction
            leading_x = max(MIN_LEADING_SPACE, ww * LEADING_SPACE_FACTOR) + abs(pred_offset_x) * 0.3
            leading_y = max(MIN_LEADING_SPACE, hh * LEADING_SPACE_FACTOR) + abs(pred_offset_y) * 0.3
            
            # Shift the view in direction of motion
            if vx > 0:  # moving right
                xmax += leading_x
            elif vx < 0:  # moving left
                xmin -= leading_x
                
            if vy > 0:  # moving down
                ymax += leading_y
            elif vy < 0:  # moving up
                ymin -= leading_y

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
    current_zoom = 1.0
    frame_count = 0
    cached_players = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # --- Ball detection ---
        cand = detect_red_candidates(frame)
        ball = pick_best_candidate(cand, last_pos)

        if ball:
            bx, by, br = ball
            last_pos = (bx, by)
            last_radius = br
            missed = 0
            
            # Update velocity
            if prev_ball_pos is not None:
                new_vx = bx - prev_ball_pos[0]
                new_vy = by - prev_ball_pos[1]
                # Smooth velocity estimate
                ball_velocity[0] = ball_velocity[0] * (1 - VELOCITY_SMOOTHING) + new_vx * VELOCITY_SMOOTHING
                ball_velocity[1] = ball_velocity[1] * (1 - VELOCITY_SMOOTHING) + new_vy * VELOCITY_SMOOTHING
            
            prev_ball_pos = (bx, by)
        else:
            if last_pos and missed < MAX_MISSED_FRAMES:
                missed += 1
                bx, by = last_pos
                br = last_radius or 10
                ball = (bx, by, br)
            else:
                ball = None
                prev_ball_pos = None

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
            
            if speed < ZOOM_OUT_VELOCITY:
                target_zoom = 1.0  # Normal zoom
            elif speed > ZOOM_MAX_VELOCITY:
                target_zoom = ZOOM_OUT_FACTOR  # Maximum zoom out
            else:
                # Interpolate zoom based on speed
                t = (speed - ZOOM_OUT_VELOCITY) / (ZOOM_MAX_VELOCITY - ZOOM_OUT_VELOCITY)
                target_zoom = 1.0 + (ZOOM_OUT_FACTOR - 1.0) * t
            
            # Smooth zoom changes
            current_zoom = current_zoom * (1 - ZOOM_SMOOTHING) + target_zoom * ZOOM_SMOOTHING
            
            # Debug: print every 30 frames
            if frame_count % 30 == 0:
                print(f"Speed: {speed:.1f} px/frame | Zoom: {current_zoom:.2f}x")
        else:
            current_zoom = 1.0

        # --- Compute target view with velocity and zoom ---
        target = compute_view(ball, near, frame.shape, ball_velocity, current_zoom)

        if view_rect is None:
            view_rect = target.copy()
        else:
            # velocity-based adaptive smoothing
            speed = np.hypot(ball_velocity[0], ball_velocity[1])
            
            if speed < VELOCITY_THRESHOLD_SLOW:
                base_alpha = SMOOTHING_NORMAL
            elif speed > VELOCITY_THRESHOLD_FAST:
                base_alpha = SMOOTHING_FAST
            else:
                # Interpolate between slow and fast based on speed
                t = (speed - VELOCITY_THRESHOLD_SLOW) / (VELOCITY_THRESHOLD_FAST - VELOCITY_THRESHOLD_SLOW)
                base_alpha = SMOOTHING_NORMAL + (SMOOTHING_FAST - SMOOTHING_NORMAL) * t
            
            alpha = base_alpha
            if ball:
                bx, by, _ = ball
                x1, y1, x2, y2 = view_rect
                vw, vh = (x2 - x1), (y2 - y1)

                mx, my = vw * BALL_SAFE_MARGIN, vh * BALL_SAFE_MARGIN
                sx1, sx2 = x1 + mx, x2 - mx
                sy1, sy2 = y1 + my, y2 - my

                if not (sx1 <= bx <= sx2 and sy1 <= by <= sy2):
                    alpha = SMOOTHING_FAST

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
        disp = frame.copy()

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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            ph, pw = preview.shape[:2]
            preview_writer = cv2.VideoWriter(args.save_preview, fourcc, 30, (pw, ph))

        if args.save_broadcast and broadcast_writer is None:
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

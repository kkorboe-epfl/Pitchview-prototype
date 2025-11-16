import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# ---------------- ARGUMENTS ---------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--save-preview", type=str, default=None,
                    help="Save the panorama preview video to this file")
parser.add_argument("--save-broadcast", type=str, default=None,
                    help="Save the broadcast view video to this file")
args = parser.parse_args()


# ---------------- SETTINGS ---------------- #


VIDEO_PATH = "Footage16-11/pano/pano08.mp4"
PREVIEW_WIDTH = 1600
EXCLUSION_RATIO = 0.13  # top band fraction to ignore vertically

# how long we keep drawing the old ball if detection fails
MAX_MISSED_FRAMES = 10

# player detection parameters (tune for your footage)
PLAYER_MIN_AREA = 100      # min bounding box area (reduced, players are small)
PLAYER_MAX_AREA = 50000    # max bounding box area
PLAYER_MIN_ASPECT = 0.5    # very relaxed aspect ratio (h/w)
PLAYER_MAX_ASPECT = 5.0

# how close a player must be to ball centre (pixels)
PLAYER_NEAR_DIST = 400

# broadcast view parameters
BROADCAST_ASPECT = 16.0 / 9.0
MIN_VIEW_WIDTH = 600        # minimum broadcast rect width in pixels
MARGIN_FACTOR = 0.35        # margin as fraction of bbox size
MARGIN_PIXELS = 40          # extra constant margin

# camera smoothing parameters
SMOOTHING_NORMAL = 0.1     # slow, cinematic pan
SMOOTHING_FAST = 0.5        # fast catch-up when ball near edge
BALL_SAFE_MARGIN = 0.25     # ball should stay inside centre 50% of view

# YOLO player detector settings
PLAYER_MODEL_PATH = "yolov8s.pt"  # or yolov8s.pt if you can afford it
PERSON_CLASS_ID = 0               # "person" class in COCO
YOLO_CONF_THRESH = 0.2           # lower to pick up smaller players

# load YOLO model once
player_model = YOLO(PLAYER_MODEL_PATH)


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


def compute_view(ball, nearby, shape):
    h, w = shape[:2]
    if ball is None:
        return np.array([0, 0, w, h], float)

    bx, by, r = ball
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

    mx = ww * MARGIN_FACTOR + MARGIN_PIXELS
    my = hh * MARGIN_FACTOR + MARGIN_PIXELS

    xmin -= mx
    xmax += mx
    ymin -= my
    ymax += my

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    # minimum width
    if (xmax - xmin) < MIN_VIEW_WIDTH:
        c = (xmin + xmax) / 2
        half = MIN_VIEW_WIDTH / 2
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

    # Writers (opened lazily later)
    preview_writer = None
    broadcast_writer = None

    last_pos = None
    last_radius = None
    missed = 0
    view_rect = None

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
        else:
            if last_pos and missed < MAX_MISSED_FRAMES:
                missed += 1
                bx, by = last_pos
                br = last_radius or 10
                ball = (bx, by, br)
            else:
                ball = None

        # --- Player detection ---
        players = detect_players_yolo(frame)

        # --- Nearby players ---
        near = choose_nearby(players, ball)

        # --- Compute target view ---
        target = compute_view(ball, near, frame.shape)

        if view_rect is None:
            view_rect = target.copy()
        else:
            # adaptive smoothing
            alpha = SMOOTHING_NORMAL
            if ball:
                bx, by, _ = ball
                x1, y1, x2, y2 = view_rect
                vw, vh = (x2 - x1), (y2 - y1)

                mx, my = vw * BALL_SAFE_MARGIN, vh * BALL_SAFE_MARGIN
                sx1, sx2 = x1 + mx, x2 - mx
                sy1, sy2 = y1 + my, y2 - my

                if not (sx1 <= bx <= sx2 and sy1 <= by <= sy2):
                    alpha = SMOOTHING_FAST

            view_rect = lerp_rect(view_rect, target, alpha)

        # clamp + int conversion
        x1, y1, x2, y2 = map(int, map(round, view_rect))
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # --- Crops ---
        broadcast_crop = frame[y1:y2, x1:x2]
        broadcast_view = cv2.resize(broadcast_crop, (960, 540))

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

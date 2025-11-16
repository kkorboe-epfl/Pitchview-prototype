import cv2
import numpy as np

VIDEO_PATH = "Footage16-11/pano/pano08.mp4"
PREVIEW_WIDTH = 1600
EXCLUSION_RATIO = 0.0  # top band fraction to ignore vertically

# how long we keep drawing the old ball if detection fails
MAX_MISSED_FRAMES = 10

# player detection parameters (tune for your footage)
PLAYER_MIN_AREA = 500      # min bounding box area
PLAYER_MAX_AREA = 20000    # max bounding box area
PLAYER_MIN_ASPECT = 1.2    # h/w for standing players
PLAYER_MAX_ASPECT = 4.0

# how close a player must be to ball centre (pixels)
PLAYER_NEAR_DIST = 140

# broadcast view parameters
BROADCAST_ASPECT = 16.0 / 9.0
MIN_VIEW_WIDTH = 320        # minimum broadcast rect width in pixels
SMOOTHING = 0.1             # 0..1, higher = faster camera movement
MARGIN_FACTOR = 0.35        # margin as fraction of bbox size
MARGIN_PIXELS = 40          # extra constant margin


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

    # exclude top band
    mask[0:exclusion_height, :] = 0

    # slight blur and morphology to stabilise
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

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 4 or radius > 70:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue

        candidates.append((int(x), int(y), int(radius), circularity, area))

    return candidates


def pick_best_candidate(candidates, last_pos=None):
    if not candidates:
        return None

    # score: favour circular and near last position
    best_score = -1
    best = None
    for (x, y, r, circ, area) in candidates:
        score = circ * 2.0  # base weight for circularity
        if last_pos is not None:
            lx, ly = last_pos
            dist = np.hypot(x - lx, y - ly)
            # penalise large jumps
            score -= dist * 0.01
        # also lightly favour reasonable size
        score += np.log(max(area, 1)) * 0.1

        if score > best_score:
            best_score = score
            best = (x, y, r)

    return best


def detect_players(frame, back_sub, exclusion_ratio=EXCLUSION_RATIO):
    """
    Very simple player detector using background subtraction + contour filtering.
    Returns list of dicts: {"bbox": (x1, y1, x2, y2), "centre": (cx, cy)}
    """
    h, w = frame.shape[:2]
    exclusion_height = int(h * exclusion_ratio)

    fg_mask = back_sub.apply(frame)

    # remove shadows if using MOG2 (shadows ~ 127)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    # exclude top band for players too, if needed
    fg_mask[0:exclusion_height, :] = 0

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    players = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        if area < PLAYER_MIN_AREA or area > PLAYER_MAX_AREA:
            continue

        aspect = h_box / float(w_box + 1e-6)
        if aspect < PLAYER_MIN_ASPECT or aspect > PLAYER_MAX_ASPECT:
            continue

        cx = x + w_box // 2
        cy = y + h_box // 2
        players.append({
            "bbox": (x, y, x + w_box, y + h_box),
            "centre": (cx, cy)
        })

    return players


def choose_nearby_players(players, ball, max_players=2):
    """
    From all detected players, choose up to max_players closest to the ball.
    """
    if ball is None:
        return []

    bx, by, _ = ball
    dists = []
    for p in players:
        cx, cy = p["centre"]
        dist = np.hypot(cx - bx, cy - by)
        if dist <= PLAYER_NEAR_DIST:
            dists.append((dist, p))

    dists.sort(key=lambda x: x[0])
    return [p for (_, p) in dists[:max_players]]


def compute_target_view_rect(ball, nearby_players, frame_shape):
    """
    Compute the desired broadcast view rect (x1, y1, x2, y2) that contains:
    - the ball
    - up to two nearby players
    plus some margin, and with a fixed aspect ratio.
    """
    h, w = frame_shape[:2]

    # default: full frame if we do not have ball
    if ball is None:
        return np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)

    bx, by, r = ball

    # start from ball point
    xs = [bx]
    ys = [by]

    for p in nearby_players:
        x1, y1, x2, y2 = p["bbox"]
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    x_min = float(min(xs))
    x_max = float(max(xs))
    y_min = float(min(ys))
    y_max = float(max(ys))

    # add margin
    width = x_max - x_min
    height = y_max - y_min

    margin_x = width * MARGIN_FACTOR + MARGIN_PIXELS
    margin_y = height * MARGIN_FACTOR + MARGIN_PIXELS

    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    # clamp to frame
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(float(w), x_max)
    y_max = min(float(h), y_max)

    # enforce minimum width
    if (x_max - x_min) < MIN_VIEW_WIDTH:
        cx = 0.5 * (x_min + x_max)
        half_w = MIN_VIEW_WIDTH / 2.0
        x_min = cx - half_w
        x_max = cx + half_w
        x_min = max(0.0, x_min)
        x_max = min(float(w), x_max)

    # enforce aspect ratio (16:9) by growing rect if needed
    width = x_max - x_min
    height = y_max - y_min
    current_aspect = width / float(height + 1e-6)

    if current_aspect > BROADCAST_ASPECT:
        # too wide, increase height
        desired_height = width / BROADCAST_ASPECT
        extra = desired_height - height
        y_min -= extra / 2.0
        y_max += extra / 2.0
    else:
        # too tall, increase width
        desired_width = height * BROADCAST_ASPECT
        extra = desired_width - width
        x_min -= extra / 2.0
        x_max += extra / 2.0

    # clamp again
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(float(w), x_max)
    y_max = min(float(h), y_max)

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def lerp_rect(current, target, alpha):
    return current * (1.0 - alpha) + target * alpha


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video")
        return

    # background subtractor for player detection
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    last_pos = None
    last_radius = None
    missed_frames = 0

    current_view_rect = None  # [x1, y1, x2, y2] as float

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        exclusion_height = int(h * EXCLUSION_RATIO)

        # --- 1. Ball detection / tracking ---
        candidates = detect_red_candidates(frame)
        ball = pick_best_candidate(candidates, last_pos)

        if ball is not None:
            x, y, r = ball
            last_pos = (x, y)
            last_radius = r
            missed_frames = 0
        else:
            # no detection this frame, keep last known for a bit
            if missed_frames < MAX_MISSED_FRAMES and last_pos is not None:
                missed_frames += 1
                x, y = last_pos
                r = last_radius if last_radius is not None else 10
                ball = (x, y, r)
            else:
                ball = None

        # --- 2. Player detection ---
        players = detect_players(frame, back_sub, exclusion_ratio=EXCLUSION_RATIO)

        # --- 3. Choose nearby players to include in broadcast view ---
        nearby_players = choose_nearby_players(players, ball, max_players=2)

        # --- 4. Compute desired broadcast view rect ---
        target_view_rect = compute_target_view_rect(ball, nearby_players, frame.shape)

        # initialise camera rect on first frame
        if current_view_rect is None:
            current_view_rect = target_view_rect.copy()
        else:
            current_view_rect = lerp_rect(current_view_rect, target_view_rect, SMOOTHING)

        # convert rect to integers and clamp
        x1, y1, x2, y2 = current_view_rect
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(x1 + 1, min(w, round(x2))))
        y2 = int(max(y1 + 1, min(h, round(y2))))

        broadcast_crop = frame[y1:y2, x1:x2]
        # resize broadcast view to something like 960x540 (16:9)
        broadcast_view = cv2.resize(broadcast_crop, (960, 540))

        # --- 5. Draw overlays on full panorama for debugging ---
        display = frame.copy()

        # draw ball
        if ball is not None:
            bx, by, br = ball
            cv2.circle(display, (bx, by), br, (0, 255, 255), 2)
            cv2.circle(display, (bx, by), 3, (0, 255, 0), -1)

        # draw player boxes
        for p in players:
            x1p, y1p, x2p, y2p = p["bbox"]
            colour = (255, 255, 0)
            if p in nearby_players:
                colour = (0, 0, 255)  # highlight nearby ones
            cv2.rectangle(display, (x1p, y1p), (x2p, y2p), colour, 2)

        # draw current broadcast rect on panorama
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # draw exclusion zone (if used)
        cv2.rectangle(display, (0, 0), (w - 1, exclusion_height - 1), (255, 0, 0), 2)

        # resize panorama for viewing
        scale = PREVIEW_WIDTH / float(w)
        preview = cv2.resize(display, (PREVIEW_WIDTH, int(h * scale)))

        cv2.imshow("Panorama view (ball + players + broadcast box)", preview)
        cv2.imshow("Broadcast view", broadcast_view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

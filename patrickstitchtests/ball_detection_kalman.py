import cv2
import numpy as np

VIDEO_PATH = "Footage16-11/pano/pano08.mp4"
PREVIEW_WIDTH = 1600
EXCLUSION_RATIO = 0.0  # top 10%
MAX_MISSED_FRAMES = 20

# New tuning parameters
MAX_DET_DIST = 120        # max allowed distance (pixels) from predicted position
MAX_RADIUS_SCALE = 1.8    # radius can grow/shrink at most this factor in one step

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


def pick_best_candidate(candidates, predicted_pos=None, last_radius=None):
    if not candidates:
        return None

    best_score = -1e9
    best = None

    for (x, y, r, circ, area) in candidates:
        score = circ * 2.0 + np.log(max(area, 1)) * 0.1

        # Prefer similar radius to last known radius
        if last_radius is not None:
            radius_ratio = r / float(last_radius + 1e-6)
            score -= abs(np.log(radius_ratio)) * 2.0  # penalise large radius changes

        if predicted_pos is not None:
            px, py = predicted_pos
            dist = np.hypot(x - px, y - py)
            # strong penalty for being far away
            score -= (dist ** 2) * 0.0005

        if score > best_score:
            best_score = score
            best = (x, y, r)

    return best


def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    # Ball motion is fairly smooth → low process noise
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3

    # Detections are a bit noisy → somewhat higher measurement noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

    return kf


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video")
        return

    kf = create_kalman_filter()
    kalman_initialised = False
    last_radius = 12
    missed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        exclusion_height = int(h * EXCLUSION_RATIO)

        # 1) Predict
        if kalman_initialised:
            prediction = kf.predict()
            pred_x, pred_y = float(prediction[0]), float(prediction[1])
            predicted_pos = (pred_x, pred_y)
        else:
            predicted_pos = None

        # 2) Detect candidates
        candidates = detect_red_candidates(frame)

        # 3) Pick best candidate
        best_detection = pick_best_candidate(candidates, predicted_pos, last_radius)

        # 4) Gating: reject detections that are too far or too different in size
        accepted_detection = None
        if best_detection is not None:
            bx, by, br = best_detection
            if predicted_pos is not None and kalman_initialised:
                dist = np.hypot(bx - predicted_pos[0], by - predicted_pos[1])
                radius_ok = True
                if last_radius is not None and last_radius > 0:
                    radius_ratio = br / float(last_radius)
                    if radius_ratio > MAX_RADIUS_SCALE or radius_ratio < (1.0 / MAX_RADIUS_SCALE):
                        radius_ok = False

                if dist <= MAX_DET_DIST and radius_ok:
                    accepted_detection = best_detection
            else:
                # If not initialised yet, accept first sensible detection
                accepted_detection = best_detection

        # 5) Correct Kalman or just coast
        if accepted_detection is not None:
            x, y, r = accepted_detection
            measurement = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)

            if not kalman_initialised:
                kf.statePost = np.array([[np.float32(x)],
                                         [np.float32(y)],
                                         [0.0],
                                         [0.0]],
                                        dtype=np.float32)
                kalman_initialised = True
            else:
                kf.correct(measurement)

            last_radius = r
            missed_frames = 0
        else:
            if kalman_initialised:
                missed_frames += 1

        # 6) Decide what to draw
        display = frame.copy()
        ball_pos_to_draw = None
        radius_to_draw = last_radius

        if kalman_initialised and missed_frames <= MAX_MISSED_FRAMES:
            # Use state after correction (or prediction if no correction)
            # statePost is updated by correct, statePre holds prediction
            state = kf.statePost if accepted_detection is not None else kf.statePre
            x, y = int(state[0]), int(state[1])
            ball_pos_to_draw = (x, y)

        # 7) Draw ball
        if ball_pos_to_draw is not None:
            x, y = ball_pos_to_draw
            cv2.circle(display, (x, y), int(radius_to_draw), (0, 255, 255), 2)
            cv2.circle(display, (x, y), 3, (0, 255, 0), -1)

        # 8) Draw exclusion box
        cv2.rectangle(display, (0, 0), (w - 1, exclusion_height - 1), (255, 0, 0), 2)

        # 9) Resize for preview
        scale = PREVIEW_WIDTH / float(w)
        preview = cv2.resize(display, (PREVIEW_WIDTH, int(h * scale)))

        cv2.imshow("Panorama ball detection + tuned Kalman tracking", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

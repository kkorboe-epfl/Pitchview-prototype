import cv2
import numpy as np

VIDEO_PATH = "Footage16-11/pano/pano08.mp4"
PREVIEW_WIDTH = 1600
EXCLUSION_RATIO = 0.0  # top 10%

# how long we keep drawing the old ball if detection fails
MAX_MISSED_FRAMES = 10

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


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video")
        return

    last_pos = None
    last_radius = None
    missed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        exclusion_height = int(h * EXCLUSION_RATIO)

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

        display = frame.copy()

        # draw ball
        if ball is not None:
            x, y, r = ball
            cv2.circle(display, (x, y), r, (0, 255, 255), 2)
            cv2.circle(display, (x, y), 3, (0, 255, 0), -1)

        # draw exclusion zone
        cv2.rectangle(display, (0, 0), (w - 1, exclusion_height - 1), (255, 0, 0), 2)

        # resize for viewing
        scale = PREVIEW_WIDTH / float(w)
        preview = cv2.resize(display, (PREVIEW_WIDTH, int(h * scale)))

        cv2.imshow("Panorama ball detection + tracking", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

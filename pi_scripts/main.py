"""
Main service script for dual-camera recording, stitching, broadcast view and upload.

States (LED patterns, BCM pin 27):
- IDLE (script running, not recording): solid ON
- RECORDING: slow blink (5 s ON, 5 s OFF)
- STITCHING: medium blink (0.5 s ON, 0.5 s OFF)
- UPLOADING: fast blink (0.1 s ON, 0.1 s OFF)
- SHUTDOWN: LED solid ON until the Pi powers off

Button (BCM pin 17):
- Short press:
    - IDLE      -> start recording
    - RECORDING -> stop recording, then stitch pano, generate broadcast, then upload both
- Long press (hold > 2.5 s):
    - stop recording if active, then power off the Pi safely
"""

import os
import sys
import time
import signal
import threading
import subprocess
import json
import queue
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

import RPi.GPIO as GPIO

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

try:
    from picamera2.outputs import FfmpegOutput
    HAVE_FFMPEG = True
except Exception:  # pragma: no cover
    HAVE_FFMPEG = False
    FfmpegOutput = None  # type: ignore


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

LED_PIN = 27
BUTTON_PIN = 17

# Recording parameters (adapt as you wish)
CAM_INDEXES = [1, 0]          # camera 0 = "left", camera 1 = "right"
FRAME_WIDTH = 2560
FRAME_HEIGHT = 1440
FRAME_FPS = 24
BITRATE = 8_000_000

# Paths
RECORD_ROOT = Path("/mnt/ssd/recordings")
PANOS_ROOT = Path("/mnt/ssd/panoramas")
BROADCAST_ROOT = Path("/mnt/ssd/broadcast")  # new: where broadcast videos are stored

CALIBRATION_JSON = Path("/home/stainer/pitchview/rig_calibration.json")  # adjust
BROADCAST_SCRIPT = Path("/home/stainer/pitchview/broadcast_yolo.py")  # adjust

# Google Drive upload via rclone
# You must have rclone configured with a remote called "gdrive" and a folder "pi-panoramas"
RCLONE_REMOTE_PATH = "gdrive:pi-panoramas"  # remote:folder

# Button press behaviour
LONG_PRESS_SECONDS = 2.5

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------

STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_STITCHING = "stitching"
STATE_UPLOADING = "uploading"
STATE_SHUTDOWN = "shutdown"

_state_lock = threading.Lock()
_current_state = STATE_IDLE

_exit_requested = False

# For recording control
_record_stop_event = None          # type: threading.Event | None
_record_thread = None              # type: threading.Thread | None
_record_result = {}                # will store {"left": Path, "right": Path, "base": str}

# Worker stop event (LED + button threads)
_workers_stop_event = threading.Event()

# Queue of button events ("short", "long")
_button_events: "queue.Queue[str]" = queue.Queue()


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    if not os.access(p, os.W_OK):
        raise RuntimeError(f"Directory not writable: {p}")


def set_state(new_state: str) -> None:
    global _current_state
    with _state_lock:
        if _current_state != new_state:
            print(f"[INFO] State change: {_current_state} -> {new_state}")
        _current_state = new_state


def get_state() -> str:
    with _state_lock:
        return _current_state


# ---------------------------------------------------------------------------
# LED WORKER
# ---------------------------------------------------------------------------

def led_worker(stop_event: threading.Event) -> None:
    """Blink patterns depending on current state."""
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.HIGH)

    def check_state() -> str:
        return get_state()

    while not stop_event.is_set():
        state = check_state()

        if state == STATE_IDLE:
            # Solid ON, but re-check state fairly often
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.2)

        elif state == STATE_RECORDING:
            # Slow blink: ~3 s ON, ~3 s OFF (adjusted here; comment above says 5 s)
            for phase in ("on", "off"):
                for _ in range(int(3.0 / 0.1)):
                    if stop_event.is_set() or check_state() != STATE_RECORDING:
                        break
                    GPIO.output(LED_PIN, GPIO.HIGH if phase == "on" else GPIO.LOW)
                    time.sleep(0.1)
                if stop_event.is_set() or check_state() != STATE_RECORDING:
                    break

        elif state == STATE_STITCHING:
            # Medium blink: 0.5 s ON, 0.5 s OFF
            for phase in ("on", "off"):
                if stop_event.is_set() or check_state() != STATE_STITCHING:
                    break
                GPIO.output(LED_PIN, GPIO.HIGH if phase == "on" else GPIO.LOW)
                time.sleep(0.5)

        elif state == STATE_UPLOADING:
            # Fast blink: 0.1 s ON, 0.1 s OFF
            for phase in ("on", "off"):
                if stop_event.is_set() or check_state() != STATE_UPLOADING:
                    break
                GPIO.output(LED_PIN, GPIO.HIGH if phase == "on" else GPIO.LOW)
                time.sleep(0.1)

        elif state == STATE_SHUTDOWN:
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.2)
        else:
            # Unknown state – keep LED on as a safe default
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.2)

    # When stopping workers, you may choose to turn LED off
    GPIO.output(LED_PIN, GPIO.LOW)


# ---------------------------------------------------------------------------
# BUTTON WORKER
# ---------------------------------------------------------------------------
def button_worker(stop_event: threading.Event, events: "queue.Queue[str]") -> None:
    """Detect short, long, and triple-press (3 shorts) on BUTTON_PIN and push to queue."""
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    pressed = False
    press_time = None

    # multi-click state
    click_count = 0
    last_click_time = None
    MULTI_GAP = 0.8  # seconds between clicks to treat as part of same sequence

    while not stop_event.is_set():
        level = GPIO.input(BUTTON_PIN)  # 1 = not pressed (pull-up), 0 = pressed

        if level == 0 and not pressed:
            # Button just pressed
            pressed = True
            press_time = time.time()

        elif level == 0 and pressed:
            # Still held; wait for release
            pass

        elif level == 1 and pressed:
            # Just released
            if press_time is not None:
                duration = time.time() - press_time
                if duration >= LONG_PRESS_SECONDS:
                    print(f"[BUTTON] Long press detected ({duration:.2f} s)")
                    events.put("long")
                    # cancel any pending multi-click sequence
                    click_count = 0
                    last_click_time = None
                elif duration >= 0.2:  # ignore very short bounces
                    print(f"[BUTTON] Short press detected ({duration:.2f} s)")
                    click_count += 1
                    last_click_time = time.time()
            pressed = False
            press_time = None

        # Handle short vs triple dispatch with timeout
        if click_count > 0 and last_click_time is not None:
            if time.time() - last_click_time > MULTI_GAP:
                if click_count >= 3:
                    print(f"[BUTTON] Interpreted as TRIPLE click ({click_count} short presses).")
                    events.put("triple")
                else:
                    # 1 or 2 quick presses => send that many 'short' events
                    for _ in range(click_count):
                        events.put("short")
                click_count = 0
                last_click_time = None

        time.sleep(0.05)


# ---------------------------------------------------------------------------
# PICAMERA2 RECORDING
# ---------------------------------------------------------------------------

def create_cam(idx: int, w: int, h: int, fps: int) -> Picamera2:
    cam = Picamera2(idx)
    cfg = cam.create_video_configuration(
        main={"size": (w, h), "format": "YUV420"},
        controls={
            "FrameDurationLimits": (int(1e6 / fps), int(1e6 / fps)),
            "NoiseReductionMode": 0,
            "AwbEnable": True,
            "AeEnable": True,
        },
        buffer_count=16,
    )
    cam.configure(cfg)
    return cam

def record_session(stop_event: threading.Event, result_dict: dict) -> None:
    """
    Record both cameras until stop_event is set.
    Saves mp4 (or .h264) files in RECORD_ROOT and updates result_dict with:
      {"left": Path, "right": Path, "base": str}
    """
    ensure_dir(RECORD_ROOT)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = ts

    cams = []
    encs = []
    outs = []
    files = []

    try:
        for idx in CAM_INDEXES:
            cam = create_cam(idx, FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS)
            cams.append(cam)

            enc = H264Encoder(bitrate=BITRATE)
            encs.append(enc)

            if HAVE_FFMPEG and FfmpegOutput is not None:
                out_path = RECORD_ROOT / f"{base_name}_cam{idx}.mp4"
                out = FfmpegOutput(str(out_path))
            else:
                out_path = RECORD_ROOT / f"{base_name}_cam{idx}.h264"
                out = FileOutput(str(out_path))

            outs.append(out)
            files.append(out_path)

        # Start cameras and encoders
        for cam in cams:
            cam.start()
        time.sleep(0.2)
        for cam, enc, out in zip(cams, encs, outs):
            cam.start_recording(enc, out)

        print("[REC] Recording started")
        while not stop_event.is_set():
            time.sleep(0.1)

        print("[REC] Stop requested")

    except Exception as e:
        print(f"[ERROR] Recording error: {e}", file=sys.stderr)
    finally:
        # Stop recording & cameras
        for cam in cams:
            try:
                cam.stop_recording()
            except Exception:
                pass

        for cam in cams:
            try:
                cam.stop()
            except Exception:
                pass

        # NEW: fully release cameras
        for cam in cams:
            try:
                cam.close()
            except Exception:
                pass

        # NEW: release global libcamera resources so we can re-initialise cleanly
        try:
            Picamera2.global_shutdown()
        except Exception:
            pass

        # Update result_dict if we have files
        if len(files) >= 2:
            # assume index 0 is "left", 1 is "right"
            result_dict["left"] = files[0]
            if len(files) > 1:
                result_dict["right"] = files[1]
            result_dict["base"] = base_name

        for p in files:
            print(f"[REC] Saved: {p}")

# ---------------------------------------------------------------------------
# STITCHING (USING PRE-SAVED JSON CALIBRATION, WITH AUTO-CROP & EDGE BLEND)
# ---------------------------------------------------------------------------

def load_calibration(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H = np.array(data["H"], dtype=np.float32)
    offset = tuple(int(v) for v in data["offset"])
    pano_size = tuple(int(v) for v in data["pano_size"])
    used_affine = bool(data.get("used_affine", False))
    return H, offset, pano_size, used_affine, data


def auto_crop_black_borders(image: np.ndarray, threshold: int = 30):
    """
    Detect black borders and return crop coordinates (x, y, w, h).
    Finds the tightest bounding box around non-black content.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    row_counts = np.sum(gray > threshold, axis=1)
    col_counts = np.sum(gray > threshold, axis=0)

    # A row/column is considered content if >50% of its pixels exceed threshold
    content_rows = np.where(row_counts > w * 0.5)[0]
    content_cols = np.where(col_counts > h * 0.5)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return 0, 0, image.shape[1], image.shape[0]

    y = content_rows[0]
    y_end = content_rows[-1] + 1
    x = content_cols[0]
    x_end = content_cols[-1] + 1

    return x, y, x_end - x, y_end - y


def stitch_pair(frameL,
                frameR,
                H,
                offset,
                pano_size,
                left_alpha: float = 1.0,
                edge_blend_width: int = 50):
    """
    Apply precomputed homography and offset to stitch a pair of frames
    into a panoramic canvas of size pano_size, with feathered seam on the
    right edge of the left frame.
    """
    ox, oy = offset
    pano_w, pano_h = pano_size

    # translation to place left frame correctly in the pano coordinates
    T = np.array([[1, 0, ox],
                  [0, 1, oy],
                  [0, 0, 1]], dtype=np.float32)
    Hs = T @ H

    # warp right into the panorama frame
    base = cv2.warpPerspective(frameR, Hs, (pano_w, pano_h))

    # region where the left image should go
    hL, wL = frameL.shape[:2]
    x0, y0 = ox, oy
    x1, y1 = ox + wL, oy + hL

    # clamp ROI to canvas (safety)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(pano_w, x1), min(pano_h, y1)
    if x1c > x0c and y1c > y0c:
        # corresponding crop in the left frame
        lx0 = x0c - x0
        ly0 = y0c - y0
        lx1 = lx0 + (x1c - x0c)
        ly1 = ly0 + (y1c - y0c)

        roi_base = base[y0c:y1c, x0c:x1c]
        roi_left = frameL[ly0:ly1, lx0:lx1]

        # Create alpha mask for blending only the RIGHT edge of the left frame
        h_roi, w_roi = roi_left.shape[:2]
        alpha_mask = np.ones((h_roi, w_roi), dtype=np.float32) * left_alpha

        blend_w = min(edge_blend_width, w_roi // 2)

        for i in range(blend_w):
            # Fade from left_alpha at the edge to 0 as we go left
            alpha_mask[:, w_roi - 1 - i] = left_alpha * (i / float(blend_w))

        # Expand mask to 3 channels
        alpha_mask_3ch = np.stack([alpha_mask] * 3, axis=2)

        # Apply the feathered blend
        roi_base_f = roi_base.astype(np.float32)
        roi_left_f = roi_left.astype(np.float32)

        blended = roi_base_f * (1.0 - alpha_mask_3ch) + roi_left_f * alpha_mask_3ch
        roi_base[:] = blended.astype(np.uint8)

    return base


def create_writer(path: Path, size, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not vw.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return vw

# ---------------------------------------------------------------------------
# CALIBRATION HELPERS (homography + canvas + JSON save)
# ---------------------------------------------------------------------------

def save_calibration(path,
                     H,
                     offset,
                     pano_size,
                     used_affine,
                     cylindrical,
                     downscale):
    """Save calibration to a JSON text file (using only built-in Python types)."""
    H_list = np.asarray(H, dtype=float).tolist()
    off_list = [int(o) for o in offset]
    size_list = [int(s) for s in pano_size]

    data = {
        "H": H_list,
        "offset": off_list,
        "pano_size": size_list,
        "used_affine": bool(used_affine),
        "cylindrical": bool(cylindrical),
        "downscale": float(downscale),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[CALIB] Saved calibration to {path}")


def detect_and_match(grayA, grayB, max_feats=4000):
    orb = cv2.ORB_create(
        nfeatures=max_feats,
        fastThreshold=7,
        scaleFactor=1.2,
        nlevels=8
    )
    kpa, desca = orb.detectAndCompute(grayA, None)
    kpb, descb = orb.detectAndCompute(grayB, None)

    if desca is None or descb is None or len(kpa) < 8 or len(kpb) < 8:
        return [], [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(desca, descb, k=2)
    good = []
    for m, n in knn:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return kpa, kpb, good


def homography_is_sane(H):
    if H is None or not np.isfinite(H).all():
        return False
    a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    lin = np.array([[a, b], [c, d]], dtype=np.float64)
    det = np.linalg.det(lin)
    if det <= 0:
        return False
    s = np.linalg.svd(lin, compute_uv=False)
    if s.min() < 1e-12:
        return False
    s_ratio = s.max() / s.min()
    if s.max() > 5.0 or s.min() < 0.2:
        return False
    if s_ratio > 5.0:
        return False
    if abs(H[2, 0]) > 1e-3 or abs(H[2, 1]) > 1e-3:
        return False
    return True


def cylindrical_warp(img, f=None):
    """Simple cylindrical projection; f in pixels (approx focal length)."""
    h, w = img.shape[:2]
    if f is None:
        f = 30 * w
    y_i, x_i = np.indices((h, w), dtype=np.float32)
    x_c = x_i - w / 2
    y_c = y_i - h / 2
    theta = x_c / f
    h_ = y_c / np.sqrt(x_c ** 2 + f ** 2)
    x_map = f * np.tan(theta) + w / 2
    y_map = f * h_ + h / 2
    out = cv2.remap(
        img, x_map, y_map, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    mask = (out[..., 0] + out[..., 1] + out[..., 2] > 0).astype(np.uint8) * 255
    return out, mask


def compute_transform(firstL, firstR, use_cylindrical=True, downscale=0.5):
    """Compute a robust homography (or affine fallback) between left/right."""
    def resize(img, s):
        if s == 1.0:
            return img
        return cv2.resize(
            img,
            (int(img.shape[1] * s), int(img.shape[0] * s)),
            interpolation=cv2.INTER_AREA,
        )

    L = resize(firstL, downscale)
    R = resize(firstR, downscale)

    if use_cylindrical:
        Lc, _ = cylindrical_warp(L)
        Rc, _ = cylindrical_warp(R)
        Lm, Rm = Lc, Rc
    else:
        Lm, Rm = L, R

    h, w = Lm.shape[:2]

    # bias to inner edges
    roiL = Lm[:, w // 2 - w // 6: w]
    roiR = Rm[:, 0: w // 2 + w // 6]

    kpa, kpb, good = detect_and_match(
        cv2.cvtColor(roiL, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(roiR, cv2.COLOR_BGR2GRAY),
    )
    offsetLx = w // 2 - w // 6

    if len(good) < 12:
        kpa, kpb, good = detect_and_match(
            cv2.cvtColor(Lm, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(Rm, cv2.COLOR_BGR2GRAY),
        )
        offsetLx = 0

    if len(good) < 8:
        raise RuntimeError("Not enough matches to compute a transform")

    ptsL = np.float32([kpa[m.queryIdx].pt for m in good])
    ptsR = np.float32([kpb[m.trainIdx].pt for m in good])
    ptsL[:, 0] += offsetLx

    H_ds, mask = cv2.findHomography(
        ptsR, ptsL,
        cv2.RANSAC,
        3.0,
        maxIters=5000,
        confidence=0.995,
    )
    use_affine = False
    if not homography_is_sane(H_ds):
        A, inl = cv2.estimateAffine2D(
            ptsR, ptsL,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.995,
        )
        if A is None:
            raise RuntimeError("Failed to estimate a stable transform")
        H_ds = np.eye(3, dtype=np.float32)
        H_ds[:2, :] = A
        use_affine = True

    S = np.array([[1 / downscale, 0, 0],
                  [0, 1 / downscale, 0],
                  [0, 0, 1]], dtype=np.float32)
    H = S @ H_ds @ np.linalg.inv(S)
    return H, use_affine


def build_canvas(firstL, firstR, H, max_w=6000, max_h=3000):
    """Compute panorama canvas size + offset for given H."""
    h, w = firstL.shape[:2]
    cornersR = np.float32(
        [[0, 0], [firstR.shape[1], 0],
         [firstR.shape[1], h], [0, h]]
    ).reshape(-1, 1, 2)
    warpedR = cv2.perspectiveTransform(cornersR, H)

    all_c = np.vstack([
        warpedR,
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2),
    ])
    x_min, y_min = np.floor(all_c.min(axis=0).ravel()).astype(np.int64)
    x_max, y_max = np.ceil(all_c.max(axis=0).ravel()).astype(np.int64)

    pano_w = int(x_max - x_min)
    pano_h = int(y_max - y_min)

    scale = min(1.0,
                max_w / max(pano_w, 1),
                max_h / max(pano_h, 1))
    if scale < 1.0:
        S = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0, 0, 1]], dtype=np.float32)
        H = S @ H @ np.linalg.inv(S)
        pano_w = int(pano_w * scale)
        pano_h = int(pano_h * scale)
        x_min = int(x_min * scale)
        y_min = int(y_min * scale)

    offset = (-x_min, -y_min)
    return H, offset, (pano_w, pano_h)

def recalibrate_rig_from_live(calib_path: Path,
                              use_cylindrical: bool = False,
                              downscale: float = 0.5) -> None:
    """
    Capture one pair of frames from physical cameras 0 (left) and 1 (right),
    compute a new homography and canvas, and overwrite the calibration JSON.

    Uses the same logic as stitch_export_transform.py but:
    - runs headless (no preview)
    - uses only a single snapshot pair
    """
    print("[CALIB] Starting live rig calibration...")

    # Capture one frame from each physical camera index
    camL = Picamera2(0)
    camR = Picamera2(1)

    cfgL = camL.create_still_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    )
    cfgR = camR.create_still_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    )

    camL.configure(cfgL)
    camR.configure(cfgR)

    camL.start()
    camR.start()
    time.sleep(1.0)  # let exposure/awb settle a bit

    firstL = camL.capture_array("main")
    firstR = camR.capture_array("main")

    camL.stop()
    camR.stop()
    camL.close()
    camR.close()

    try:
        Picamera2.global_shutdown()
    except Exception:
        pass

    # Normalise heights like in the desktop script
    if firstL.shape[0] != firstR.shape[0]:
        th = min(firstL.shape[0], firstR.shape[0])

        def rh(img):
            s = th / img.shape[0]
            return cv2.resize(
                img,
                (int(img.shape[1] * s), th),
                interpolation=cv2.INTER_AREA,
            )

        firstL = rh(firstL)
        firstR = rh(firstR)

    # Rotate 180° to match your stitching pipeline
    firstL = cv2.rotate(firstL, cv2.ROTATE_180)
    firstR = cv2.rotate(firstR, cv2.ROTATE_180)

    ds = max(0.2, min(1.0, downscale))

    H_raw, used_affine = compute_transform(
        firstL, firstR,
        use_cylindrical=use_cylindrical,
        downscale=ds,
    )
    H, offset, pano_size = build_canvas(
        firstL, firstR, H_raw,
        max_w=6000,
        max_h=3000,
    )

    save_calibration(
        str(calib_path),
        H,
        offset,
        pano_size,
        used_affine=used_affine,
        cylindrical=use_cylindrical,
        downscale=ds,
    )

    print(f"[CALIB] Done. pano_size={pano_size}, offset={offset}")


def stitch_videos(left_path: Path,
                  right_path: Path,
                  calib_path: Path,
                  output_path: Path,
                  fps: float) -> Path:
    """
    Stitch left & right videos into a panoramic mp4 using pre-saved calibration JSON,
    with auto-cropping of black borders and 180° rotation.
    """
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {calib_path}")

    print(f"[STITCH] Loading calibration from {calib_path}")
    H, offset, pano_size, used_affine, meta = load_calibration(calib_path)
    print(f"[STITCH] Transform: {'affine' if used_affine else 'homography'}; "
          f"pano size={pano_size}, offset={offset}")

    ensure_dir(output_path.parent)

    capL = cv2.VideoCapture(str(left_path))
    capR = cv2.VideoCapture(str(right_path))

    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Could not open one or both input videos for stitching")

    # Read first frames
    okL, firstL = capL.read()
    okR, firstR = capR.read()
    if not okL or not okR:
        capL.release()
        capR.release()
        raise RuntimeError("Could not read initial frames from both videos")

    # Normalise heights (if needed)
    hL, wL = firstL.shape[:2]
    hR, wR = firstR.shape[:2]

    h_ref = min(hL, hR)

    def resize_to_ref(img):
        if img.shape[0] == h_ref:
            return img
        s = h_ref / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h_ref), interpolation=cv2.INTER_AREA)

    firstL = resize_to_ref(firstL)
    firstR = resize_to_ref(firstR)
    # Rotate each camera frame 180° because cameras are mounted upside down
    firstL = cv2.rotate(firstL, cv2.ROTATE_180)
    firstR = cv2.rotate(firstR, cv2.ROTATE_180)


    # First stitched pano for cropping detection
    test_pano = stitch_pair(firstL, firstR, H, offset, pano_size,
                            left_alpha=1.0, edge_blend_width=50)

    # Auto-crop black borders from first stitched frame
    crop_x, crop_y, crop_w, crop_h = auto_crop_black_borders(test_pano)
    crop_region = (crop_x, crop_y, crop_w, crop_h)
    output_size = (crop_w, crop_h)

    print(f"[STITCH] Auto-crop: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
    print(f"[STITCH] Output size: {output_size}")

    writer = create_writer(output_path, output_size, fps)
    frames = 0
    t0 = time.time()

    try:
        # Process first frame
        pano = stitch_pair(firstL, firstR, H, offset, pano_size,
                           left_alpha=1.0, edge_blend_width=50)



        cx, cy, cw, ch = crop_region
        pano_crop = pano[cy:cy + ch, cx:cx + cw]

        writer.write(pano_crop)
        frames += 1

        while True:
            okL, fL = capL.read()
            okR, fR = capR.read()
            if not okL or not okR:
                break

            fL = resize_to_ref(fL)
            fR = resize_to_ref(fR)
            # Rotate each camera frame 180° before stitching
            fL = cv2.rotate(fL, cv2.ROTATE_180)
            fR = cv2.rotate(fR, cv2.ROTATE_180)


            pano = stitch_pair(fL, fR, H, offset, pano_size,
                               left_alpha=1.0, edge_blend_width=50)


            cx, cy, cw, ch = crop_region
            pano_crop = pano[cy:cy + ch, cx:cx + cw]

            writer.write(pano_crop)
            frames += 1

        dt = time.time() - t0
        if dt > 0:
            print(f"[STITCH] Done. {frames} frames in {dt:.1f} s "
                  f"({frames / dt:.1f} fps approx)")

    finally:
        capL.release()
        capR.release()
        writer.release()

    print(f"[STITCH] Output saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# BROADCAST VIEW GENERATION (YOLO SCRIPT)
# ---------------------------------------------------------------------------

def generate_broadcast_view(pano_path: Path, base_name: str) -> Path:
    """
    Call the broadcast_yolo.py script to generate a broadcast view
    from the panoramic video and save it under BROADCAST_ROOT.
    """
    ensure_dir(BROADCAST_ROOT)
    broadcast_out = BROADCAST_ROOT / f"{base_name}_broadcast.mp4"
    preview_out = BROADCAST_ROOT / f"{base_name}_preview.mp4"

    cmd = [
        "python3",
        str(BROADCAST_SCRIPT),
        "--video", str(pano_path),
        "--save-broadcast", str(broadcast_out),
        "--save-preview", str(preview_out),
    ]

    print(f"[BCAST] Running broadcast script:\n        {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print("[BCAST] Broadcast generation completed successfully.")
        else:
            print(f"[BCAST] Script exited with code {result.returncode}", file=sys.stderr)
            if result.stdout:
                print("[BCAST] STDOUT:", result.stdout, file=sys.stderr)
            if result.stderr:
                print("[BCAST] STDERR:", result.stderr, file=sys.stderr)
    except FileNotFoundError:
        print("[BCAST] python3 or broadcast script not found. "
              "Check BROADCAST_SCRIPT path and Python installation.",
              file=sys.stderr)

    return broadcast_out


# ---------------------------------------------------------------------------
# UPLOAD TO GOOGLE DRIVE (RCLONE)
# ---------------------------------------------------------------------------

def upload_to_gdrive(path: Path) -> None:
    """
    Upload given file to Google Drive using rclone.
    You must have rclone installed and a configured remote.
    """
    if not path.exists():
        print(f"[UPLOAD] File does not exist, skipping upload: {path}", file=sys.stderr)
        return

    print(f"[UPLOAD] Uploading {path} to {RCLONE_REMOTE_PATH}")
    cmd = ["rclone", "copy", str(path), RCLONE_REMOTE_PATH, "-v"]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print("[UPLOAD] Upload completed successfully.")
        else:
            print(f"[UPLOAD] rclone exited with code {result.returncode}", file=sys.stderr)
            if result.stdout:
                print("[UPLOAD] STDOUT:", result.stdout, file=sys.stderr)
            if result.stderr:
                print("[UPLOAD] STDERR:", result.stderr, file=sys.stderr)
    except FileNotFoundError:
        print("[UPLOAD] rclone not found. Please install and configure rclone.",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# BUTTON EVENT HANDLERS
# ---------------------------------------------------------------------------

def handle_short_press() -> None:
    """
    Short press:
      - IDLE      -> start recording
      - RECORDING -> stop recording, then stitch pano, generate broadcast & upload both
    """
    global _record_stop_event, _record_thread, _record_result

    state = get_state()

    if state == STATE_IDLE:
        # Start recording
        print("[FSM] Short press in IDLE: start recording.")
        _record_stop_event = threading.Event()
        _record_result = {}
        _record_thread = threading.Thread(
            target=record_session,
            args=(_record_stop_event, _record_result),
            daemon=True,
        )
        set_state(STATE_RECORDING)
        _record_thread.start()

    elif state == STATE_RECORDING:
        # Stop recording and process
        print("[FSM] Short press in RECORDING: stop and process.")
        if _record_stop_event is not None:
            _record_stop_event.set()
        if _record_thread is not None:
            _record_thread.join()
        print("[FSM] Recording thread joined.")

        left_path = _record_result.get("left")
        right_path = _record_result.get("right")
        base_name = _record_result.get("base")

        if not left_path or not right_path:
            print("[FSM] No recordings produced; returning to IDLE.", file=sys.stderr)
            set_state(STATE_IDLE)
            return

        left_path = Path(left_path)
        right_path = Path(right_path)

        # SWAP here: cam0 file becomes "right", cam1 file becomes "left"
        left_path, right_path = right_path, left_path

        if base_name is None:
            base_name = left_path.stem.split("_cam")[0]

        pano_path = PANOS_ROOT / f"{base_name}_pano.mp4"
        broadcast_path = None

        try:
            set_state(STATE_STITCHING)

            # Pano stitching (auto-crop + rotate)
            stitched = stitch_videos(
                left_path=left_path,
                right_path=right_path,
                calib_path=CALIBRATION_JSON,
                output_path=pano_path,
                fps=float(FRAME_FPS),
            )

            # Broadcast view generation from pano
            broadcast_path = generate_broadcast_view(stitched, base_name)

            # Upload pano + broadcast
            set_state(STATE_UPLOADING)
            upload_to_gdrive(stitched)
            if broadcast_path is not None:
                upload_to_gdrive(broadcast_path)

        except Exception as e:
            print(f"[ERROR] Processing failed: {e}", file=sys.stderr)

        set_state(STATE_IDLE)

    else:
        print(f"[FSM] Short press ignored in state: {state}")

def handle_triple_press() -> None:
    """
    Triple press:
      - In IDLE: capture live frames and recompute rig_calibration.json.
      - In other states: ignored.
    """
    state = get_state()
    if state != STATE_IDLE:
        print(f"[FSM] Triple press ignored in state: {state}")
        return

    print("[FSM] Triple press in IDLE: recalibrating rig from live cameras.")
    try:
        set_state(STATE_STITCHING)
        # You can turn cylindrical=True if you want the same behaviour
        # as the desktop script that used --cylindrical
        recalibrate_rig_from_live(
            CALIBRATION_JSON,
            use_cylindrical=True,
            downscale=0.5,
        )
    except Exception as e:
        print(f"[CALIB] Live calibration failed: {e}", file=sys.stderr)
    finally:
        set_state(STATE_IDLE)

def handle_long_press() -> None:
    """
    Long press: stop recording if needed, then power off the Pi safely.
    """
    global _record_stop_event, _record_thread

    state = get_state()
    print(f"[FSM] Long press in state {state}: shutting down.")

    if state == STATE_RECORDING and _record_thread is not None:
        print("[FSM] Stopping recording before shutdown.")
        if _record_stop_event is not None:
            _record_stop_event.set()
        _record_thread.join(timeout=10.0)
        print("[FSM] Recording thread joined for shutdown.")

    set_state(STATE_SHUTDOWN)

    # Ask workers to stop
    _workers_stop_event.set()
    time.sleep(0.5)

    # Clean up GPIO before poweroff
    try:
        GPIO.cleanup()
    except Exception:
        pass

    # Power off the Pi
    print("[FSM] Calling 'sudo poweroff'")
    try:
        subprocess.run(["sudo", "poweroff"])
    except Exception as e:
        print(f"[ERROR] Failed to call poweroff: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# SIGNAL HANDLING
# ---------------------------------------------------------------------------

def _signal_handler(signum, frame):
    global _exit_requested
    print(f"[SIGNAL] Caught signal {signum}, exiting main loop.")
    _exit_requested = True
    _workers_stop_event.set()
    # If recording, request stop
    global _record_stop_event
    if _record_stop_event is not None:
        _record_stop_event.set()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    global _exit_requested

    # GPIO setup
    GPIO.setmode(GPIO.BCM)

    # Set up signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    # Start worker threads
    led_thread = threading.Thread(
        target=led_worker,
        args=(_workers_stop_event,),
        daemon=True,
    )
    btn_thread = threading.Thread(
        target=button_worker,
        args=(_workers_stop_event, _button_events),
        daemon=True,
    )
    led_thread.start()
    btn_thread.start()

    set_state(STATE_IDLE)
    print("[MAIN] Service started. Waiting for button presses.")

    try:
        while not _exit_requested:
            try:
                ev = _button_events.get(timeout=0.5)
            except queue.Empty:
                continue

            print(f"[MAIN] Event from queue: {ev}")  # optional debug

            if ev == "short":
                handle_short_press()
            elif ev == "long":
                handle_long_press()
                break
            elif ev == "triple":
                handle_triple_press()

    finally:
        print("[MAIN] Cleaning up...")
        _workers_stop_event.set()
        time.sleep(0.2)
        try:
            GPIO.cleanup()
        except Exception:
            pass
        print("[MAIN] Exit.")


if __name__ == "__main__":
    main()

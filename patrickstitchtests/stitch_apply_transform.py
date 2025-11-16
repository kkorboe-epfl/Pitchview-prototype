#!/usr/bin/env python3
import argparse
import sys
import time
from typing import Tuple, Optional

import json
import numpy as np
import cv2


def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def open_source(src: str, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    """Open file or camera index."""
    if is_int(src):
        cam = cv2.VideoCapture(int(src), cv2.CAP_ANY)
        if width:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # you can tweak FPS if needed
        # cam.set(cv2.CAP_PROP_FPS, 30)
    else:
        cam = cv2.VideoCapture(src)

    if not cam.isOpened():
        raise RuntimeError(f"Could not open source: {src}")
    return cam


def read_synced(capL: cv2.VideoCapture, capR: cv2.VideoCapture):
    """Read a frame from both sources; return False if either fails."""
    okL, fL = capL.read()
    okR, fR = capR.read()
    if not okL or not okR:
        return False, None, None
    return True, fL, fR


def load_calibration(path: str):
    """Load calibration (H, offset, pano_size, used_affine, meta) from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H = np.array(data["H"], dtype=np.float32)
    offset = tuple(int(v) for v in data["offset"])
    pano_size = tuple(int(v) for v in data["pano_size"])
    used_affine = bool(data.get("used_affine", False))
    return H, offset, pano_size, used_affine, data


def stitch_pair(frameL: np.ndarray,
                frameR: np.ndarray,
                H: np.ndarray,
                offset: Tuple[int, int],
                pano_size: Tuple[int, int],
                left_alpha: float = 1.0) -> np.ndarray:
    """
    Apply precomputed homography and offset to stitch a pair of frames
    into a panoramic canvas of size pano_size.
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

        if 0.0 <= left_alpha < 1.0:
            # alpha blend left over right (semi-transparent left)
            cv2.addWeighted(roi_base, 1.0 - left_alpha,
                            roi_left, left_alpha, 0.0, dst=roi_base)
        else:
            # opaque left
            roi_base[:] = roi_left

    return base


def writer_from_args(path: Optional[str],
                     size: Tuple[int, int],
                     fps: float) -> Optional[cv2.VideoWriter]:
    if not path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, size)
    if not vw.isOpened():
        raise RuntimeError(f"Could not open writer: {path}")
    return vw


def main():
    ap = argparse.ArgumentParser(
        description="Apply precomputed panoramic calibration (homography) to two streams."
    )
    ap.add_argument("--left", required=True,
                    help="Left input: file path or camera index")
    ap.add_argument("--right", required=True,
                    help="Right input: file path or camera index")
    ap.add_argument("--calib", required=True,
                    help="Calibration JSON file produced by the calibration script")
    ap.add_argument("--width", type=int, default=None,
                    help="Optional capture width hint")
    ap.add_argument("--height", type=int, default=None,
                    help="Optional capture height hint")
    ap.add_argument("--preview", action="store_true",
                    help="Show a live preview window")
    ap.add_argument("--output", type=str, default=None,
                    help="Output MP4 file (optional)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Output FPS (if writing to file)")
    ap.add_argument("--left-alpha", type=float, default=1.0,
                    help="Opacity of the left stream in [0..1] (e.g. 0.5)")

    args = ap.parse_args()

    # Load calibration
    H, offset, pano_size, used_affine, meta = load_calibration(args.calib)
    print(f"[info] Loaded calibration from {args.calib}")
    print(f"[info] Transform: {'affine' if used_affine else 'homography'}  "
          f"|  Panorama size: {pano_size}  |  Offset: {offset}")

    # Open sources
    capL = open_source(args.left, args.width, args.height)
    capR = open_source(args.right, args.width, args.height)

    # Read first frames to sanity-check sizes
    ok, fL, fR = read_synced(capL, capR)
    if not ok:
        print("Could not read initial frames from both sources", file=sys.stderr)
        sys.exit(1)

    # If heights differ slightly, resize right to match left.
    # Ideally your Pi feeds match the resolution used during calibration.
    hL, wL = fL.shape[:2]
    hR, wR = fR.shape[:2]

    if hL != hR:
        print(f"[warn] Height mismatch (left={hL}, right={hR}); "
              f"resizing right to match left height.", file=sys.stderr)
        scale = hL / float(hR)
        fR = cv2.resize(fR, (int(wR * scale), hL), interpolation=cv2.INTER_AREA)

    # Prepare writer (after we know pano size)
    vw = writer_from_args(args.output, pano_size, args.fps) if args.output else None

    # Prepare preview window
    if args.preview:
        disp_w = min(1600, pano_size[0])
        disp_h = int(disp_w * pano_size[1] / max(pano_size[0], 1))
        cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Panorama", max(640, disp_w), max(360, disp_h))

    t0 = time.time()
    frames = 0

    while True:
        ok, fL, fR = read_synced(capL, capR)
        if not ok:
            break

        # Keep heights matched as above
        hL, wL = fL.shape[:2]
        hR, wR = fR.shape[:2]
        if hL != hR:
            scale = hL / float(hR)
            fR = cv2.resize(fR, (int(wR * scale), hL), interpolation=cv2.INTER_AREA)

        pano = stitch_pair(fL, fR, H, offset, pano_size, left_alpha=args.left_alpha)

        frames += 1
        if frames % 10 == 0:
            fps_now = frames / (time.time() - t0 + 1e-9)
            cv2.putText(pano, f"{fps_now:.1f} fps",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)

        if vw is not None:
            vw.write(pano)

        if args.preview:
            # scale for display
            scale = min(1.0, 1600 / max(pano.shape[1], 1))
            if scale < 1.0:
                disp = cv2.resize(pano,
                                  (int(pano.shape[1] * scale),
                                   int(pano.shape[0] * scale)),
                                  interpolation=cv2.INTER_AREA)
            else:
                disp = pano

            cv2.imshow("Panorama", disp)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break

    capL.release()
    capR.release()
    if vw is not None:
        vw.release()
    if args.preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

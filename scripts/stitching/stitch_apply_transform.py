#!/usr/bin/env python3
"""
Apply pre-computed stitching transform to dual-camera video streams.

Sample usage:
  python scripts/stitching/stitch_apply_transform.py \
    --left data/raw/20251116_103024_left.mp4 \
    --right data/raw/20251116_103024_right.mp4 \
    --calib data/calibration/rig_calibration.json \
    --output output/stitched/20251116_103024_stitched.mp4
"""
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


def read_synced(capL: cv2.VideoCapture, capR: cv2.VideoCapture, offset: int = 0):
    """
    Read a frame from both sources with optional frame offset for sync.
    
    offset: number of frames to offset right camera (positive = right is ahead, skip frames)
            negative = left is ahead
    """
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


def auto_crop_black_borders(image: np.ndarray, threshold: int = 30, content_threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """
    Detect black borders and return crop coordinates (x, y, w, h).
    Finds the tightest bounding box around non-black content.
    
    Uses a stricter threshold to ensure all black borders (including bottom) are removed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find rows and columns that have enough non-black pixels
    # A row/column is considered "content" if more than content_threshold of pixels exceed threshold
    h, w = gray.shape
    row_counts = np.sum(gray > threshold, axis=1)
    col_counts = np.sum(gray > threshold, axis=0)
    
    # Find first and last content rows and columns
    content_rows = np.where(row_counts > w * content_threshold)[0]
    content_cols = np.where(col_counts > h * content_threshold)[0]
    
    if len(content_rows) == 0 or len(content_cols) == 0:
        return 0, 0, image.shape[1], image.shape[0]
    
    y = content_rows[0]
    y_end = content_rows[-1] + 1
    x = content_cols[0]
    x_end = content_cols[-1] + 1
    
    # Add extra crop to bottom to ensure black borders are fully removed
    crop_h = y_end - y
    extra_bottom_crop = int(crop_h * 0.15)  # Remove extra 15% from bottom
    y_end = max(y + 1, y_end - extra_bottom_crop)
    
    return x, y, x_end - x, y_end - y


def match_exposure(frameL: np.ndarray, frameR: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match exposure and color balance between left and right frames.
    Uses histogram matching to align brightness and color distributions.
    """
    # Convert to LAB color space for better color/brightness separation
    lab_L = cv2.cvtColor(frameL, cv2.COLOR_BGR2LAB)
    lab_R = cv2.cvtColor(frameR, cv2.COLOR_BGR2LAB)
    
    # Match each channel of right frame to left frame
    matched_channels = []
    for i in range(3):
        # Compute histograms
        hist_L = cv2.calcHist([lab_L], [i], None, [256], [0, 256])
        hist_R = cv2.calcHist([lab_R], [i], None, [256], [0, 256])
        
        # Compute CDFs
        cdf_L = hist_L.cumsum()
        cdf_R = hist_R.cumsum()
        
        # Normalize CDFs
        cdf_L = cdf_L / cdf_L[-1]
        cdf_R = cdf_R / cdf_R[-1]
        
        # Create lookup table for histogram matching
        lut = np.zeros(256, dtype=np.uint8)
        for j in range(256):
            # Find closest CDF value in L for each value in R
            idx = np.searchsorted(cdf_L, cdf_R[j])
            lut[j] = min(idx, 255)
        
        # Apply lookup table to right channel
        matched_channels.append(cv2.LUT(lab_R[:, :, i], lut))
    
    # Merge matched channels
    matched_lab_R = cv2.merge(matched_channels)
    matched_R = cv2.cvtColor(matched_lab_R, cv2.COLOR_LAB2BGR)
    
    return frameL, matched_R


def stitch_pair(frameL: np.ndarray,
                frameR: np.ndarray,
                H: np.ndarray,
                offset: Tuple[int, int],
                pano_size: Tuple[int, int],
                left_alpha: float = 1.0,
                edge_blend_width: int = 50,
                match_colors: bool = False) -> np.ndarray:
    """
    Apply precomputed homography and offset to stitch a pair of frames
    into a panoramic canvas of size pano_size.
    
    edge_blend_width: number of pixels to feather at the RIGHT edge of the left frame only
    match_colors: if True, apply exposure compensation to match camera colors/brightness
    """
    # Apply exposure matching if requested
    if match_colors:
        frameL, frameR = match_exposure(frameL, frameR)
    
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
        
        # Feather only the right edge
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


def writer_from_args(path: Optional[str],
                     size: Tuple[int, int],
                     fps: float) -> Optional[cv2.VideoWriter]:
    if not path:
        return None
    
    # Create output directory if it doesn't exist
    from pathlib import Path
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    ap.add_argument("--edge-blend", type=int, default=50,
                    help="Edge blend width in pixels for smoother seam at right edge (default: 50)")
    ap.add_argument("--auto-crop", action="store_true",
                    help="Automatically crop black borders from panorama")
    ap.add_argument("--crop-threshold", type=int, default=30,
                    help="Brightness threshold for detecting black borders (default: 30)")
    ap.add_argument("--crop-content-ratio", type=float, default=0.5,
                    help="Ratio of non-black pixels needed to consider a row/column as content (default: 0.5)")
    ap.add_argument("--match-colors", action="store_true",
                    help="Apply exposure compensation to match camera colors and brightness")
    ap.add_argument("--sync-offset", type=int, default=2,
                    help="Frame offset for sync: positive if right camera is ahead, negative if left is ahead (default: 2)")

    args = ap.parse_args()

    # Load calibration
    H, offset, pano_size, used_affine, meta = load_calibration(args.calib)
    print(f"[info] Loaded calibration from {args.calib}")
    print(f"[info] Transform: {'affine' if used_affine else 'homography'}  "
          f"|  Panorama size: {pano_size}  |  Offset: {offset}")

    # Open sources
    capL = open_source(args.left, args.width, args.height)
    capR = open_source(args.right, args.width, args.height)

    # Apply sync offset by skipping frames
    if args.sync_offset > 0:
        print(f"[info] Skipping {args.sync_offset} frames from right camera for sync")
        for _ in range(args.sync_offset):
            capR.read()
    elif args.sync_offset < 0:
        print(f"[info] Skipping {-args.sync_offset} frames from left camera for sync")
        for _ in range(-args.sync_offset):
            capL.read()

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

    # Detect crop region from first stitched frame if auto-crop enabled
    crop_region = None
    output_size = pano_size
    
    if args.auto_crop:
        test_pano = stitch_pair(fL, fR, H, offset, pano_size, 
                               left_alpha=args.left_alpha,
                               edge_blend_width=args.edge_blend,
                               match_colors=args.match_colors)
        crop_x, crop_y, crop_w, crop_h = auto_crop_black_borders(
            test_pano, 
            threshold=args.crop_threshold,
            content_threshold=args.crop_content_ratio
        )
        crop_region = (crop_x, crop_y, crop_w, crop_h)
        output_size = (crop_w, crop_h)
        print(f"[info] Auto-crop detected: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        print(f"[info] Output size: {output_size}")

    # Prepare writer (after we know output size)
    vw = writer_from_args(args.output, output_size, args.fps) if args.output else None

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

        pano = stitch_pair(fL, fR, H, offset, pano_size, 
                          left_alpha=args.left_alpha,
                          edge_blend_width=args.edge_blend,
                          match_colors=args.match_colors)

        # Apply crop if enabled
        if crop_region is not None:
            cx, cy, cw, ch = crop_region
            pano = pano[cy:cy+ch, cx:cx+cw]

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

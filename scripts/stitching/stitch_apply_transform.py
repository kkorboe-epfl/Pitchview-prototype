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
    extra_bottom_crop = int(crop_h * 0.20)  # Remove extra 20% from bottom
    y_end = max(y + 1, y_end - extra_bottom_crop)
    
    return x, y, x_end - x, y_end - y


def match_exposure(frameL: np.ndarray, frameR: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match exposure and color tint between left and right frames.
    Matches both luminance and color channels with gentle blending.
    """
    # Convert to LAB color space for better color/brightness separation
    lab_L = cv2.cvtColor(frameL, cv2.COLOR_BGR2LAB)
    lab_R = cv2.cvtColor(frameR, cv2.COLOR_BGR2LAB)
    
    # Match all three channels: L (luminance), A (green-magenta), B (blue-yellow)
    matched_channels = []
    
    for channel_idx in range(3):
        channel_L = lab_L[:, :, channel_idx]
        channel_R = lab_R[:, :, channel_idx]
        
        # Compute histograms
        hist_L = cv2.calcHist([channel_L], [0], None, [256], [0, 256])
        hist_R = cv2.calcHist([channel_R], [0], None, [256], [0, 256])
        
        # Compute CDFs
        cdf_L = hist_L.cumsum()
        cdf_R = hist_R.cumsum()
        
        # Normalize CDFs
        cdf_L = cdf_L / cdf_L[-1]
        cdf_R = cdf_R / cdf_R[-1]
        
        # Create lookup table for histogram matching
        lut = np.zeros(256, dtype=np.uint8)
        for j in range(256):
            idx = np.searchsorted(cdf_L, cdf_R[j])
            lut[j] = min(idx, 255)
        
        # Apply lookup table
        matched_channel = cv2.LUT(channel_R, lut)
        
        # Gentle blend: more matching for color channels (A, B), less for luminance (L)
        if channel_idx == 0:  # L channel (luminance)
            blend_strength = 0.5
        else:  # A and B channels (color tint)
            blend_strength = 0.7
        
        blended_channel = cv2.addWeighted(matched_channel, blend_strength, channel_R, 1.0 - blend_strength, 0)
        matched_channels.append(blended_channel.astype(np.uint8))
    
    # Merge all matched channels
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
                seam_x: Optional[int] = None) -> np.ndarray:
    """
    Apply precomputed homography and offset to stitch a pair of frames
    into a panoramic canvas of size pano_size.
    
    edge_blend_width: number of pixels to feather at the seam
    seam_x: X coordinate in panorama where the seam should be placed.
            Left frame shows to the left of this line, right frame to the right.
            If None, uses the natural overlap from the offset.
    """
    # Apply exposure matching (always enabled)
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
    
    # If seam_x is specified, use it to control the blend point
    if seam_x is not None:
        # Create a mask that shows left frame left of seam, right frame right of seam
        blend_mask = np.zeros((pano_h, pano_w), dtype=np.float32)
        
        # Left of seam: full left frame (1.0)
        blend_mask[:, :seam_x - edge_blend_width // 2] = 1.0
        
        # Blend zone: gradient from left to right
        blend_start = seam_x - edge_blend_width // 2
        blend_end = seam_x + edge_blend_width // 2
        blend_start = max(0, blend_start)
        blend_end = min(pano_w, blend_end)
        
        for x in range(blend_start, blend_end):
            alpha = 1.0 - (x - blend_start) / float(edge_blend_width)
            blend_mask[:, x] = alpha
        
        # Right of seam: no left frame (0.0) - already initialized to 0
        
        # Apply left frame with the blend mask
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(pano_w, x1), min(pano_h, y1)
        
        if x1c > x0c and y1c > y0c:
            lx0 = x0c - x0
            ly0 = y0c - y0
            lx1 = lx0 + (x1c - x0c)
            ly1 = ly0 + (y1c - y0c)
            
            roi_base = base[y0c:y1c, x0c:x1c]
            roi_left = frameL[ly0:ly1, lx0:lx1]
            roi_mask = blend_mask[y0c:y1c, x0c:x1c]
            
            # Blend
            roi_base_f = roi_base.astype(np.float32)
            roi_left_f = roi_left.astype(np.float32)
            roi_mask_3ch = np.stack([roi_mask] * 3, axis=2)
            
            blended = roi_base_f * (1.0 - roi_mask_3ch) + roi_left_f * roi_mask_3ch
            base[y0c:y1c, x0c:x1c] = blended.astype(np.uint8)
        
        return base

    # Original blending logic when seam_x is not specified
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

        # Create alpha mask for blending
        h_roi, w_roi = roi_left.shape[:2]
        
        # Apply the blended overlay
        roi_base_f = roi_base.astype(np.float32)
        roi_left_f = roi_left.astype(np.float32)
        
        if left_alpha >= 1.0:
            # Full opacity: left frame completely replaces the base
            base[y0c:y1c, x0c:x1c] = roi_left
        else:
            # Create alpha mask
            alpha_mask = np.ones((h_roi, w_roi), dtype=np.float32) * left_alpha
            
            # Feather only the right edge for smooth transition
            blend_w = min(edge_blend_width, w_roi // 2)
            
            for i in range(blend_w):
                # Fade from left_alpha to 0 at the right edge
                fade_factor = i / float(blend_w)
                alpha_mask[:, w_roi - 1 - i] = left_alpha * fade_factor
            
            # Expand mask to 3 channels
            alpha_mask_3ch = np.stack([alpha_mask] * 3, axis=2)
            
            # Blend: base * (1 - alpha) + left * alpha
            blended = roi_base_f * (1.0 - alpha_mask_3ch) + roi_left_f * alpha_mask_3ch
            base[y0c:y1c, x0c:x1c] = blended.astype(np.uint8)

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
    ap.add_argument("--fps", type=float, default=None,
                    help="Output FPS (if writing to file). If not specified, uses input video FPS")
    ap.add_argument("--left-alpha", type=float, default=0.5,
                    help="Opacity of the left stream in [0..1] (e.g. 0.5)")
    ap.add_argument("--edge-blend", type=int, default=50,
                    help="Edge blend width in pixels for smoother seam at right edge (default: 50)")
    ap.add_argument("--seam-x", type=int, default=None,
                    help="X coordinate in panorama where seam should be placed (optional). If not set, uses natural overlap.")
    ap.add_argument("--crop-threshold", type=int, default=30,
                    help="Brightness threshold for detecting black borders (default: 30)")
    ap.add_argument("--crop-content-ratio", type=float, default=0.5,
                    help="Ratio of non-black pixels needed to consider a row/column as content (default: 0.5)")
    ap.add_argument("--sync-offset", type=int, default=0,
                    help="Frame offset for sync: positive if right camera is behind, negative if left is behind (default: 1)")

    args = ap.parse_args()

    # Load calibration
    H, offset, pano_size, used_affine, meta = load_calibration(args.calib)
    print(f"[info] Loaded calibration from {args.calib}")
    print(f"[info] Transform: {'affine' if used_affine else 'homography'}  "
          f"|  Panorama size: {pano_size}  |  Offset: {offset}")

    # Open sources
    capL = open_source(args.left, args.width, args.height)
    capR = open_source(args.right, args.width, args.height)

    # Get FPS from input video if not specified
    if args.fps is None:
        detected_fps = capL.get(cv2.CAP_PROP_FPS)
        if detected_fps > 0:
            args.fps = detected_fps
            print(f"[info] Detected input FPS: {detected_fps:.2f}")
        else:
            args.fps = 30.0
            print(f"[warn] Could not detect FPS, using default: 30.0")
    else:
        print(f"[info] Using specified FPS: {args.fps}")

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

    # Detect crop region from first stitched frame (always enabled)
    crop_region = None
    output_size = pano_size
    
    test_pano = stitch_pair(fL, fR, H, offset, pano_size, 
                           left_alpha=args.left_alpha,
                           edge_blend_width=args.edge_blend,
                           seam_x=args.seam_x)
    crop_x, crop_y, crop_w, crop_h = auto_crop_black_borders(
        test_pano, 
        threshold=args.crop_threshold,
        content_threshold=args.crop_content_ratio
    )
    crop_region = (crop_x, crop_y, crop_w, crop_h)
    output_size = (crop_w, crop_h)
    print(f"[info] Auto-crop detected: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
    print(f"[info] Output size: {output_size}")
    
    if args.seam_x is not None:
        print(f"[info] Using custom seam position at x={args.seam_x}")

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
                          seam_x=args.seam_x)

        # Apply crop (always enabled)
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

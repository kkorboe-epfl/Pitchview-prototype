#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, cv2, math, time, argparse
import numpy as np

# -------------------- helpers --------------------

def log(*a):
    try: print(*a)
    except Exception: pass

def cylindrical_warp(img, fov_deg=150.0):
    """Cylindrical projection (good for ultra-wide). Returns (warped, mask)."""
    h, w = img.shape[:2]
    cx, cy = w/2.0, h/2.0
    f = (w/2.0) / math.tan(math.radians(fov_deg)/2.0)

    ys, xs = np.indices((h, w), dtype=np.float32)
    x = (xs - cx) / f
    y = (ys - cy) / f

    map_x = f*np.tan(x) + cx
    map_y = f*y/np.sqrt(x*x + 1.0) + cy

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y,
                     interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, mask

def hann_window(shape):
    """2D Hann window using NumPy (avoids OpenCV's >1 assertion)."""
    h, w = shape
    if h < 2 or w < 2:
        return np.ones((max(h, 2), max(w, 2)), dtype=np.float32)
    winy = np.hanning(h).astype(np.float32)[:, None]   # (h,1)
    winx = np.hanning(w).astype(np.float32)[None, :]   # (1,w)
    return (winy * winx)  # (h,w) float32

def preproc_grey(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    return g

def phase_corr_shift(left_edge, right_edge):
    """Estimate translation (dx, dy) to move RIGHT onto LEFT via phase correlation."""
    gL = preproc_grey(left_edge)
    gR = preproc_grey(right_edge)

    # crop to common size
    h = min(gL.shape[0], gR.shape[0])
    w = min(gL.shape[1], gR.shape[1])
    gL = gL[:h, :w]
    gR = gR[:h, :w]

    # ensure reasonable minimum size for stability
    min_side = 64
    if h < min_side or w < min_side:
        new_h = max(min_side, h)
        new_w = max(min_side, w)
        gL = cv2.resize(gL, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gR = cv2.resize(gR, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    win = hann_window((h, w)).astype(np.float32)
    (sx, sy), resp = cv2.phaseCorrelate(gL.astype(np.float32), gR.astype(np.float32), win)
    # returns shift to add to gR to align to gL
    return float(sx), float(sy), float(resp)

def refine_ecc(left_edge, right_edge, dx, dy, iters=200, eps=1e-5):
    """Optional ECC refinement in translation mode starting from (dx, dy)."""
    gL = preproc_grey(left_edge)
    gR = preproc_grey(right_edge)
    h = min(gL.shape[0], gR.shape[0])
    w = min(gL.shape[1], gR.shape[1])
    gL = gL[:h, :w]
    gR = gR[:h, :w]

    # keep sizes decent
    min_side = 64
    if h < min_side or w < min_side:
        new_h = max(min_side, h)
        new_w = max(min_side, w)
        gL = cv2.resize(gL, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gR = cv2.resize(gR, (new_w, new_h), interpolation=cv2.INTER_AREA)

    warp = np.array([[1, 0, dx],
                     [0, 1, dy]], dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(gL, gR, warp,
                                        motionType=cv2.MOTION_TRANSLATION,
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps),
                                        inputMask=None, gaussFiltSize=5)
        return float(warp[0,2]), float(warp[1,2]), float(cc)
    except cv2.error:
        return dx, dy, -1.0

def build_canvas_sizes(wL, hL, wR, hR, dx, dy):
    """Canvas that fits left at (0,0) and right translated by (dx,dy)."""
    min_x = min(0, int(math.floor(dx)))
    min_y = min(0, int(math.floor(dy)))
    max_x = max(wL, int(math.ceil(dx + wR)))
    max_y = max(hL, int(math.ceil(dy + hR)))
    tx, ty = -min_x, -min_y
    outW, outH = max_x - min_x, max_y - min_y
    return tx, ty, outW, outH

def feather_blend(base, over, mask, feather=60):
    if mask.ndim == 2:
        mask = cv2.merge([mask, mask, mask])
    if feather > 0:
        k = max(1, int(feather))
        if k % 2 == 0: k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    a = (mask.astype(np.float32)/255.0)
    out = over.astype(np.float32)*a + base.astype(np.float32)*(1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)

def constrained_preview(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    s = min(max_w/float(w), max_h/float(h), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Two-stream panorama via cylindrical warp + single translation (phase correlation)")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", default="stitched.mp4")
    ap.add_argument("--fov", type=float, default=150.0, help="approx horizontal FoV per lens in degrees")
    ap.add_argument("--downscale", type=float, default=1.0, help="optional pre-scale (e.g. 0.7)")
    ap.add_argument("--strip", type=float, default=0.30, help="edge strip fraction for matching (0.15–0.45 works)")
    ap.add_argument("--ecc", action="store_true", help="refine translation using ECC")
    ap.add_argument("--flip", action="store_true", help="swap left/right roles")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--max-preview-w", type=int, default=1600)
    ap.add_argument("--max-preview-h", type=int, default=900)
    ap.add_argument("--feather", type=int, default=70)
    ap.add_argument("--limit", type=int, default=0, help="limit frames (0 = all)")
    args = ap.parse_args()

    srcL = args.right if args.flip else args.left
    srcR = args.left if args.flip else args.right

    capL = cv2.VideoCapture(srcL)
    capR = cv2.VideoCapture(srcR)
    if not capL.isOpened() or not capR.isOpened():
        sys.exit("Could not open one of the videos")

    okL, fL = capL.read()
    okR, fR = capR.read()
    if not okL or not okR:
        sys.exit("Could not read first frames")

    if args.downscale != 1.0 and args.downscale > 0:
        fL = cv2.resize(fL, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
        fR = cv2.resize(fR, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)

    wL, _ = cylindrical_warp(fL, fov_deg=args.fov)
    wR, _ = cylindrical_warp(fR, fov_deg=args.fov)
    log("[warp] Cylindrical enabled")

    hL, wLw = wL.shape[:2]
    hR, wRw = wR.shape[:2]
    sfrac = max(0.1, min(0.45, args.strip))
    sL = int(round(wLw * sfrac))
    sR = int(round(wRw * sfrac))

    # edge strips
    edgeL = wL[:, wLw - sL : ].copy()
    edgeR = wR[:, : sR].copy()

    # equalise widths for matching
    tgtw = min(edgeL.shape[1], edgeR.shape[1])
    if tgtw < 16:
        tgtw = 64
    if edgeL.shape[1] != tgtw:
        edgeL = cv2.resize(edgeL, (tgtw, edgeL.shape[0]), interpolation=cv2.INTER_AREA)
    if edgeR.shape[1] != tgtw:
        edgeR = cv2.resize(edgeR, (tgtw, edgeR.shape[0]), interpolation=cv2.INTER_AREA)

    dx_pc, dy_pc, resp = phase_corr_shift(edgeL, edgeR)
    dx, dy = dx_pc, dy_pc
    log(f"[shift] phaseCorr dx={dx:.2f}, dy={dy:.2f}, resp={resp:.3f}")

    if args.ecc:
        dx, dy, cc = refine_ecc(edgeL, edgeR, dx, dy, iters=300, eps=1e-5)
        log(f"[shift] ECC-refined dx={dx:.2f}, dy={dy:.2f}, cc={cc:.4f}")

    # canvas and constant warps
    tx, ty, outW, outH = build_canvas_sizes(wLw, hL, wRw, hR, dx, dy)
    log(f"[canvas] {outW}x{outH} (offsets tx={tx}, ty={ty})")

    A_right = np.array([[1, 0, dx + tx],
                        [0, 1, dy + ty]], dtype=np.float32)
    A_left  = np.array([[1, 0, tx],
                        [0, 1, ty]], dtype=np.float32)

    # masks from first frame
    left_on_canvas  = cv2.warpAffine(wL, A_left,  (outW, outH))
    right_on_canvas = cv2.warpAffine(wR, A_right, (outW, outH))
    mL_can = (cv2.cvtColor(left_on_canvas,  cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)*255
    mR_can = (cv2.cvtColor(right_on_canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)*255
    overlap = cv2.bitwise_and(mL_can, mR_can)

    # feather weights
    dL = cv2.distanceTransform(255 - mL_can, cv2.DIST_L2, 3)
    dR = cv2.distanceTransform(255 - mR_can, cv2.DIST_L2, 3)
    denom = dL + dR + 1e-6
    wR_feather = (dL/denom * 255.0).astype(np.uint8)
    wR_feather = cv2.bitwise_and(wR_feather, overlap)

    # writer
    fps = capL.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1: fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (outW, outH))

    if args.preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    # streaming loop
    frame_count, t0 = 0, time.time()
    while True:
        okL, fL = capL.read()
        okR, fR = capR.read()
        if not okL or not okR:
            break

        if args.downscale != 1.0 and args.downscale > 0:
            fL = cv2.resize(fL, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
            fR = cv2.resize(fR, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)

        wL, _ = cylindrical_warp(fL, fov_deg=args.fov)
        wR, _ = cylindrical_warp(fR, fov_deg=args.fov)

        left_on_canvas  = cv2.warpAffine(wL, A_left,  (outW, outH))
        right_on_canvas = cv2.warpAffine(wR, A_right, (outW, outH))

        pano = left_on_canvas.copy()
        right_only = cv2.bitwise_and(mR_can, cv2.bitwise_not(mL_can))
        pano[right_only > 0] = right_on_canvas[right_only > 0]

        alpha = np.zeros_like(mR_can)
        alpha[overlap > 0] = wR_feather[overlap > 0]
        pano = feather_blend(pano, right_on_canvas, alpha, feather=args.feather)

        writer.write(pano)

        if args.preview:
            pv = constrained_preview(pano, args.max_preview_w, args.max_preview_h)
            cv2.imshow("preview", pv)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        frame_count += 1
        if args.limit and frame_count >= args.limit:
            break

    writer.release()
    capL.release()
    capR.release()
    if args.preview:
        cv2.destroyAllWindows()

    dt = time.time() - t0
    if frame_count:
        log(f"[done] wrote {frame_count} frames at ~{frame_count/max(dt,1e-6):.1f} fps")

if __name__ == "__main__":
    main()

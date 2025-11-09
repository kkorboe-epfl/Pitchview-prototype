#!/usr/bin/env python3
import os, sys, math, argparse
from pathlib import Path
import cv2
import numpy as np

# ------------- helpers -------------
def focal_from_fov_px(width_px: int, hfov_deg: float) -> float:
    hfov = math.radians(hfov_deg)
    return width_px / (2.0 * math.tan(hfov / 2.0))

def cylindrical_warp(img, f_px: float):
    h, w = img.shape[:2]
    y_i, x_i = np.indices((h, w), dtype=np.float32)
    x_c = x_i - w * 0.5
    y_c = y_i - h * 0.5
    theta = x_c / f_px
    h_ = y_c / np.sqrt(x_c * x_c + f_px * f_px)
    x_map = f_px * np.tan(theta) + w * 0.5
    y_map = f_px * h_ + h * 0.5
    warped = cv2.remap(img, x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = (warped[...,0] | warped[...,1] | warped[...,2]) > 0
    return warped, (mask.astype(np.uint8) * 255)

def warp_corners(size, H):
    h, w = size
    pts = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], dtype=np.float64).T
    wp = H @ pts
    wp = wp[:2] / (wp[2:] + 1e-9)
    return wp.T

def make_translation(tx, ty):
    T = np.eye(3, dtype=np.float64)
    T[0,2], T[1,2] = tx, ty
    return T

def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {path}")
    return cap

def read_at_time(cap, t_sec):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(t_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, f = cap.read()
    if not ok:
        raise RuntimeError("Failed to read a frame")
    return f

def apply_clahe_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(g)

def crop_border(img, frac):
    if frac <= 0: return img
    h, w = img.shape[:2]
    dx = int(w * frac)
    dy = int(h * frac)
    return img[dy:h-dy, dx:w-dx].copy()

def build_canvas_size(Lw0, Rw0, H_R2L):
    hL, wL = Lw0.shape[:2]
    hR, wR = Rw0.shape[:2]
    cR = warp_corners((hR, wR), H_R2L)
    cL = np.array([[0,0],[wL,0],[wL,hL],[0,hL]], dtype=np.float64)
    allp = np.vstack([cL, cR])
    min_x, min_y = np.floor(allp.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(allp.max(axis=0)).astype(int)
    tx, ty = -min(0, min_x), -min(0, min_y)
    T = make_translation(tx, ty)
    return (max_x + tx, max_y + ty), T

def feather_blend(A, Am, B, Bm, sigma=50):
    A = A.astype(np.float32); B = B.astype(np.float32)
    wa = cv2.GaussianBlur(Am.astype(np.float32)/255.0, (0,0), sigma)
    wb = cv2.GaussianBlur(Bm.astype(np.float32)/255.0, (0,0), sigma)
    s = wa + wb; s[s < 1e-6] = 1.0
    wa /= s; wb /= s
    out = A * wa[...,None] + B * wb[...,None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    mask = ((Am>0) | (Bm>0)).astype(np.uint8) * 255
    return out, mask

# ------------- detectors & matching -------------
def make_detectors():
    dets = []
    if hasattr(cv2, "SIFT_create"):
        dets.append(("SIFT", cv2.SIFT_create()))
    if hasattr(cv2, "AKAZE_create"):
        dets.append(("AKAZE", cv2.AKAZE_create()))
    dets.append(("ORB", cv2.ORB_create(nfeatures=6000)))
    return dets

def match_and_homography(imgL, imgR, name, det, ratio=None):
    if name == "SIFT":
        norm = cv2.NORM_L2; ratio = ratio or 0.75
    elif name == "AKAZE":
        norm = cv2.NORM_HAMMING; ratio = ratio or 0.8
    else:
        norm = cv2.NORM_HAMMING; ratio = ratio or 0.8

    kL, dL = det.detectAndCompute(imgL, None)
    kR, dR = det.detectAndCompute(imgR, None)
    if dL is None or dR is None or len(kL) < 8 or len(kR) < 8:
        return None, 0

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(dR, dL, k=2)  # map Right->Left
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    if len(good) < 8:
        return None, 0

    ptsR = np.float32([kR[m.queryIdx].pt for m in good])
    ptsL = np.float32([kL[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.995)
    inliers = int(inl.sum()) if (inl is not None) else 0
    return H, inliers

def validate_homography(H, img_shape):
    if H is None: return False
    # basic sanity: avoid extreme scales or flips
    Hn = H / (H[2,2] if abs(H[2,2])>1e-9 else 1.0)
    A = Hn[:2,:2]
    s, _ = np.linalg.eig(A @ A.T)
    min_sv, max_sv = np.sqrt(np.sort(s))
    if not (0.2 < min_sv < 5.0 and 0.2 < max_sv < 5.0):
        return False
    # corners must map to finite area
    h, w = img_shape[:2]
    try:
        _ = warp_corners((h, w), Hn)
    except Exception:
        return False
    return True

# ------------- ECC (fallback) -------------
def ecc_homography(grayL, grayR, iters=200, pyr_levels=3):
    warp = np.eye(3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(grayL, grayR, warp, cv2.MOTION_HOMOGRAPHY,
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6),
                                        inputMask=None, gaussFiltSize=5)
        return warp.astype(np.float64), cc
    except Exception:
        return None, -1

# ------------- calibration -------------
def calibrate_transform(capL, capR, fov_deg, border_crop=0.05, samples=(0.5, 1.5, 2.5), use_ecc=True, debug_dir=None):
    # read a few time points and try best
    frames = []
    for t in samples:
        fL = read_at_time(capL, t)
        fR = read_at_time(capR, t)
        frames.append((fL, fR))

    best = dict(score=-1, H=None, fL=0, fR=0, Lw=None, Rw=None, Lm=None, Rm=None)
    for idx, (fL_raw, fR_raw) in enumerate(frames):
        fLpx = focal_from_fov_px(fL_raw.shape[1], fov_deg)
        fRpx = focal_from_fov_px(fR_raw.shape[1], fov_deg)
        Lw, Lm = cylindrical_warp(fL_raw, fLpx)
        Rw, Rm = cylindrical_warp(fR_raw, fRpx)

        # crop borders to remove black and lens edges
        Lc = crop_border(Lw, border_crop)
        Rc = crop_border(Rw, border_crop)

        gL = apply_clahe_gray(Lc)
        gR = apply_clahe_gray(Rc)

        # try detectors
        for name, det in make_detectors():
            H, inl = match_and_homography(gL, gR, name, det)
            if H is not None and validate_homography(H, gR.shape):
                score = inl
                if score > best["score"]:
                    # compensate crop back to full-warp coords
                    hL, wL = Lw.shape[:2]; hR, wR = Rw.shape[:2]
                    dxL = int(wL * border_crop); dyL = int(hL * border_crop)
                    dxR = int(wR * border_crop); dyR = int(hR * border_crop)
                    T_add_L = make_translation(-dxL, -dyL)  # to full
                    T_add_R = make_translation(dxR, dyR)    # from full to crop -> invert in mapping R->L
                    H_full = T_add_L @ H @ T_add_R
                    best.update(score=score, H=H_full, fL=fLpx, fR=fRpx, Lw=Lw, Rw=Rw, Lm=Lm, Rm=Rm)

        # fallback: ECC (single scale, on cropped)
        if use_ecc and best["score"] < 8:
            Hecc, cc = ecc_homography(gL, gR, iters=300, pyr_levels=3)
            if Hecc is not None and validate_homography(Hecc, gR.shape):
                # compensate crop offsets
                hL, wL = Lw.shape[:2]; hR, wR = Rw.shape[:2]
                dxL = int(wL * border_crop); dyL = int(hL * border_crop)
                dxR = int(wR * border_crop); dyR = int(hR * border_crop)
                H_full = make_translation(-dxL, -dyL) @ Hecc @ make_translation(dxR, dyR)
                best.update(score=max(best["score"], 20), H=H_full, fL=fLpx, fR=fRpx, Lw=Lw, Rw=Rw, Lm=Lm, Rm=Rm)

    if best["H"] is None:
        raise RuntimeError("Calibration failed: could not find a reliable transform with features or ECC")

    # optional debug
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(debug_dir)/"left_warp_debug.jpg"), best["Lw"])
        cv2.imwrite(str(Path(debug_dir)/"right_warp_debug.jpg"), best["Rw"])

    return best["H"], best["fL"], best["fR"], best["Lw"], best["Rw"], best["Lm"], best["Rm"]

# ------------- main stitch -------------
def main():
    ap = argparse.ArgumentParser(description="Two-stream panorama (robust calibration, fixed transform)")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", default="stitched.mp4")
    ap.add_argument("--fov", type=float, default=160.0, help="approx horizontal FOV for each lens")
    ap.add_argument("--border_crop", type=float, default=0.06, help="crop fraction to ignore warped borders during matching")
    ap.add_argument("--feather", type=int, default=50)
    ap.add_argument("--calib_seconds", type=str, default="0.5,1.5,2.5", help="comma list of seconds to sample for calibration")
    ap.add_argument("--no_ecc", action="store_true", help="disable ECC fallback")
    ap.add_argument("--debug_dir", default="", help="write debug images here")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    capL = open_video(args.left)
    capR = open_video(args.right)

    samples = [float(s.strip()) for s in args.calib_seconds.split(",") if s.strip()]
    H_R2L, fLpx, fRpx, Lw0, Rw0, Lm0, Rm0 = calibrate_transform(
        capL, capR, fov_deg=args.fov, border_crop=args.border_crop, samples=samples,
        use_ecc=(not args.no_ecc), debug_dir=args.debug_dir or None
    )

    canvas_size, T = build_canvas_size(Lw0, Rw0, H_R2L)
    H_total = T @ H_R2L
    TL = make_translation(T[0,2], T[1,2])

    W, Hh = canvas_size
    if W <= 0 or Hh <= 0 or W > 10000 or Hh > 10000:
        raise RuntimeError(f"Unreasonable canvas {canvas_size}. Check FOV or inputs")

    # prepare writer
    fps = capL.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240: fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, Hh))

    # rewind to start
    capL.set(cv2.CAP_PROP_POS_FRAMES, 0)
    capR.set(cv2.CAP_PROP_POS_FRAMES, 0)

    show = args.show and os.environ.get("DISPLAY", "") is not None
    frames = 0
    while True:
        okL, fL_raw = capL.read()
        okR, fR_raw = capR.read()
        if not (okL and okR): break

        Lw, Lm = cylindrical_warp(fL_raw, fLpx)
        Rw, Rm = cylindrical_warp(fR_raw, fRpx)

        R_on = cv2.warpPerspective(Rw, H_total, (W, Hh))
        Rm_on = cv2.warpPerspective(Rm, H_total, (W, Hh))
        L_on = cv2.warpPerspective(Lw, TL, (W, Hh))
        Lm_on = cv2.warpPerspective(Lm, TL, (W, Hh))

        blended, _ = feather_blend(L_on, Lm_on, R_on, Rm_on, sigma=args.feather)
        writer.write(blended)

        if show:
            view = blended
            if view.shape[1] > 1280:
                s = 1280 / view.shape[1]
                view = cv2.resize(view, (1280, int(view.shape[0]*s)), interpolation=cv2.INTER_AREA)
            cv2.imshow("Panorama preview (q to quit)", view)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        frames += 1
        if args.max_frames and frames >= args.max_frames: break

    capL.release(); capR.release(); writer.release()
    if show: cv2.destroyAllWindows()
    print(f"[ok] Wrote {frames} frames to {args.out}. Canvas {W}x{Hh}, FPS {fps:.2f}")

if __name__ == "__main__":
    main()

import argparse, sys, time
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
import json


def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


# ---------- stitching calibration (between cameras) ----------

def save_calibration(path: str,
                     H: np.ndarray,
                     offset: Tuple[int, int],
                     pano_size: Tuple[int, int],
                     used_affine: bool,
                     cylindrical: bool,
                     downscale: float):
    """Save stitch calibration (H, canvas) to JSON."""
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
    print(f"[info] Saved calibration to {path}")


def load_calibration(path: str):
    """Load stitch calibration from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H = np.array(data["H"], dtype=np.float32)
    offset = tuple(int(v) for v in data["offset"])
    pano_size = tuple(int(v) for v in data["pano_size"])
    used_affine = bool(data.get("used_affine", False))
    return H, offset, pano_size, used_affine, data


# ---------- per-camera lens calibration ----------

def load_lens_calib(path: str):
    """
    Load per-camera lens calibration (intrinsics K and distortion D) from JSON.

    Expected structure:
    {
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "D": [k1, k2, p1, p2, k3],
      "image_size": [w, h]   # optional
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["K"], dtype=np.float32)
    D = np.array(data["D"], dtype=np.float32)
    img_size = tuple(int(v) for v in data.get("image_size", []))  # (w, h) if present
    return K, D, img_size


# ---------- IO helpers ----------

def open_source(src: str, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    if is_int(src):
        cam = cv2.VideoCapture(int(src), cv2.CAP_ANY)
        if width:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cam.set(cv2.CAP_PROP_FPS, 30)
    else:
        cam = cv2.VideoCapture(src)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open source: {src}")
    return cam


def read_synced(capL, capR):
    okL, fL = capL.read()
    okR, fR = capR.read()
    if not okL or not okR:
        return False, None, None
    return True, fL, fR


# ---------- feature detection and transform ----------

def detect_and_match(grayA, grayB, max_feats=4000):
    orb = cv2.ORB_create(nfeatures=max_feats, fastThreshold=7, scaleFactor=1.2, nlevels=8)
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


def homography_is_sane(H: np.ndarray) -> bool:
    """Reject transforms that imply crazy scales/shears."""
    if H is None or not np.isfinite(H).all():
        return False
    a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    lin = np.array([[a, b], [c, d]], dtype=np.float64)
    det = np.linalg.det(lin)
    if det <= 0:
        return False
    s = np.linalg.svd(lin, compute_uv=False)
    s_ratio = (s.max() / s.min()) if s.min() > 1e-12 else np.inf
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
        f = 30 * w  # heuristic
    y_i, x_i = np.indices((h, w), dtype=np.float32)
    x_c = x_i - w / 2
    y_c = y_i - h / 2
    theta = x_c / f
    h_ = y_c / np.sqrt(x_c ** 2 + f ** 2)
    x_map = f * np.tan(theta) + w / 2
    y_map = f * h_ + h / 2
    out = cv2.remap(img, x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = (out[..., 0] + out[..., 1] + out[..., 2] > 0).astype(np.uint8) * 255
    return out, mask


def compute_transform(firstL, firstR, use_cylindrical=False, downscale=0.5):
    """Estimate homography (or affine) between left and right views."""
    def resize(img, s):
        if s == 1.0:
            return img
        return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), cv2.INTER_AREA)

    L = resize(firstL, downscale)
    R = resize(firstR, downscale)

    if use_cylindrical:
        Lc, _ = cylindrical_warp(L)
        Rc, _ = cylindrical_warp(R)
        Lm, Rm = Lc, Rc
    else:
        Lm, Rm = L, R

    h, w = Lm.shape[:2]

    # bias to inner edges where overlap is stronger
    roiL = Lm[:, w // 2 - w // 6: w]
    roiR = Rm[:, 0: w // 2 + w // 6]
    kpa, kpb, good = detect_and_match(cv2.cvtColor(roiL, cv2.COLOR_BGR2GRAY),
                                      cv2.cvtColor(roiR, cv2.COLOR_BGR2GRAY))
    offsetLx = w // 2 - w // 6

    if len(good) < 12:
        kpa, kpb, good = detect_and_match(cv2.cvtColor(Lm, cv2.COLOR_BGR2GRAY),
                                          cv2.cvtColor(Rm, cv2.COLOR_BGR2GRAY))
        offsetLx = 0

    if len(good) < 8:
        raise RuntimeError("Not enough matches to compute a transform")

    ptsL = np.float32([kpa[m.queryIdx].pt for m in good])
    ptsR = np.float32([kpb[m.trainIdx].pt for m in good])
    ptsL[:, 0] += offsetLx

    # Try homography then affine fallback
    H_ds, mask = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 3.0,
                                    maxIters=5000, confidence=0.995)
    use_affine = False
    if not homography_is_sane(H_ds):
        A, inl = cv2.estimateAffine2D(ptsR, ptsL,
                                      ransacReprojThreshold=3.0,
                                      maxIters=5000, confidence=0.995)
        if A is None:
            raise RuntimeError("Failed to estimate a stable transform")
        H_ds = np.eye(3, dtype=np.float32)
        H_ds[:2, :] = A
        use_affine = True

    # Upscale H back to full resolution
    S = np.array([[1 / downscale, 0, 0],
                  [0, 1 / downscale, 0],
                  [0, 0, 1]], dtype=np.float32)
    H = S @ H_ds @ np.linalg.inv(S)

    return H, use_affine


def build_canvas(firstL, firstR, H, max_w=6000, max_h=3000):
    """Compute panorama canvas size and offset."""
    h, w = firstL.shape[:2]
    cornersR = np.float32([[0, 0],
                           [firstR.shape[1], 0],
                           [firstR.shape[1], h],
                           [0, h]]).reshape(-1, 1, 2)
    warpedR = cv2.perspectiveTransform(cornersR, H)
    all_c = np.vstack([warpedR,
                       np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)])
    x_min, y_min = np.floor(all_c.min(axis=0).ravel()).astype(np.int64)
    x_max, y_max = np.ceil(all_c.max(axis=0).ravel()).astype(np.int64)
    pano_w = int(x_max - x_min)
    pano_h = int(y_max - y_min)

    # cap to reasonable maximums
    scale = min(1.0, max_w / max(pano_w, 1), max_h / max(pano_h, 1))
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


# ---------- blending and stitching ----------

def feather_blend(base, overlay, mask_overlay, feather_px=40):
    mo = (mask_overlay > 0).astype(np.uint8)
    mb = ((cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8))
    overlap = (mo & mb).astype(np.uint8)
    out = base.copy()

    if overlap.sum() == 0:
        out[mo.astype(bool)] = overlay[mo.astype(bool)]
        return out

    dist_o = cv2.distanceTransform((1 - mo) * overlap, cv2.DIST_L2, 3)
    dist_b = cv2.distanceTransform((1 - mb) * overlap, cv2.DIST_L2, 3)
    w_o = dist_o / (dist_o + dist_b + 1e-6)
    w_o = np.clip(w_o / max(feather_px, 1), 0, 1)
    w_b = 1.0 - w_o
    w_o3 = np.dstack([w_o] * 3)
    w_b3 = np.dstack([w_b] * 3)

    only_o = (mo & (1 - mb)).astype(bool)
    out[only_o] = overlay[only_o]
    both = overlap.astype(bool)
    out[both] = (overlay[both] * w_o3[both] + base[both] * w_b3[both]).astype(np.uint8)
    return out


def stitch_pair(frameL, frameR, H, offset, pano_size, left_alpha=1.0):
    """Warp right into panorama and blend with left."""
    ox, oy = offset
    pano_w, pano_h = pano_size
    T = np.array([[1, 0, ox],
                  [0, 1, oy],
                  [0, 0, 1]], dtype=np.float32)
    Hs = T @ H

    base = cv2.warpPerspective(frameR, Hs, (pano_w, pano_h))

    hL, wL = frameL.shape[:2]
    x0, y0 = ox, oy
    x1, y1 = ox + wL, oy + hL

    # clamp ROI to canvas
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(pano_w, x1), min(pano_h, y1)
    if x1c > x0c and y1c > y0c:
        lx0 = x0c - x0
        ly0 = y0c - y0
        lx1 = lx0 + (x1c - x0c)
        ly1 = ly0 + (y1c - y0c)

        roi_base = base[y0c:y1c, x0c:x1c]
        roi_left = frameL[ly0:ly1, lx0:lx1]

        if 0.0 <= left_alpha < 1.0:
            cv2.addWeighted(roi_base, 1.0 - left_alpha,
                            roi_left, left_alpha, 0.0, dst=roi_base)
        else:
            roi_base[:] = roi_left

    return base


def writer_from_args(path: Optional[str], size: Tuple[int, int], fps: float):
    if not path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, size)
    if not vw.isOpened():
        raise RuntimeError(f"Could not open writer: {path}")
    return vw


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Panoramic stitch of two streams with optional lens undistortion."
    )
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-pano-w", type=int, default=6000)
    ap.add_argument("--max-pano-h", type=int, default=3000)
    ap.add_argument("--downscale", type=float, default=0.5,
                    help="Scale used only for transform estimation")
    ap.add_argument("--cylindrical", action="store_true",
                    help="Apply cylindrical pre-warp before feature matching")
    ap.add_argument("--left-alpha", type=float, default=1.0,
                    help="Opacity of the left stream in [0..1] (e.g. 0.5)")
    ap.add_argument("--save-calib", type=str, default=None,
                    help="Path to save stitch calibration (H, offset, pano size) as JSON")
    ap.add_argument("--load-calib", type=str, default=None,
                    help="Path to load stitch calibration JSON and skip feature matching")

    # new: per-camera lens calibration
    ap.add_argument("--left-calib", type=str, default=None,
                    help="JSON with K, D for the left camera (lens distortion)")
    ap.add_argument("--right-calib", type=str, default=None,
                    help="JSON with K, D for the right camera (lens distortion)")

    args = ap.parse_args()

    capL = open_source(args.left, args.width, args.height)
    capR = open_source(args.right, args.width, args.height)

    ok, firstL, firstR = read_synced(capL, capR)
    if not ok:
        print("Could not read initial frames from both sources", file=sys.stderr)
        sys.exit(1)

    # normalise heights for transform consistency
    if firstL.shape[0] != firstR.shape[0]:
        th = min(firstL.shape[0], firstR.shape[0])

        def rh(img):
            s = th / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * s), th),
                              interpolation=cv2.INTER_AREA)

        firstL = rh(firstL)
        firstR = rh(firstR)

    # ---------- optional lens undistortion ----------

    map1L = map2L = map1R = map2R = None

    if args.left_calib and args.right_calib:
        K_L, D_L, _ = load_lens_calib(args.left_calib)
        K_R, D_R, _ = load_lens_calib(args.right_calib)

        h_ref, w_ref = firstL.shape[:2]
        size = (w_ref, h_ref)  # (w, h)

        newK_L, _ = cv2.getOptimalNewCameraMatrix(K_L, D_L, size, 0)
        newK_R, _ = cv2.getOptimalNewCameraMatrix(K_R, D_R, size, 0)

        map1L, map2L = cv2.initUndistortRectifyMap(
            K_L, D_L, None, newK_L, size, cv2.CV_16SC2
        )
        map1R, map2R = cv2.initUndistortRectifyMap(
            K_R, D_R, None, newK_R, size, cv2.CV_16SC2
        )

        # undistort the first frames so stitching calibration is done in undistorted space
        firstL = cv2.remap(firstL, map1L, map2L, cv2.INTER_LINEAR)
        firstR = cv2.remap(firstR, map1R, map2R, cv2.INTER_LINEAR)

    # ---------- load or compute inter-camera transform ----------

    if args.load_calib:
        H, offset, pano_size, used_affine, meta = load_calibration(args.load_calib)
        print(f"[info] Loaded stitch calibration from {args.load_calib}")
        cal_cyl = bool(meta.get("cylindrical", False))
        cal_down = float(meta.get("downscale", args.downscale))
        if cal_cyl != bool(args.cylindrical):
            print("[warn] --cylindrical flag does not match calibration file. "
                  "Panos may not align as expected.", file=sys.stderr)
        if abs(cal_down - args.downscale) > 1e-3:
            print("[warn] --downscale value differs from calibration file. "
                  "This is usually harmless once H is fixed.", file=sys.stderr)
    else:
        ds = max(0.2, min(1.0, args.downscale))
        H_raw, used_affine = compute_transform(
            firstL, firstR,
            use_cylindrical=args.cylindrical,
            downscale=ds
        )
        H, offset, pano_size = build_canvas(
            firstL, firstR, H_raw,
            max_w=args.max_pano_w,
            max_h=args.max_pano_h
        )

        if args.save_calib:
            save_calibration(args.save_calib, H, offset, pano_size,
                             used_affine=used_affine,
                             cylindrical=args.cylindrical,
                             downscale=ds)

    print(f"[info] Transform: {'affine' if 'used_affine' in locals() and used_affine else 'homography'}"
          f"  |  Panorama size: {pano_size}  |  Offset: {offset}")

    # prepare writer
    vw = writer_from_args(args.output, pano_size, args.fps) if args.output else None

    # preview window
    if args.preview:
        disp_w = min(1600, pano_size[0])
        disp_h = int(disp_w * pano_size[1] / max(pano_size[0], 1))
        cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Panorama", max(640, disp_w), max(360, disp_h))

    t0 = time.time()
    frames = 0
    h_ref = firstL.shape[0]

    while True:
        ok, fL, fR = read_synced(capL, capR)
        if not ok:
            break

        # resize to match reference height
        if fL.shape[0] != h_ref:
            s = h_ref / fL.shape[0]
            fL = cv2.resize(fL, (int(fL.shape[1] * s), h_ref),
                            interpolation=cv2.INTER_AREA)
        if fR.shape[0] != h_ref:
            s = h_ref / fR.shape[0]
            fR = cv2.resize(fR, (int(fR.shape[1] * s), h_ref),
                            interpolation=cv2.INTER_AREA)

        # apply lens undistortion if maps are available
        if map1L is not None and map1R is not None:
            fL_ud = cv2.remap(fL, map1L, map2L, cv2.INTER_LINEAR)
            fR_ud = cv2.remap(fR, map1R, map2R, cv2.INTER_LINEAR)
        else:
            fL_ud, fR_ud = fL, fR

        pano = stitch_pair(fL_ud, fR_ud, H, offset, pano_size,
                           left_alpha=args.left_alpha)

        frames += 1
        if frames % 10 == 0:
            fps = frames / (time.time() - t0 + 1e-9)
            cv2.putText(pano, f"{fps:.1f} fps",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)

        if vw is not None:
            vw.write(pano)

        if args.preview:
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

#!/usr/bin/env python3
import argparse, math, sys
import numpy as np, cv2

def open_source(src: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(src), cv2.CAP_ANY) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Could not open {src}")
    return cap

def est_fisheye_f(frame_shape, in_fov_deg):
    h, w = frame_shape[:2]
    r_max = min(w, h) * 0.5 * 0.98
    theta_max = math.radians(in_fov_deg * 0.5)
    return r_max / max(theta_max, 1e-6)

def rectilinear_maps(h_out, w_out, cx, cy, f_e, out_fov_deg):
    out_fov = math.radians(out_fov_deg)
    f_rect = (w_out * 0.5) / math.tan(out_fov * 0.5)
    x = (np.arange(w_out, dtype=np.float32) - w_out*0.5) / f_rect
    y = (np.arange(h_out, dtype=np.float32) - h_out*0.5) / f_rect
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)
    n = np.sqrt(xx*xx + yy*yy + zz*zz); dx, dy, dz = xx/n, yy/n, zz/n
    theta = np.arccos(np.clip(dz, -1, 1)).astype(np.float32)
    phi = np.arctan2(dy, dx).astype(np.float32)
    r = f_e * theta
    map_x = (cx + r*np.cos(phi)).astype(np.float32)
    map_y = (cy + r*np.sin(phi)).astype(np.float32)
    return map_x, map_y

def cylindrical_maps(h_out, w_out, cx, cy, f_e, out_hfov_deg, out_vfov_deg):
    lam = (np.linspace(-out_hfov_deg*0.5, out_hfov_deg*0.5, w_out, dtype=np.float32)) * (math.pi/180.0)
    alp = (np.linspace(-out_vfov_deg*0.5, out_vfov_deg*0.5, h_out, dtype=np.float32)) * (math.pi/180.0)
    lam_g, alp_g = np.meshgrid(lam, alp)

    # Cylinder to unit directions
    dx = np.sin(lam_g)
    dz = np.cos(lam_g)
    dy = np.tan(alp_g)  # vertical angle

    n = np.sqrt(dx*dx + dy*dy + dz*dz); dx, dy, dz = dx/n, dy/n, dz/n
    theta = np.arccos(np.clip(dz, -1, 1)).astype(np.float32)
    phi = np.arctan2(dy, dx).astype(np.float32)

    r = f_e * theta
    map_x = (cx + r*np.cos(phi)).astype(np.float32)
    map_y = (cy + r*np.sin(phi)).astype(np.float32)
    return map_x, map_y

def build_maps(shape, mode, in_fov, out_hfov, out_vfov, out_w, out_h, keep_height=True):
    h_in, w_in = shape[:2]
    cx, cy = w_in*0.5, h_in*0.5
    f_e = est_fisheye_f(shape, in_fov)

    # choose output size: keep input height by default so you don’t “crop”
    if out_w is None and out_h is None:
        out_h = h_in if keep_height else int(h_in * (out_vfov / in_fov))
        out_w = int(out_h * (out_hfov / max(out_vfov, 1e-3)))  # rough aspect from FoVs
    elif out_w is None:
        out_w = int(out_h * (w_in / h_in))
    elif out_h is None:
        out_h = int(out_w * (h_in / w_in))

    if mode == "rectilinear":
        mx, my = rectilinear_maps(out_h, out_w, cx, cy, f_e, out_hfov)
    else:
        # clamp vertical FoV to what the lens can actually see
        out_vfov = min(out_vfov, in_fov)  # no point asking for more than the lens covers
        mx, my = cylindrical_maps(out_h, out_w, cx, cy, f_e, out_hfov, out_vfov)

    return mx, my, out_w, out_h, f_e

def main():
    ap = argparse.ArgumentParser("FOV correction preview (equidistant fisheye → rectilinear/cylindrical)")
    ap.add_argument("--src", required=True)
    ap.add_argument("--mode", choices=["rectilinear", "cylindrical"], default="cylindrical")
    ap.add_argument("--in-fov", type=float, default=160.0, help="Fisheye FOV of the lens")
    ap.add_argument("--out-hfov", type=float, default=120.0, help="Horizontal FOV of the output view")
    ap.add_argument("--out-vfov", type=float, default=120.0, help="Vertical FOV (only used for cylindrical)")
    ap.add_argument("--out-w", type=int, default=None)
    ap.add_argument("--out-h", type=int, default=None)
    ap.add_argument("--keep-height", action="store_true", help="Keep input height for output (prevents top/bottom loss)")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    cap = open_source(args.src)
    ok, frame = cap.read()
    if not ok: print("No frame"), sys.exit(1)

    maps = build_maps(frame.shape, args.mode, args.in_fov, args.out_hfov, args.out_vfov, args.out_w, args.out_h, keep_height=args.keep_height)
    map_x, map_y, ow, oh, f_e = maps
    print(f"[info] input {frame.shape[1]}x{frame.shape[0]} | f_e≈{f_e:.1f}px | mode={args.mode} | out {ow}x{oh} | H-FOV={args.out_hfov}°, V-FOV={args.out_vfov}°")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30), (ow, oh))

    cv2.namedWindow("corrected", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("corrected", min(1280, ow), min(720, oh))

    while True:
        ok, frame = cap.read()
        if not ok: break
        corrected = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("corrected", corrected)
        if writer is not None: writer.write(corrected)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

    cap.release()
    writer and writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

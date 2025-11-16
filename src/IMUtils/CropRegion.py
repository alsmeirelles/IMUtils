import cv2
import os
import numpy as np
from math import degrees
from typing import Tuple, Optional

# ---------------------- Helper functions ----------------------------------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    # robust TL,TR,BR,BL ordering
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    pts = pts[np.argsort(ang)]
    top = pts[pts[:,1].argsort()[:2]]
    bot = pts[pts[:,1].argsort()[2:]]
    tl = top[np.argmin(top[:,0])]
    tr = top[np.argmax(top[:,0])]
    bl = bot[np.argmin(bot[:,0])]
    br = bot[np.argmax(bot[:,0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _quad_is_convex(quad: np.ndarray) -> bool:
    q = quad.reshape(-1,2)
    sign = 0
    for i in range(4):
        p0, p1, p2 = q[i], q[(i+1) % 4], q[(i+2) % 4]
        v1, v2 = p1-p0, p2-p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if i == 0:
            sign = np.sign(cross)
        elif np.sign(cross) != sign:
            return False
    return True

def _min_angle_deg(quad: np.ndarray) -> float:
    qs = quad.reshape(-1,2)
    angs = []
    for i in range(4):
        p0, p1, p2 = qs[(i-1)%4], qs[i], qs[(i+1)%4]
        v1, v2 = p0-p1, p2-p1
        c = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-8)
        angs.append(degrees(np.arccos(np.clip(c, -1, 1))))
    return min(angs)

def _side_lengths(quad: np.ndarray):
    tl, tr, br, bl = quad
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    return top, bottom, left, right

def _ensure_dir(p: Optional[str]) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _longest_true_run(b: np.ndarray) -> tuple[int, int]:
    best_len = best_start = cur_len = cur_start = 0
    for i, v in enumerate(b):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_len, best_start

def _estimate_region_hue_hsv(hsv: np.ndarray) -> Optional[int]:
    """
    Estimate dominant hue of the target region (e.g., belt, board) from the image.
    Looks for moderately saturated, bright blue–cyan pixels, returns hue in [0..179].
    """
    H, S, V = cv2.split(hsv)
    mask_sv = (S > 60) & (V > 60)
    mask_blue = (H >= 70) & (H <= 120)
    sel = mask_sv & mask_blue
    if not np.any(sel):
        return None
    hvals = H[sel].astype(np.int16)
    hist = np.bincount(hvals, minlength=180)
    return int(np.argmax(hist))


def _color_mask(
    image_bgr: np.ndarray,
    background_bgr: Optional[Tuple[int, int, int]],
    tol_h: int,
    tol_s: int,
    tol_v: int) -> np.ndarray:
    """
    Robust color mask.

    Strategy:
      1) Use the provided color (or auto-estimate) **only** to center the HUE band.
      2) Find pixels inside that hue band. From those pixels, compute adaptive lower
         bounds for S and V using quantiles, but never lower than sv_floor.
      3) Apply final S/V thresholds with wide upper caps.

    Args:
        image_bgr: BGR frame.
        background_bgr: (B,G,R) color hint for the region. None = auto hue estimate.
        tol_h: Hue tolerance (±) in OpenCV hue space [0..179].
        tol_s/tol_v: Kept for compatibility; we still cap S/V on the upper side.

    Returns:
        uint8 mask {0,255}.
    """
    # Hardcoded robustness knobs (as requested)
    SV_FLOOR_S = 35  # minimal S accepted under glare
    SV_FLOOR_V = 35  # minimal V accepted in shadows
    Q_S = 0.20  # adaptive floor from hue-selected pixels
    Q_V = 0.20

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.uint8)
    S = hsv[:, :, 1].astype(np.uint8)
    V = hsv[:, :, 2].astype(np.uint8)

    # Center hue from BGR hint or auto-estimate
    if background_bgr is not None:
        target = np.uint8([[list(background_bgr)]])
        th, ts, tv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[0, 0]
    else:
        th_est = _estimate_region_hue_hsv(hsv)
        if th_est is None:
            th, ts, tv = 100, 120, 160  # generic cyan/blue
        else:
            th, ts, tv = th_est, 120, 160

    th = int(th);
    tol_h = int(tol_h)

    # Hue mask with wrap-around handling, using NumPy (not cv2.inRange)
    lo = (th - tol_h) % 180
    hi = (th + tol_h) % 180
    if lo <= hi:
        hue_mask = ((H >= lo) & (H <= hi))
    else:
        # wraps around 0
        hue_mask = ((H >= 0) & (H <= hi)) | ((H >= lo) & (H <= 179))

    # Adaptive S/V lower bounds from pixels inside the hue band
    sel = hue_mask
    if sel.any():
        s_min = max(SV_FLOOR_S, int(np.quantile(S[sel], Q_S)))
        v_min = max(SV_FLOOR_V, int(np.quantile(V[sel], Q_V)))
    else:
        s_min, v_min = SV_FLOOR_S, SV_FLOOR_V

    # Upper caps based on provided tol around the hint center (cast to int first!)
    s_max = min(255, int(ts) + int(tol_s))
    v_max = min(255, int(tv) + int(tol_v))

    # Final S/V masks via NumPy; convert to uint8 0/255 at the end
    s_ok = (S.astype(np.int16) >= s_min) & (S.astype(np.int16) <= s_max)
    v_ok = (V.astype(np.int16) >= v_min) & (V.astype(np.int16) <= v_max)

    mask_bool = hue_mask & s_ok & v_ok
    return mask_bool.astype(np.uint8) * 255

# ---------------------- Public functions ----------------------------------------------------

def crop_white_board(
    image: np.ndarray,
    *,
    expand_px: int = 20,             # desired padding
    max_expand_frac: float = 0.12,   # cap each side's expansion
    min_area_frac: float = 0.10,     # board must cover ≥10% of frame
    robustness: int = 2              # how many times to relax thresholds if area too small
):
    """
    Adaptive white-board crop that is robust to darker shots.
    Returns (crop, (x,y,w,h)).
    """
    H, W = image.shape[:2]
    max_expand_px = int(min(H, W) * max_expand_frac)
    expand_px = int(np.clip(expand_px, 0, max_expand_px))

    # --- 1) Local contrast normalization (helps low light / vignetting)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Lc = clahe.apply(L)
    img_eq = cv2.cvtColor(cv2.merge([Lc, A, B]), cv2.COLOR_LAB2BGR)

    # Precompute channels we’ll use repeatedly
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = cv2.split(hsv)
    lab2 = cv2.cvtColor(img_eq, cv2.COLOR_BGR2LAB)
    L2, A2, B2 = cv2.split(lab2)

    # Structuring elements
    k7 = np.ones((7, 7), np.uint8)
    k11 = np.ones((11, 11), np.uint8)

    # --- 2) Try with progressively more permissive thresholds
    best_rect = (0, 0, W, H)
    best_area = 0

    for relax in range(robustness + 1):
        # HSV near-white (low S, high V) with ADAPTIVE cutoffs
        s_max = 60 + 20 * relax                         # allow a little more saturation each step
        v_base_pct = max(40, 75 - 10 * relax)           # percentile to define "bright"
        v_lo = int(np.percentile(Vv, v_base_pct))       # adapt to brightness
        v_lo = max(100 - 20 * relax, v_lo - 10)         # don't go too low; allow some relaxation

        mask_hsv = (Ss < s_max) & (Vv > v_lo)

        # LAB near-neutral (white/grey) + high L*
        ab_tol = 12 + 6 * relax                         # widen neutrality band
        l_pct = max(35, 60 - 10 * relax)                # percentile for L* threshold
        l_lo = int(np.percentile(L2, l_pct))
        l_lo = max(90 - 15 * relax, l_lo - 5)
        mask_lab = (np.abs(A2.astype(np.int16) - 128) < ab_tol) & \
                   (np.abs(B2.astype(np.int16) - 128) < ab_tol) & \
                   (L2 > l_lo)

        mask = (mask_hsv | mask_lab).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k7, iterations=1)
        mask = cv2.dilate(mask, k11, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > best_area:
            x, y, w, h = cv2.boundingRect(c)
            best_rect = (x, y, w, h)
            best_area = area

        # Stop early if it looks like the board (enough area)
        if area >= min_area_frac * (W * H):
            break

    # If nothing reasonable, fall back to full frame
    x, y, w, h = best_rect
    if best_area < min_area_frac * (W * H):
        return image, (0, 0, W, H)

    # --- 3) Apply padding with max-expand clamp and image bounds
    x0 = max(0, x - expand_px)
    y0 = max(0, y - expand_px)
    x1 = min(W, x + w + expand_px)
    y1 = min(H, y + h + expand_px)

    # enforce per-side cap
    x0 = max(x - max_expand_px, x0)
    y0 = max(y - max_expand_px, y0)
    x1 = min(x + w + max_expand_px, x1)
    y1 = min(y + h + max_expand_px, y1)

    crop = image[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1 - x0, y1 - y0)


def correct_perspective(
    image: np.ndarray,                          # BGR image
    *,
    target_aspect_ratio: float | None = None,   # e.g. 3.0/2.0 for a 3x2 grid
    ar_strength: float = 0.65,                  # 0..1, how strongly to pull toward target_ar
    allow_ar_flip: bool = True,                 # automatically adjusts orientation
    min_area_frac: float = 0.10,                # reject if quad <10% of crop
    min_side: float = 20.0,                     # reject tiny quads
    cond_max: float = 1e6,                       # homography condition threshold
    return_matrix: bool = False
):
    """
    Returns (warped, H, (out_w,out_h)). If geometry is unsafe, returns original image and identity H.
    """
    Hc, Wc = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7,7), 0), 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image, np.eye(3), (Wc, Hc) if return_matrix else image

    # Use convex hull + minAreaRect for stability against ragged edges/glare
    largest = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    rect = cv2.minAreaRect(hull)
    src = cv2.boxPoints(rect).astype(np.float32)
    src = _order_quad(src)

    # validations
    area = cv2.contourArea(src)
    if area < min_area_frac * (Wc * Hc):  # not confident enough
        return image, np.eye(3), (Wc, Hc)
    if not _quad_is_convex(src):
        return image, np.eye(3), (Wc, Hc)
    if _min_angle_deg(src) < 15:
        return image, np.eye(3), (Wc, Hc)

    top, bottom, left, right = _side_lengths(src)
    if min(top, bottom, left, right) < min_side:
        return image, np.eye(3), (Wc, Hc)

    # base output size from measured sides (prevents crazy stretch)
    out_w = float(max(top, bottom))
    out_h = float(max(left, right))
    measured_ar = out_w / max(out_h, 1.0)

    # Adjusts target_aspect_ratio to be orientation agnostic
    if target_aspect_ratio and target_aspect_ratio > 0:
        ar_cand = float(target_aspect_ratio)

        # --- orientation-agnostic: choose ar or 1/ar, whichever is closer to measured
        if allow_ar_flip:
            if abs(np.log(measured_ar / ar_cand)) > abs(np.log(measured_ar / (1.0 / ar_cand))):
                ar_cand = 1.0 / ar_cand

        # softly pull measured_ar toward ar_cand
        blend_ar = measured_ar ** (1.0 - ar_strength) * (ar_cand ** ar_strength)

        approx_area = out_w * out_h
        out_w = np.sqrt(approx_area * blend_ar)
        out_h = max(1.0, out_w / blend_ar)

    # sanity clamp (still allow perspective but avoid extreme skinny targets)
    ar = out_w / max(out_h, 1.0)
    if ar < 0.3 or ar > 4.0:
        out_w = min(Wc, max(int(Wc*0.9), int(max(top, bottom))))
        out_h = min(Hc, max(int(Hc*0.9), int(max(left, right))))

    out_w, out_h = int(np.clip(out_w, 16, 8192)), int(np.clip(out_h, 16, 8192))
    dst = np.array([[0,0],
                    [out_w-1, 0],
                    [out_w-1, out_h-1],
                    [0, out_h-1]], dtype=np.float32)

    # Prefer RANSAC when available for extra robustness; fall back to 4-point transform
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        H = cv2.getPerspectiveTransform(src, dst)

    # reject ill-conditioned transforms
    if abs(np.linalg.det(H)) < 1e-10 or np.linalg.cond(H) > cond_max:
        return image, np.eye(3), (Wc, Hc) if return_matrix else image

    warped = cv2.warpPerspective(image, H, (out_w, out_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    # post-warp sanity: if everything went mushy, keep original
    if cv2.Laplacian(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 5.0:
        return image, np.eye(3), (Wc, Hc) if return_matrix else image

    return warped, H, (out_w, out_h) if return_matrix else warped


def crop_color_region(
    image_bgr: np.ndarray,
    *,
    background_bgr: Optional[Tuple[int, int, int]] = None,
    tol_h: int = 16,
    tol_s: int = 80,
    tol_v: int = 80,
    min_area_frac: float = 0.05,
    margin_px: int = 24,
    morph_close_x_frac: float = 1 / 18,
    morph_close_y_frac: float = 1 / 120,
    row_ratio_thresh: Optional[float] = None,
    row_min_run_frac: float = 0.35,
    min_crop_h_frac: float = 0.22,
    win_h_fracs: Tuple[float, float, float] = (0.22, 0.38, 0.06),
    debug_dir: Optional[str] = None,
    debug_prefix: str = "frame",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop the main color region (e.g., conveyor belt, board) while keeping aspect.

    Steps:
      1) HSV color mask (provided or auto hue estimation).
      2) Morphological closing to fill gaps.
      3) Try largest contour; fallback to adaptive row-projection.

    Returns:
        (crop_bgr, (x, y, w, h))
    """
    H, W = image_bgr.shape[:2]
    _ensure_dir(debug_dir)

    mask = _color_mask(image_bgr, background_bgr, tol_h, tol_s, tol_v)

    # horizontal closing to fill occlusions
    kx = max(25, int(W * morph_close_x_frac)); kx += (kx + 1) % 2
    ky = max(5, int(H * morph_close_y_frac)); ky += (ky + 1) % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_mask.png"), mask)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_closed.png"), closed)

    # try contour first
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    use_contour = False
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) >= (min_area_frac * W * H):
            x, y, w, h = cv2.boundingRect(c)
            use_contour = True

    # 3b) fallback: row projection (adaptive + sliding-window)
    if not use_contour:
        row_counts = (closed > 0).sum(axis=1).astype(np.float32)  # [H]
        row_ratio = row_counts / float(W)

        if debug_dir:
            np.savetxt(os.path.join(debug_dir, f"{debug_prefix}_row_ratio.csv"),
                       row_ratio, delimiter=",", fmt="%.6f")

        Hf = float(H)
        # (A) First try the old adaptive-threshold run (quick win on easy frames)
        if row_ratio_thresh is None:
            m, s = float(row_ratio.mean()), float(row_ratio.std())
            thr = max(0.15, min(0.60, m + 0.50 * s))
        else:
            thr = float(row_ratio_thresh)

        inside = (row_ratio >= thr)
        best_len, best_start = _longest_true_run(inside)
        have_reasonable_run = (best_len >= int(row_min_run_frac * H))

        if have_reasonable_run:
            y0 = best_start
            y1 = best_start + best_len
        else:
            # (B) Sliding-window search: maximize total "blue fraction" over a band height
            #    Search window heights in [win_min..win_max] * H with step
            win_min, win_max, win_step = win_h_fracs
            Ls = [int(Hf * f) for f in np.arange(win_min, win_max + 1e-9, win_step)]
            Ls = [L for L in Ls if L >= max(8, int(Hf * min_crop_h_frac)) and L <= H] or [int(Hf * min_crop_h_frac)]

            # prefix sum for O(1) window score
            prefix = np.concatenate([[0.0], np.cumsum(row_ratio, dtype=np.float64)])
            best_score, y0, y1 = -1.0, 0, H
            for L in Ls:
                # slide window of height L
                max_start = H - L
                if max_start < 0:
                    continue
                # coarse stride for speed, then refine around best
                stride = max(1, L // 8)
                for s0 in range(0, max_start + 1, stride):
                    s1 = s0 + L
                    score = prefix[s1] - prefix[s0]
                    if score > best_score:
                        best_score, y0, y1 = score, s0, s1

                # local refine around the current best
                s0r = max(0, y0 - stride)
                s1r = min(H - L, y0 + stride)
                for s0 in range(s0r, s1r + 1):
                    s1 = s0 + L
                    score = prefix[s1] - prefix[s0]
                    if score > best_score:
                        best_score, y0, y1 = score, s0, s1

        # (C) Enforce a minimum crop height (avoid thin strips)
        min_h = max(int(Hf * min_crop_h_frac), 8)
        cur_h = y1 - y0
        if cur_h < min_h:
            extra = (min_h - cur_h) // 2 + 1
            y0 = max(0, y0 - extra)
            y1 = min(H, y1 + extra)

        # (D) Apply margin and full-width crop
        x, y, w, h = 0, max(0, y0), W, max(1, y1 - y0)

    x0 = max(0, x - margin_px); y0 = max(0, y - margin_px)
    x1 = min(W, x + w + margin_px); y1 = min(H, y + h + margin_px)
    crop = image_bgr[y0:y1, x0:x1].copy()

    return crop, (x0, y0, x1 - x0, y1 - y0)
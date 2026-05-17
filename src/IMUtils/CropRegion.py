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
        use_upper_sv_caps = True
    else:
        th_est = _estimate_region_hue_hsv(hsv)
        if th_est is None:
            th = 100
        else:
            th = th_est
        ts, tv = 255, 255
        use_upper_sv_caps = False

    th = int(th)
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
    if use_upper_sv_caps:
        s_max = min(255, int(ts) + int(tol_s))
        v_max = min(255, int(tv) + int(tol_v))
    else:
        s_max = 255
        v_max = 255

    # Final S/V masks via NumPy; convert to uint8 0/255 at the end
    s_ok = (S.astype(np.int16) >= s_min) & (S.astype(np.int16) <= s_max)
    v_ok = (V.astype(np.int16) >= v_min) & (V.astype(np.int16) <= v_max)

    mask_bool = hue_mask & s_ok & v_ok
    return mask_bool.astype(np.uint8) * 255

def _odd_at_least(value: int, minimum: int = 3) -> int:
    """
    Return the smallest odd integer greater than or equal to value.
    OpenCV morphology kernels work better with odd dimensions.
    """
    value = max(int(value), minimum)
    return value if value % 2 else value + 1

def _aspect_score(aspect: float, target_ratio: Optional[float]) -> float:
    """
    Return 0..1 score for how close `aspect` is to `target_ratio`.

    Orientation-agnostic: ratio 1.5 accepts both 1.5 and 1/1.5.
    """
    if target_ratio is None or target_ratio <= 0:
        return 1.0

    target = max(float(target_ratio), 1.0 / float(target_ratio))
    observed = max(float(aspect), 1.0 / max(float(aspect), 1e-6))
    return float(np.exp(-abs(np.log(observed / target))))

def _runs_from_bool(v: np.ndarray, *, min_len: int = 2) -> list[tuple[int, int]]:
    """
    Return inclusive-exclusive runs from a boolean 1D array.
    """
    runs = []
    start = None

    for i, value in enumerate(v):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None

    if start is not None and len(v) - start >= min_len:
        runs.append((start, len(v)))

    return runs

def _bbox_from_largest_contour(mask: np.ndarray) -> tuple[Optional[tuple[int, int, int, int]], float]:
    """
    Return bbox and contour area of the largest external contour.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    return cv2.boundingRect(c), area

def _bbox_from_projection_support(
    mask: np.ndarray,
    *,
    min_col_ratio: float = 0.08,
    min_row_ratio: float = 0.08,
    smooth_frac: float = 0.025,
) -> Optional[tuple[int, int, int, int]]:
    """
    Estimate bbox from row/column projection support instead of connected contours.

    This is useful when glare or printed grid lines split the board into pieces.
    It uses all foreground evidence along each axis, so missing internal columns
    or rows do not necessarily cut the crop.
    """
    H, W = mask.shape[:2]
    fg = mask > 0

    col_ratio = fg.mean(axis=0).astype(np.float32)
    row_ratio = fg.mean(axis=1).astype(np.float32)

    kx = max(3, int(W * smooth_frac))
    ky = max(3, int(H * smooth_frac))
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1

    col_ratio = cv2.blur(col_ratio.reshape(1, -1), (1, kx)).ravel()
    row_ratio = cv2.blur(row_ratio.reshape(-1, 1), (ky, 1)).ravel()

    cols = np.where(col_ratio >= min_col_ratio)[0]
    rows = np.where(row_ratio >= min_row_ratio)[0]

    if cols.size == 0 or rows.size == 0:
        return None

    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1

    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, x1 - x0, y1 - y0


def _bbox_from_grid_lines(
    image_bgr: np.ndarray,
    white_mask: np.ndarray,
    *,
    board_ratio: Optional[float] = None,
    min_lines_x: int = 4,
    min_lines_y: int = 3,
) -> Optional[tuple[int, int, int, int]]:
    """
    Estimate board bbox from dark grid/outer-border lines.

    This is a rescue fallback for cases where glare breaks the white-board mask.
    It uses black line evidence gated by the detected white board area.
    """
    H, W = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use the white mask only as a coarse board-region gate.
    gate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (_odd_at_least(W * 0.025, 9), _odd_at_least(H * 0.025, 9)),
    )
    gate = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, gate_kernel, iterations=2)
    gate = cv2.dilate(gate, gate_kernel, iterations=1) > 0

    # Dark pixels on a locally bright background: grid lines, not granite texture.
    local = cv2.blur(gray, (31, 31))
    dark = ((gray < 95) | (gray < (local - 35))) & gate
    dark = dark.astype(np.uint8) * 255

    # Extract long horizontal and vertical line evidence.
    kh = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (_odd_at_least(W * 0.035, 21), 3),
    )
    kv = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, _odd_at_least(H * 0.035, 21)),
    )

    h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kh, iterations=1)
    v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kv, iterations=1)

    # Slightly reconnect interrupted printed lines.
    h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, kh, iterations=1)
    v_lines = cv2.morphologyEx(v_lines, cv2.MORPH_CLOSE, kv, iterations=1)

    x_score = (v_lines > 0).mean(axis=0).astype(np.float32)
    y_score = (h_lines > 0).mean(axis=1).astype(np.float32)

    x_score = cv2.blur(x_score.reshape(1, -1), (1, _odd_at_least(W * 0.006, 5))).ravel()
    y_score = cv2.blur(y_score.reshape(-1, 1), (_odd_at_least(H * 0.006, 5), 1)).ravel()

    if x_score.max() <= 0 or y_score.max() <= 0:
        return None

    xs = x_score >= max(0.015, 0.25 * float(x_score.max()))
    ys = y_score >= max(0.015, 0.25 * float(y_score.max()))

    x_runs = _runs_from_bool(xs, min_len=max(2, int(W * 0.0015)))
    y_runs = _runs_from_bool(ys, min_len=max(2, int(H * 0.0015)))

    if len(x_runs) < min_lines_x or len(y_runs) < min_lines_y:
        return None

    x_centers = np.array([(a + b) // 2 for a, b in x_runs], dtype=np.int32)
    y_centers = np.array([(a + b) // 2 for a, b in y_runs], dtype=np.int32)

    x0 = int(x_centers.min())
    x1 = int(x_centers.max())
    y0 = int(y_centers.min())
    y1 = int(y_centers.max())

    if x1 <= x0 or y1 <= y0:
        return None

    # Add grid-derived margin so the crop contains the white border around the grid.
    if len(x_centers) >= 2:
        dx = float(np.median(np.diff(np.sort(x_centers))))
    else:
        dx = W * 0.05

    if len(y_centers) >= 2:
        dy = float(np.median(np.diff(np.sort(y_centers))))
    else:
        dy = H * 0.05

    pad_x = int(max(8, 0.20 * dx))
    pad_y = int(max(8, 0.25 * dy))

    x0 = max(0, x0 - pad_x)
    x1 = min(W, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(H, y1 + pad_y)

    rect = (x0, y0, x1 - x0, y1 - y0)

    # Final ratio repair helps when one outer grid line was weak/missed.
    return _repair_rect_to_ratio(
        rect,
        image_shape=image_bgr.shape,
        board_ratio=board_ratio,
        min_expand_frac=0.08,
    )

def _has_board_border_support(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    min_x_side_support: float = 0.08,
    min_y_side_support: float = 0.12,
    band_x_frac: float = 0.04,
    band_y_frac: float = 0.08,
) -> bool:
    """
    Check dark grid/border evidence near candidate sides.
    Stricter on top/bottom to catch missing rows.
    """
    x, y, w, h = rect
    roi = image_bgr[y:y + h, x:x + w]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    local = cv2.blur(gray, (31, 31))
    dark = (gray < 100) | (gray < (local - 35))

    band_x = max(3, int(w * band_x_frac))
    band_y = max(5, int(h * band_y_frac))

    left = float(dark[:, :band_x].mean())
    right = float(dark[:, -band_x:].mean())
    top = float(dark[:band_y, :].mean())
    bottom = float(dark[-band_y:, :].mean())

    return (
        left >= min_x_side_support
        and right >= min_x_side_support
        and top >= min_y_side_support
        and bottom >= min_y_side_support
    )

def _board_candidate_is_valid(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    contour_area: float,
    frame_area: float,
    min_area_frac: float,
    board_ratio: Optional[float],
    min_ratio_score: float = 0.68,
    min_mask_fill: float = 0.28,
) -> bool:
    """
    Decide whether a detected rectangle is likely a complete board.

    This rejects glare-split crops that are too small, too skinny/wide, or have
    too little white-mask support inside the candidate rectangle.
    """
    x, y, w, h = rect

    if w <= 0 or h <= 0:
        return False

    bbox_area = float(w * h)
    area_frac = bbox_area / frame_area

    aspect = w / max(h, 1)
    ratio_score = _aspect_score(aspect, board_ratio)

    # Allow smaller boards if geometry strongly matches the expected board.
    adaptive_min_area = min_area_frac * (1.0 - 0.45 * ratio_score)

    if area_frac < adaptive_min_area:
        return False

    if contour_area < 0.18 * min_area_frac * frame_area:
        return False

    if ratio_score < min_ratio_score:
        return False

    roi = mask[y:y + h, x:x + w] > 0
    if roi.size == 0:
        return False

    fill = float(roi.mean())
    if fill < min_mask_fill:
        return False

    if not _has_board_border_support(
        image_bgr,
        rect,
        min_x_side_support=0.1,
        min_y_side_support=0.15,
        band_x_frac=0.06,
        band_y_frac=0.08,):

        return False

    return True


def _expand_rect(
    rect: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, int],
    expand_px: int,
    max_expand_px: int,
) -> tuple[int, int, int, int]:
    """
    Expand bbox with image-bound clamping.
    """
    H, W = image_shape[:2]
    x, y, w, h = rect

    expand_px = int(np.clip(expand_px, 0, max_expand_px))

    x0 = max(0, x - expand_px)
    y0 = max(0, y - expand_px)
    x1 = min(W, x + w + expand_px)
    y1 = min(H, y + h + expand_px)

    x0 = max(x - max_expand_px, x0)
    y0 = max(y - max_expand_px, y0)
    x1 = min(x + w + max_expand_px, x1)
    y1 = min(y + h + max_expand_px, y1)

    return x0, y0, x1 - x0, y1 - y0

def _repair_rect_to_ratio(
    rect: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, int],
    board_ratio: Optional[float],
    min_expand_frac: float = 0.08,
) -> tuple[int, int, int, int]:
    """
    Expand a partial board bbox toward the expected board aspect ratio.

    Useful when glare removes full rows/columns and the mask-based bbox is
    systematically too narrow or too short.
    """
    if board_ratio is None or board_ratio <= 0:
        return rect

    H, W = image_shape[:2]
    x, y, w, h = rect

    target = max(float(board_ratio), 1.0 / float(board_ratio))
    observed = w / max(h, 1)

    # Make comparison orientation-agnostic.
    landscape_like = observed >= 1.0
    target_ar = target if landscape_like else 1.0 / target

    new_x, new_y, new_w, new_h = x, y, w, h

    if observed < target_ar:
        # Too narrow: likely lost columns.
        desired_w = int(round(h * target_ar))
        delta = max(desired_w - w, int(w * min_expand_frac))
        new_x = x - delta // 2
        new_w = w + delta
    else:
        # Too short: likely lost rows.
        desired_h = int(round(w / target_ar))
        delta = max(desired_h - h, int(h * min_expand_frac))
        new_y = y - delta // 2
        new_h = h + delta

    x0 = max(0, new_x)
    y0 = max(0, new_y)
    x1 = min(W, new_x + new_w)
    y1 = min(H, new_y + new_h)

    return x0, y0, x1 - x0, y1 - y0


# ---------------------- Public functions ----------------------------------------------------

def crop_white_board(
    image: np.ndarray,
    *,
    board_ratio: Optional[float] = None,
    expand_px: int = 10,
    max_expand_frac: float = 0.08,
    min_area_frac: float = 0.10,
    robustness: int = 2,
    image_name: str = "unknown",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop a white rectangular board from a BGR image.

    Strategy:
      1. Try the original tight largest-contour crop first.
      2. Validate it using area, expected board aspect ratio, and mask support.
      3. If it looks split/cut, use a stronger fallback mask that bridges glare
         seams and grid/column splits.
      4. Return the best valid rectangle cropped from the original image.

    Args:
        image: BGR input image.
        board_ratio: Expected board width / height ratio. Orientation-agnostic.
        expand_px: Desired padding around detected board.
        max_expand_frac: Maximum expansion as fraction of min(H, W).
        min_area_frac: Minimum accepted board bbox area as fraction of frame.
        robustness: Number of relaxed threshold passes.

    Returns:
        (crop, (x, y, w, h)).
    """
    H, W = image.shape[:2]
    frame_area = float(W * H)

    max_expand_px = int(min(H, W) * max_expand_frac)
    expand_px = int(np.clip(expand_px, 0, max_expand_px))

    # --- 1) Local contrast normalization, preserving your original behavior.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Lc = clahe.apply(L)

    img_eq = cv2.cvtColor(cv2.merge([Lc, A, B]), cv2.COLOR_LAB2BGR)

    # Color thresholding: bright board becomes foreground.
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold: bright board becomes foreground.
    mask_adapt = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        _odd_at_least(min(H, W) * 0.08, 51),
        -10,
    )

    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    _, Ss, Vv = cv2.split(hsv)

    lab2 = cv2.cvtColor(img_eq, cv2.COLOR_BGR2LAB)
    L2, A2, B2 = cv2.split(lab2)

    # Tight path kernels: close to your rollback version.
    k7 = np.ones((7, 7), np.uint8)
    k11 = np.ones((11, 11), np.uint8)

    # Fallback kernels: stronger bridge only when tight detection fails.
    k_bridge = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (_odd_at_least(W * 0.040, 17), _odd_at_least(H * 0.040, 17)),
    )

    best_fallback_rect: Optional[tuple[int, int, int, int]] = None
    best_fallback_score = 0.0

    for relax in range(robustness + 1):
        # --- 2) Build the same adaptive white mask logic as tight version.
        s_max = 60 + 20 * relax
        v_base_pct = max(40, 75 - 10 * relax)
        v_lo = int(np.percentile(Vv, v_base_pct))
        v_lo = max(100 - 20 * relax, v_lo - 10)

        mask_hsv = (Ss < s_max) & (Vv > v_lo)

        ab_tol = 12 + 6 * relax
        l_pct = max(35, 60 - 10 * relax)
        l_lo = int(np.percentile(L2, l_pct))
        l_lo = max(90 - 15 * relax, l_lo - 5)

        mask_lab = (
            (np.abs(A2.astype(np.int16) - 128) < ab_tol)
            & (np.abs(B2.astype(np.int16) - 128) < ab_tol)
            & (L2 > l_lo)
        )

        mask_white = mask_hsv | mask_lab
        mask_raw = ((mask_white & (mask_adapt > 0)).astype(np.uint8)) * 255

        # --- 3) Tight candidate: original-style morphology and largest contour.
        tight_mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, k7, iterations=2)
        tight_mask = cv2.morphologyEx(tight_mask, cv2.MORPH_OPEN, k7, iterations=1)
        tight_mask = cv2.dilate(tight_mask, k11, iterations=1)

        tight_rect, tight_area = _bbox_from_largest_contour(tight_mask)

        if tight_rect is not None and _board_candidate_is_valid(
            img_eq,
            tight_mask,
            tight_rect,
            contour_area=tight_area,
            frame_area=frame_area,
            min_area_frac=min_area_frac,
            board_ratio=board_ratio,
            min_ratio_score=0.75,
            min_mask_fill=0.35,
        ):
            rect = _expand_rect(
                tight_rect,
                image_shape=image.shape,
                expand_px=expand_px,
                max_expand_px=max_expand_px,
            )
            x, y, w, h = rect
            print(f"Done tight crop for {image_name}")

            return image[y : y + h, x : x + w].copy(), rect

        print(f"Tight crop failed ({image_name}), relax {relax} of {robustness}")

        # --- 4) Robust fallback candidate: bridge only if tight crop was invalid.
        fallback_mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, k7, iterations=2)
        fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, k7, iterations=1)
        fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, k_bridge, iterations=1)

        fallback_rect = _bbox_from_grid_lines(
            img_eq,
            fallback_mask,
            board_ratio=board_ratio,
        )

        if fallback_rect is None:
            print(f"Fallback from grid lines failed ({image_name}), relax {relax} of {robustness}. Trying projection support.")
            fallback_rect = _bbox_from_projection_support(
                fallback_mask,
                min_col_ratio=0.04,
                min_row_ratio=0.04,
                smooth_frac=0.035,
            )

        if fallback_rect is None:
            print(f"Fallback from projection ({image_name}) support failed, relax {relax} of {robustness}.")
            continue

        candidate_rect = _repair_rect_to_ratio(
            fallback_rect,
            image_shape=image.shape,
            board_ratio=board_ratio,
            min_expand_frac=0.10,
        )

        x, y, w, h = candidate_rect
        bbox_area = float(w * h)
        aspect = w / max(h, 1)
        ratio_score = _aspect_score(aspect, board_ratio)

        score = bbox_area * (ratio_score**2)

        if (
            bbox_area >= min_area_frac * frame_area
            and ratio_score >= 0.80
            and score > best_fallback_score
        ):
            best_fallback_score = score
            best_fallback_rect = candidate_rect

    if best_fallback_rect is None:
        return image, (0, 0, W, H)

    best_fallback_rect = _repair_rect_to_ratio(
        best_fallback_rect,
        image_shape=image.shape,
        board_ratio=board_ratio,
        min_expand_frac=0.10,
    )

    rect = _expand_rect(
        best_fallback_rect,
        image_shape=image.shape,
        expand_px=expand_px,
        max_expand_px=max_expand_px,
    )

    x, y, w, h = rect
    return image[y:y + h, x:x + w].copy(), rect


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
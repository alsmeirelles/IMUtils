import cv2
import numpy as np
from dataclasses import dataclass

from .ImageOps import normalize_illumination_gray


@dataclass(frozen=True)
class BoardDarkAreaResult:
    """Raw dark-pixel measurement inside the detected board area."""

    dark_pct: float
    dark_area_px: int
    board_area_px: int
    board_bbox: tuple[int, int, int, int]  # x, y, w, h
    threshold_used: int

    board_mask: np.ndarray | None = None
    dark_mask: np.ndarray | None = None
    debug_image: np.ndarray | None = None


@dataclass(frozen=True)
class EmptyBoardBaseline:
    """Scalar dark-pixel baseline estimated from empty-board images."""

    mean_dark_pct: float
    std_dark_pct: float
    median_dark_pct: float
    min_dark_pct: float
    max_dark_pct: float
    n_images: int
    values: list[float]

    @property
    def recommended_dark_pct(self) -> float:
        """
        Production baseline value.

        Median is used instead of mean because it is robust to outlier
        empty-board images with extra background, glare, or bad crops.
        """
        return self.median_dark_pct


@dataclass(frozen=True)
class CoverageResult:
    """Estimated insect coverage after scalar empty-board baseline subtraction."""

    coverage_pct: float
    raw_dark_pct: float
    baseline_mean_dark_pct: float
    baseline_std_dark_pct: float
    std_penalty: float
    dark_area_px: int
    board_area_px: int
    board_bbox: tuple[int, int, int, int]
    threshold_used: int

    board_mask: np.ndarray | None = None
    dark_mask: np.ndarray | None = None
    debug_image: np.ndarray | None = None


#-----------------------PRIVATE FUNCTIONS-----------------------
def _largest_filled_contour(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = np.zeros_like(mask)
    if not contours:
        h, w = mask.shape[:2]
        return out, (0, 0, w, h)

    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(out, [contour], -1, 255, cv2.FILLED)

    x, y, w, h = cv2.boundingRect(contour)
    return out, (x, y, w, h)


#-----------------------PUBLIC FUNCTIONS-----------------------

def detect_full_board_mask(
        img_bgr: np.ndarray,
        *,
        min_board_area_frac: float = 0.35,
        close_frac: float = 0.035,
        debug: bool = False,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray | None]:
    """
    Detect the full adhesive board region, including white margins and legend strip.

    The result is filled, so grid lines, text, icons, checkboxes, and logos do not
    create holes in the denominator mask.
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a valid BGR image with shape HxWx3.")

    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    lightness = lab[:, :, 0]

    # Adhesive board is bright and weakly saturated. Thresholds are intentionally
    # permissive to include white, gray, and slightly yellowish board areas.
    light_low_sat = ((saturation < 105) & (value > 70) & (lightness > 70)).astype(np.uint8) * 255

    close_size = max(21, int(min_dim * close_frac))
    close_size = close_size if close_size % 2 == 1 else close_size + 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(light_low_sat, cv2.MORPH_CLOSE, close_kernel, iterations=3)

    open_size = max(9, close_size // 3)
    open_size = open_size if open_size % 2 == 1 else open_size + 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    board_mask, bbox = _largest_filled_contour(mask)

    board_area = int(cv2.countNonZero(board_mask))
    image_area = h * w

    # Tight crops can make the whole image the best denominator approximation.
    if board_area < min_board_area_frac * image_area:
        board_mask = np.full((h, w), 255, dtype=np.uint8)
        bbox = (0, 0, w, h)

    debug_image = None
    if debug:
        debug_image = img_bgr.copy()
        x, y, bw, bh = bbox
        cv2.rectangle(debug_image, (x, y), (x + bw, y + bh), (0, 255, 0), 3)

        overlay = debug_image.copy()
        overlay[board_mask > 0] = (0, 255, 0)
        debug_image = cv2.addWeighted(overlay, 0.25, debug_image, 0.75, 0)

    return board_mask, bbox, debug_image


def calculate_board_dark_area(
        img_bgr: np.ndarray,
        *,
        fixed_threshold: int = 145,
        threshold_mode: str = "otsu",
        min_board_area_frac: float = 0.35,
        close_frac: float = 0.035,
        return_debug: bool = False,
) -> BoardDarkAreaResult:
    """
    Compute dark-pixel percentage inside the detected full board region.

    This does not try to distinguish insects from printed board content. It is
    used both for empty-board baseline estimation and production raw dark area.
    """
    board_mask, board_bbox, debug_image = detect_full_board_mask(
        img_bgr,
        min_board_area_frac=min_board_area_frac,
        close_frac=close_frac,
        debug=return_debug,
    )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    normalized = normalize_illumination_gray(img_bgr)

    valid = normalized[board_mask > 0]
    if valid.size == 0:
        return BoardDarkAreaResult(
            dark_pct=0.0,
            dark_area_px=0,
            board_area_px=0,
            board_bbox=board_bbox,
            threshold_used=0,
            board_mask=board_mask if return_debug else None,
            dark_mask=None,
            debug_image=debug_image,
        )

    if threshold_mode == "otsu":
        threshold, _ = cv2.threshold(valid, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        threshold_used = int(np.clip(threshold, 35, fixed_threshold))
    elif threshold_mode == "fixed":
        threshold_used = int(fixed_threshold)
    else:
        raise ValueError(f"Invalid threshold_mode: {threshold_mode!r}. Use 'otsu' or 'fixed'.")

    dark_mask = (((normalized < threshold_used) | (gray < 90)) & (board_mask > 0)).astype(np.uint8) * 255

    # Suppress isolated JPEG/sensor speckles without removing actual printed lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    dark_area_px = int(cv2.countNonZero(dark_mask))
    board_area_px = int(cv2.countNonZero(board_mask))
    dark_pct = 0.0 if board_area_px == 0 else 100.0 * dark_area_px / board_area_px

    if return_debug and debug_image is not None:
        overlay = debug_image.copy()
        overlay[dark_mask > 0] = (0, 0, 255)
        debug_image = cv2.addWeighted(overlay, 0.35, debug_image, 0.65, 0)

    return BoardDarkAreaResult(
        dark_pct=round(float(dark_pct), 4),
        dark_area_px=dark_area_px,
        board_area_px=board_area_px,
        board_bbox=board_bbox,
        threshold_used=threshold_used,
        board_mask=board_mask if return_debug else None,
        dark_mask=dark_mask if return_debug else None,
        debug_image=debug_image,
    )

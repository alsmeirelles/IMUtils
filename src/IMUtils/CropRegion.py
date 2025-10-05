import cv2
import numpy as np
from math import degrees

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

# Old implementations, to be removed
def _crop_white_board(image:np.ndarray | str):
    # Read the image
    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        raise ValueError("Could not load the image")

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to segment the white board from the background
    # Since the board is lighter, we invert the threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours of the white board
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image")

    # Find the largest contour (assuming the board is the largest light area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # More robust correction: Check the edges of the bounding rectangle for the board
    # Use the grayscale image to check pixel intensity near the edges
    padding_factor = 0.05  # 5% of width/height as a dynamic padding
    dynamic_padding_x = int(w * padding_factor)
    dynamic_padding_y = int(h * padding_factor)

    # Ensure the dynamic padding doesn’t exceed image boundaries
    x = max(0, x - dynamic_padding_x)
    y = max(0, y - dynamic_padding_y)
    w = min(image.shape[1] - x, w + 2 * dynamic_padding_x)
    h = min(image.shape[0] - y, h + 2 * dynamic_padding_y)

    # Additional check: Ensure we capture the full board by verifying edge intensity
    # Check if the edges of the cropped region still contain light pixels (board)
    cropped_gray = gray[y:y + h, x:x + w]
    _, edge_thresh = cv2.threshold(cropped_gray, 200, 255, cv2.THRESH_BINARY)
    edge_pixels = cv2.countNonZero(edge_thresh)

    # If there are still light pixels near the edges, expand further (optional safeguard)
    if edge_pixels > 0:
        additional_padding = 10  # Add a small fixed padding if light pixels are detected
        x = max(0, x - additional_padding)
        y = max(0, y - additional_padding)
        w = min(image.shape[1] - x, w + 2 * additional_padding)
        h = min(image.shape[0] - y, h + 2 * additional_padding)

    # Crop the image with the adjusted bounding rectangle
    cropped_image = image[y:y + h, x:x + w]

    return cropped_image, (x, y, w, h)

def crop_white_board(
    image: np.ndarray,
    *,
    expand_px: int = 20,                 # desired uniform padding in pixels
    max_expand_frac: float = 0.12,       # cap expansion to ≤ 12% of min(H,W)
    min_area_frac: float = 0.10          # ignore tiny blobs
):
    """
    Returns cropped image and crop rectangle (x,y,w,h).
    The expansion is clamped to min(H,W)*max_expand_frac.
    """
    H, W = image.shape[:2]
    max_expand_px = int(max(0, min(H, W) * max_expand_frac))
    expand_px = int(np.clip(expand_px, 0, max_expand_px))

    # 1) estimate board mask (bright + smooth)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # white-ish with tolerance (tune if lighting changes)
    lower = np.array([0, 0, 150], dtype=np.uint8)
    upper = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # fallback: return original (no risky crop)
        return image, (0, 0, W, H)

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * (W * H):
        return image, (0, 0, W, H)

    x, y, w, h = cv2.boundingRect(c)

    # 2) apply padding but **clamp by max_expand_px** and **image bounds**
    x0 = max(0, x - expand_px)
    y0 = max(0, y - expand_px)
    x1 = min(W, x + w + expand_px)
    y1 = min(H, y + h + expand_px)

    # ensure we really limited expansion
    if x - x0 > max_expand_px: x0 = x - max_expand_px
    if y - y0 > max_expand_px: y0 = y - max_expand_px
    if x1 - (x + w) > max_expand_px: x1 = x + w + max_expand_px
    if y1 - (y + h) > max_expand_px: y1 = y + h + max_expand_px

    crop = image[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1 - x0, y1 - y0)


def correct_perspective(
    image: np.ndarray,
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

# Old implementation, to be removed
def _correct_perspective(cropped_image: np.ndarray, return_matrix: bool = False):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 100, 200)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the cropped image")
    largest = max(contours, key=cv2.contourArea)
    eps = 0.03 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    if len(approx) < 4:
        raise ValueError("Could not detect enough corners")
    corners = approx[:4].reshape(-1, 2).astype(np.float32)

    # Order corners TL, TR, BR, BL robustly
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:,1]-c[1], corners[:,0]-c[0])
    ordered = corners[np.argsort(angles)]
    top = ordered[ordered[:,1].argsort()[:2]]
    bottom = ordered[ordered[:,1].argsort()[2:]]
    tl = top[top[:,0].argmin()]; tr = top[top[:,0].argmax()]
    bl = bottom[bottom[:,0].argmin()]; br = bottom[bottom[:,0].argmax()]
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    h, w = cropped_image.shape[:2]
    pad = 10
    dst = np.array([[0,0],
                    [w+2*pad-1, 0],
                    [w+2*pad-1, h+2*pad-1],
                    [0, h+2*pad-1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    out_w, out_h = int(w + 2*pad), int(h + 2*pad)
    corrected = cv2.warpPerspective(cropped_image, H, (out_w, out_h))
    return (corrected, H, (out_w, out_h)) if return_matrix else corrected

def process_board(image_path):
    # Step 1: Crop the white board region
    cropped_image, bbox = crop_white_board(image_path, 'cropped_temp.jpg')

    # Step 2: Apply perspective correction to the cropped image
    corrected_image = correct_perspective(cropped_image)

    # Optionally, display the original, cropped, and corrected images
    original_image = cv2.imread(image_path)
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Cropped White Board', cropped_image)
    cv2.imshow('Corrected White Board', corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the corrected image
    cv2.imwrite('corrected_board.jpg', corrected_image)

    # Clean up temporary file
    # if os.path.exists('cropped_temp.jpg'):
    #    os.remove('cropped_temp.jpg')

    return corrected_image


# Example usage with command-line argument or user input
if __name__ == "__main__":
    import argparse

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='This is a white board processor. Can create crops and skew corrections.')
    parser.add_argument('-img', dest='img', type=str, default='.',
                        help='Path to image to process.', required=True)
    parser.add_argument('-vi', action='store_true', default=False, dest='visualize',
                        help='Visualize results.')
    parser.add_argument('-v', action='count', default=0, dest='verbose',
                        help='Amount of verbosity (more \'v\'s means more verbose).')
    config, unparsed = parser.parse_known_args()

    process_board(config.img)

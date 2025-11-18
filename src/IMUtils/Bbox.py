import cv2
import numpy as np
from typing import List, Tuple, Optional

# Locals
from .Types import BBoxYolo, Colors, LetterboxParams

def yolo_to_pixels(box: BBoxYolo, H: int, W: int) -> Tuple[int,int,int,int]:
    cls, cx, cy, w, h = box
    px = cx * W; py = cy * H; pw = w * W; ph = h * H
    x0 = px - pw/2; y0 = py - ph/2
    x1 = px + pw/2; y1 = py + ph/2
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

def pixels_to_yolo(cls: int, x0: float, y0: float, x1: float, y1: float, H: int, W: int) -> BBoxYolo:
    x0 = max(0, min(W-1, x0)); x1 = max(0, min(W-1, x1))
    y0 = max(0, min(H-1, y0)); y1 = max(0, min(H-1, y1))
    if x1 <= x0 or y1 <= y0:
        return (cls, 0.0, 0.0, 0.0, 0.0)
    cx = ((x0 + x1) / 2.0) / W
    cy = ((y0 + y1) / 2.0) / H
    w  = (x1 - x0) / W
    h  = (y1 - y0) / H
    return cls, float(cx), float(cy), float(w), float(h)

def transform_labels_after_perspective_warp(
    labels: List[BBoxYolo],
    Hmat: np.ndarray,
    in_size: Tuple[int,int],   # (H, W) BEFORE warp
    out_size: Tuple[int,int],  # (H, W) AFTER warp
) -> List[BBoxYolo]:
    Hi, Wi = in_size
    Ho, Wo = out_size
    out: List[BBoxYolo] = []
    for box in labels:
        cls = box[0]
        x0, y0, x1, y1 = yolo_to_pixels(box, Hi, Wi)
        # 4 corners
        pts = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
        pts_h = np.concatenate([pts, np.ones((4,1), dtype=np.float32)], axis=1)
        warped = (Hmat @ pts_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        wx0, wy0 = warped.min(axis=0)
        wx1, wy1 = warped.max(axis=0)
        out.append(pixels_to_yolo(cls, wx0, wy0, wx1, wy1, Ho, Wo))
    # filter invalid (zero-area) boxes
    return [b for b in out if b[3] > 0.0 and b[4] > 0.0]

def transform_labels_after_resize_with_pad(
    labels: list[tuple[int, float, float, float, float]],
    H0: int,
    W0: int,
    params  # LetterboxParams(ratio: float, new_size: (new_w, new_h), pad: (left, top, right, bottom))
) -> list[BBoxYolo]:
    """
    Adjust YOLO-normalized boxes after resize_with_pad.

    - Uses the SAME params returned by ImageOps.resize_with_pad.
    - Assumes labels are normalized to the ORIGINAL image (W0,H0).
    - Applies: scale by ratio, then shift by (left, top), then renormalize by final size.
    """
    ratio = float(params.ratio)
    new_w, new_h = params.new_size        # (width, height) after scaling
    left, top, right, bottom = params.pad # (l, t, r, b) – important

    Wt_final = int(new_w + left + right)
    Ht_final = int(new_h + top + bottom)

    out: list[tuple[int, float, float, float, float]] = []
    for cls, cx, cy, w, h in labels:
        # denormalize in ORIGINAL image coords
        cx_px = cx * W0
        cy_px = cy * H0
        w_px  = w  * W0
        h_px  = h  * H0

        # scale then pad shift
        cx_px = cx_px * ratio + left
        cy_px = cy_px * ratio + top
        w_px  = w_px  * ratio
        h_px  = h_px  * ratio

        # renormalize to FINAL canvas (with pad)
        cx2 = cx_px / Wt_final
        cy2 = cy_px / Ht_final
        w2  = w_px  / Wt_final
        h2  = h_px  / Ht_final

        # keep only positive boxes
        if w2 > 0.0 and h2 > 0.0:
            out.append((int(cls), float(cx2), float(cy2), float(w2), float(h2)))

    return out

def adjust_box_90degree(orientation: str, original_bbox: list | tuple) -> tuple[int, float, float, float, float]:
    """
    Adjust YOLO-format bounding box coordinates for a 90° rotation.

    Args:
        orientation (str): Rotation orientation, either "cw" for clockwise or "ccw" for counterclockwise.
        original_bbox (list or tuple): A bounding box in the format
                             [class_index, x_center, y_center, width, height]
                             with coordinates normalized between 0 and 1.

    Returns:
        list: Adjusted bounding box in the format
              (class_index, new_x_center, new_y_center, new_width, new_height)

    Raises:
        ValueError: If the angle is not 90° or the orientation is not 'cw' or 'ccw'.
    """

    class_idx, x_center, y_center, width, height = original_bbox
    class_idx = int(class_idx)
    x_center = float(x_center)
    y_center = float(y_center)
    width = float(width)
    height = float(height)

    if orientation == "ccw":
        # For a 90° counterclockwise rotation:
        new_x_center = y_center
        new_y_center = 1 - x_center
    elif orientation == "cw":
        # For a 90° clockwise rotation:
        new_x_center = 1 - y_center
        new_y_center = x_center
    else:
        raise ValueError("Orientation must be either 'cw' or 'ccw'.")

    # Swap width and height after rotation
    new_width = height
    new_height = width

    return class_idx, new_x_center, new_y_center, new_width, new_height


def adjust_bounding_box_to_crop(original_bbox, original_size, crop_bbox, min_area=25):
    """
    Adjust a normalized bounding box to fit a cropped image.

    Parameters:
    - original_bbox: Tuple of (class_id, x_center, y_center, width, height) normalized [0, 1]
    - original_size: Tuple of (original_height, original_width) in pixels
    - crop_bbox: Tuple of (x, y, w, h) in pixels for the crop region in the original image
    - min_area: Minimum area of bounding box to fit in the cropped image
    Returns:
    - new_bbox: Tuple of (class_id, x_center, y_center, width, height) normalized [0, 1] for the cropped image
    """
    # Unpack inputs
    class_id, x_center, y_center, width, height = original_bbox
    original_height, original_width = original_size
    crop_x, crop_y, crop_w, crop_h = crop_bbox

    # Original bbox comes as str
    class_id = int(class_id)
    x_center = float(x_center)
    y_center = float(y_center)
    width = float(width)
    height = float(height)

    # Convert normalized bounding box to pixel coordinates in the original image
    bbox_x_center = x_center * original_width
    bbox_y_center = y_center * original_height
    bbox_width = width * original_width
    bbox_height = height * original_height

    # Calculate the top-left and bottom-right corners of the original bounding box
    bbox_x_min = bbox_x_center - (bbox_width / 2)
    bbox_y_min = bbox_y_center - (bbox_height / 2)
    bbox_x_max = bbox_x_center + (bbox_width / 2)
    bbox_y_max = bbox_y_center + (bbox_height / 2)

    # Adjust the bounding box coordinates based on the crop region
    # Shift the coordinates by subtracting the crop's top-left corner (x, y)
    adjusted_x_min = bbox_x_min - crop_x
    adjusted_y_min = bbox_y_min - crop_y
    adjusted_x_max = bbox_x_max - crop_x
    adjusted_y_max = bbox_y_max - crop_y

    # Ensure the adjusted coordinates are within the crop region
    adjusted_x_min = max(0, min(adjusted_x_min, crop_w))
    adjusted_y_min = max(0, min(adjusted_y_min, crop_h))
    adjusted_x_max = max(0, min(adjusted_x_max, crop_w))
    adjusted_y_max = max(0, min(adjusted_y_max, crop_h))

    # Calculate the width and height in pixels
    new_width = adjusted_x_max - adjusted_x_min
    new_height = adjusted_y_max - adjusted_y_min

    # Check if the area is smaller than the minimum area
    area = new_width * new_height
    if area < min_area:
        return class_id, 0, 0, 0, 0  # Return zeroed coordinates for discarded boxes

    # Calculate the new center and size in pixel coordinates for the cropped image
    new_x_center = (adjusted_x_min + adjusted_x_max) / 2
    new_y_center = (adjusted_y_min + adjusted_y_max) / 2
    new_width = adjusted_x_max - adjusted_x_min
    new_height = adjusted_y_max - adjusted_y_min

    # Normalize the new bounding box to the cropped image size (crop_w, crop_h)
    new_x_center_normalized = new_x_center / crop_w
    new_y_center_normalized = new_y_center / crop_h
    new_width_normalized = new_width / crop_w
    new_height_normalized = new_height / crop_h

    # Ensure the normalized values are within [0, 1]
    new_x_center_normalized = max(0, min(1, new_x_center_normalized))
    new_y_center_normalized = max(0, min(1, new_y_center_normalized))
    new_width_normalized = max(0, min(1, new_width_normalized))
    new_height_normalized = max(0, min(1, new_height_normalized))

    return (class_id, new_x_center_normalized, new_y_center_normalized,
            new_width_normalized, new_height_normalized)


def bbox_convert(gtr, oshape, verbose=0):
    """
    Returns a tuple of (F1,F2)
    GTR is XYWHN, read from annotation file;
    F1: XYXY equivalent to model prediction;
    F2: XYWH compatible with CV2 and Albumentations
    OSHAPE is the original image´s shape (tupple) (usually returned by .shape attribute)
    """
    #Invert shape to WH
    oshape = oshape[::-1]
    if verbose > 0:
        print(f"Normalized GTR:\n {gtr}")
        print(f"Original shape (inverted): {oshape}")

    convcoord = np.zeros((len(gtr), 4), dtype=np.uint)
    convcoord[:, :] = convcoord[:, :] + oshape * 2  #Duplicates the tupple, it´s not a multiplication x2
    convcoord[:, 2:] = convcoord[:, 2:] / 2
    gtr = np.round((gtr * convcoord))

    gtr[:, :2] = gtr[:, :2] - gtr[:, 2:]
    gtr[:, 2:] *= 2  #Restablish Width x Height values
    f2 = gtr[:, :].copy()
    if verbose > 0:
        print(f"GTR (XYWH):\n {f2}")

    gtr[:, 2:] += gtr[:, :2]
    f1 = gtr[:, :]
    if verbose > 0:
        print(f"GTR (XYXY):\n {f1}")

    return f1, f2


def draw_bbox(img, bbox, class_name, box_color=None, cls: Optional[int] = None, thickness:int = -1, draw_text=True):
    """
    Visualizes a single bounding box on the image.
    Parameters:
        img (np.array): Image to visualize.
        bbox: tuple with values XYWH
        class_name: string, name to display for class
        box_color: RGB color of bounding box
        cls: class id of bounding box
        thickness: number of pixels to draw box
        draw_text: whether to draw text on image
    Returns:
        ndarray: Image visualization.
    """
    if box_color is None:
        # deterministic color per class if provided; else green
        if cls is not None:
            # simple hash -> BGR
            c = (37 * (cls + 1)) % 255
            box_color = (int(100 + c) % 255, int(170 + c * 2) % 255, int(200 + c * 3) % 255)
        else:
            box_color = Colors.BOX_COLOR.value
    else:
        box_color = (box_color[2], box_color[1], box_color[0]) # RGB -> BGR

    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    if thickness <= 0:
        H, W = img.shape[:2]
        thickness = max(1, int(round(min(H, W) * 0.0025)))

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=box_color, thickness=thickness)

    if draw_text:
        font_scale = max(0.4, min(1.6, thickness * 0.6))
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness-1))
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=Colors.TEXT_COLOR.value,
            lineType=cv2.LINE_AA)
    return img


def read_bbox(lpath):
    """
    Read label file and return bounding boxes and corresponding classes
    """
    with open(lpath, "r") as fd:
        labels = fd.readlines()

    gtr = [l.strip().split(' ') for l in labels]
    gtr = np.array(gtr).astype(np.float32)
    classes = gtr[:, :1].T[0].astype(np.int32)

    bboxes = gtr[:, 1:]
    return bboxes, classes


# Example usage:
if __name__ == "__main__":
    # Original box: class 0, centered at (0.25, 0.35) with width 0.2 and height 0.1
    original_box = [0, 0.25, 0.35, 0.2, 0.1]

    # Rotate clockwise by 90 degrees
    rotated_box_cw = adjust_box_90degree("cw", original_box)
    print("Clockwise rotated box:", rotated_box_cw)

    # Rotate counterclockwise by 90 degrees
    rotated_box_ccw = adjust_box_90degree("ccw", original_box)
    print("Counterclockwise rotated box:", rotated_box_ccw)

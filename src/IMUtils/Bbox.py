"""Utilities for converting, transforming, reading, and drawing bounding boxes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# Local imports
from .Types import BBoxYolo, Colors, LetterboxParams


def yolo_to_pixels(box: BBoxYolo, H: int, W: int) -> tuple[int, int, int, int]:
    """Convert a YOLO-normalized bounding box to pixel corner coordinates.

    Args:
        box: Bounding box in ``(class_id, x_center, y_center, width, height)``
            format, with coordinates normalized to ``[0, 1]``.
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        The pixel-space corner coordinates as ``(x_min, y_min, x_max, y_max)``.
        Coordinates are rounded to the nearest integer and are not clamped to
        image bounds.
    """
    cls, cx, cy, w, h = box
    px = cx * W
    py = cy * H
    pw = w * W
    ph = h * H
    x0 = px - pw / 2
    y0 = py - ph / 2
    x1 = px + pw / 2
    y1 = py + ph / 2
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def pixels_to_yolo(
    cls: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    H: int,
    W: int,
) -> BBoxYolo:
    """Convert pixel corner coordinates to a YOLO-normalized bounding box.

    Input coordinates are clamped to the image bounds before conversion. If the
    resulting box is empty or inverted, the returned box keeps ``cls`` and uses
    zero coordinates.

    Args:
        cls: Class identifier to store in the returned box.
        x0: Left coordinate in pixels.
        y0: Top coordinate in pixels.
        x1: Right coordinate in pixels.
        y1: Bottom coordinate in pixels.
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        A YOLO-format tuple ``(class_id, x_center, y_center, width, height)``.
    """
    x0 = max(0, min(W - 1, x0))
    x1 = max(0, min(W - 1, x1))
    y0 = max(0, min(H - 1, y0))
    y1 = max(0, min(H - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return (cls, 0.0, 0.0, 0.0, 0.0)
    cx = ((x0 + x1) / 2.0) / W
    cy = ((y0 + y1) / 2.0) / H
    w = (x1 - x0) / W
    h = (y1 - y0) / H
    return cls, float(cx), float(cy), float(w), float(h)


def transform_labels_after_perspective_warp(
    labels: list[BBoxYolo],
    Hmat: np.ndarray,
    in_size: tuple[int, int],
    out_size: tuple[int, int],
    max_elongate: float = 0.0,
    min_dimension: float = 0.0,
    verbose: bool = False,
) -> list[BBoxYolo]:
    """Transform YOLO labels after applying a perspective warp.

    The function maps each box center and four axis extrema through the
    homography matrix, estimates the transformed width and height from those
    warped extrema, clamps the result to the output image, and returns labels in
    normalized YOLO format.

    Args:
        labels: YOLO-format labels from the input image.
        Hmat: ``3 x 3`` homography matrix used for the image warp.
        in_size: Input image size as ``(height, width)``.
        out_size: Output image size as ``(height, width)``.
        max_elongate: Maximum allowed aspect ratio. A value of ``0.0`` disables
            aspect-ratio filtering unless ``min_dimension`` allows the box.
        min_dimension: Minimum width and height, in pixels, accepted regardless
            of the aspect-ratio filter.
        verbose: If true, print messages when labels are dropped.

    Returns:
        Transformed labels in YOLO format, normalized to ``out_size``.
    """
    Hi, Wi = in_size
    Ho, Wo = out_size
    out: list[BBoxYolo] = []

    for box in labels:
        cls, cx, cy, w, h = box

        # Denormalize to pixel coordinates.
        px = cx * Wi
        py = cy * Hi
        pw = w * Wi
        ph = h * Hi

        # Define center and extreme points for width/height mapping.
        pts = np.array(
            [
                [px, py],  # Center
                [px - pw / 2, py],  # Left
                [px + pw / 2, py],  # Right
                [px, py - ph / 2],  # Top
                [px, py + ph / 2],  # Bottom
            ],
            dtype=np.float32,
        )

        # Warp points.
        pts_h = np.concatenate([pts, np.ones((5, 1), dtype=np.float32)], axis=1)
        warped = (Hmat @ pts_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]

        # Extract new center.
        new_cx, new_cy = warped[0]

        # Calculate new width and height from warped extrema. This avoids
        # ballooning when the homography shears the grid.
        new_w = np.linalg.norm(warped[2] - warped[1])
        new_h = np.linalg.norm(warped[4] - warped[3])

        # Convert to bounding box corners for clamping.
        wx0 = new_cx - new_w / 2
        wy0 = new_cy - new_h / 2
        wx1 = new_cx + new_w / 2
        wy1 = new_cy + new_h / 2

        # Clamp to output boundaries.
        wx0 = float(max(0.0, min(Wo - 1.0, wx0)))
        wy0 = float(max(0.0, min(Ho - 1.0, wy0)))
        wx1 = float(max(0.0, min(Wo - 1.0, wx1)))
        wy1 = float(max(0.0, min(Ho - 1.0, wy1)))

        bw = wx1 - wx0
        bh = wy1 - wy0

        if bw <= 0 or bh <= 0:
            if verbose:
                print(
                    f"[transform_labels_perspective_warp] Dropping label (BW or BH <= 0): {box}"
                )
            continue

        aspect = max(bw, bh) / min(bw, bh)
        dimension_ok = bw > min_dimension and bh > min_dimension

        if (aspect <= max_elongate or max_elongate == 0) or dimension_ok:
            out.append(pixels_to_yolo(cls, wx0, wy0, wx1, wy1, Ho, Wo))
        elif verbose:
            print(
                "[transform_labels_perspective_warp] Dropping label "
                f"(aspect {aspect:.2f} > {max_elongate})"
            )

    return out


def transform_labels_after_resize_with_pad(
    labels: list[BBoxYolo],
    H0: int,
    W0: int,
    params: LetterboxParams,
    verbose: bool = False,
) -> list[BBoxYolo]:
    """Adjust YOLO labels after resizing an image with padding.

    Use the same ``LetterboxParams`` returned by
    :func:`IMUtils.ImageOps.resize_with_pad`. Labels are assumed to be
    normalized to the original image dimensions.

    Args:
        labels: Original YOLO-format labels.
        H0: Original image height in pixels.
        W0: Original image width in pixels.
        params: Resize ratio, scaled size, and padding returned by
            ``resize_with_pad``.
        verbose: If true, print messages when labels are dropped.

    Returns:
        YOLO-format labels normalized to the final padded canvas.
    """
    ratio = float(params.ratio)
    new_w, new_h = params.new_size  # (width, height) after scaling
    left, top, right, bottom = params.pad

    Wt_final = int(new_w + left + right)
    Ht_final = int(new_h + top + bottom)

    out: list[BBoxYolo] = []
    for bbox in labels:
        cls, cx, cy, w, h = bbox

        # Denormalize in original image coordinates.
        cx_px = cx * W0
        cy_px = cy * H0
        w_px = w * W0
        h_px = h * H0

        # Scale, then shift by padding.
        cx_px = cx_px * ratio + left
        cy_px = cy_px * ratio + top
        w_px = w_px * ratio
        h_px = h_px * ratio

        # Renormalize to final canvas with padding.
        cx2 = cx_px / Wt_final
        cy2 = cy_px / Ht_final
        w2 = w_px / Wt_final
        h2 = h_px / Ht_final

        # Keep only positive boxes.
        if w2 > 0.0 and h2 > 0.0:
            out.append((int(cls), float(cx2), float(cy2), float(w2), float(h2)))
        elif verbose:
            print(f"[transform_label_after_resize] Dropping box (w or h <= 0): {bbox}")

    return out


def adjust_box_90degree(
    orientation: str,
    original_bbox: Sequence[int | float],
) -> BBoxYolo:
    """Adjust a YOLO-format bounding box for a 90-degree rotation.

    Args:
        orientation: Rotation orientation. Use ``"cw"`` for clockwise or
            ``"ccw"`` for counter-clockwise.
        original_bbox: Bounding box in
            ``(class_id, x_center, y_center, width, height)`` format, with
            coordinates normalized to ``[0, 1]``.

    Returns:
        Adjusted YOLO-format bounding box after rotation.

    Raises:
        ValueError: If ``orientation`` is not ``"cw"`` or ``"ccw"``.
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

def adjust_box_exif(
    exif_tag: int,
    original_bbox: Sequence[int | float],
) -> Tuple[int, float, float, float, float]:
    """
    Adjust a YOLO-format bounding box based on its EXIF orientation tag.
    Maps EXIF tags 1-8 to their normalized coordinate transformations.
    """
    class_idx, cx, cy, w, h = original_bbox
    class_idx = int(class_idx)
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)

    if exif_tag == 1:
        # Normal
        pass
    elif exif_tag == 2:
        # Mirrored horizontally
        cx = 1.0 - cx
    elif exif_tag == 3:
        # Rotated 180 degrees
        cx = 1.0 - cx
        cy = 1.0 - cy
    elif exif_tag == 4:
        # Mirrored vertically
        cy = 1.0 - cy
    elif exif_tag == 5:
        # Mirrored horizontally, then rotated 90 CCW
        cx, cy = cy, cx
        w, h = h, w
    elif exif_tag == 6:
        # Rotated 90 degrees CW
        new_cx = 1.0 - cy
        new_cy = cx
        cx, cy = new_cx, new_cy
        w, h = h, w
    elif exif_tag == 7:
        # Mirrored horizontally, then rotated 90 CW
        new_cx = 1.0 - cy
        new_cy = 1.0 - cx
        cx, cy = new_cx, new_cy
        w, h = h, w
    elif exif_tag == 8:
        # Rotated 90 degrees CCW
        new_cx = cy
        new_cy = 1.0 - cx
        cx, cy = new_cx, new_cy
        w, h = h, w
    else:
        # Unknown tag or missing, assume no change
        pass

    return class_idx, cx, cy, w, h

def adjust_bounding_box_to_crop(
    original_bbox: Sequence[int | float | str],
    original_size: tuple[int, int],
    crop_bbox: tuple[int, int, int, int],
    min_area: int = 25,
) -> BBoxYolo:
    """Adjust a normalized bounding box to fit inside a cropped image.

    Args:
        original_bbox: Original YOLO-format bounding box. Values may be numeric
            or string values read from a label file.
        original_size: Original image size as ``(height, width)``.
        crop_bbox: Crop region in original-image pixels as
            ``(x, y, width, height)``.
        min_area: Minimum retained box area in crop pixels. Smaller boxes return
            zero coordinates.

    Returns:
        A YOLO-format bounding box normalized to the cropped image. If the
        adjusted box area is smaller than ``min_area``, the class is preserved
        and all coordinates are set to zero.
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

    return (
        class_id,
        new_x_center_normalized,
        new_y_center_normalized,
        new_width_normalized,
        new_height_normalized,
    )


def bbox_convert(
    gtr: np.ndarray,
    oshape: tuple[int, int] | tuple[int, int, int],
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized ``XYWH`` boxes to pixel ``XYXY`` and ``XYWH`` arrays.

    Args:
        gtr: Annotation boxes in normalized ``(x_center, y_center, width,
            height)`` format.
        oshape: Original image shape, usually from ``image.shape``. The first
            two dimensions are expected to be ``(height, width)``.
        verbose: Verbosity level. Values greater than zero print intermediate
            arrays.

    Returns:
        A tuple ``(f1, f2)`` where ``f1`` contains pixel ``(x_min, y_min,
        x_max, y_max)`` boxes and ``f2`` contains pixel ``(x_min, y_min, width,
        height)`` boxes.
    """
    # Invert shape to WH.
    oshape = oshape[::-1]
    if verbose > 0:
        print(f"Normalized GTR:\n {gtr}")
        print(f"Original shape (inverted): {oshape}")

    convcoord = np.zeros((len(gtr), 4), dtype=np.uint)
    convcoord[:, :] = convcoord[:, :] + oshape * 2
    convcoord[:, 2:] = convcoord[:, 2:] / 2
    gtr = np.round(gtr * convcoord)

    gtr[:, :2] = gtr[:, :2] - gtr[:, 2:]
    gtr[:, 2:] *= 2
    f2 = gtr[:, :].copy()
    if verbose > 0:
        print(f"GTR (XYWH):\n {f2}")

    gtr[:, 2:] += gtr[:, :2]
    f1 = gtr[:, :]
    if verbose > 0:
        print(f"GTR (XYXY):\n {f1}")

    return f1, f2


def draw_bbox(
    img: np.ndarray,
    bbox: tuple[int | float, int | float, int | float, int | float],
    class_name: str,
    box_color: tuple[int, int, int] | None = None,
    cls: Optional[int] = None,
    thickness: int = -1,
    draw_text: bool = True,
) -> np.ndarray:
    """Draw a single ``XYWH`` bounding box on an image.

    The image is modified in place and returned for convenience. When
    ``box_color`` is omitted, the function uses a deterministic BGR color for
    ``cls`` or the default color from :class:`IMUtils.Types.Colors`.

    Args:
        img: Image array to draw on.
        bbox: Box in pixel ``(x_min, y_min, width, height)`` format.
        class_name: Text label to display above the box.
        box_color: Optional RGB box color. It is converted to OpenCV BGR order.
        cls: Optional class id used to generate a deterministic color when
            ``box_color`` is omitted.
        thickness: Rectangle line thickness in pixels. Non-positive values pick
            a thickness based on image size.
        draw_text: If true, draw ``class_name`` above the box.

    Returns:
        The same image array with the bounding box drawn.
    """
    if box_color is None:
        # Deterministic color per class if provided; otherwise use default.
        if cls is not None:
            # Simple hash -> BGR.
            c = (37 * (cls + 1)) % 255
            box_color = (
                int(100 + c) % 255,
                int(170 + c * 2) % 255,
                int(200 + c * 3) % 255,
            )
        else:
            box_color = Colors.BOX_COLOR.value
    else:
        box_color = (box_color[2], box_color[1], box_color[0])  # RGB -> BGR

    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = (
        int(x_min),
        int(x_min + w),
        int(y_min),
        int(y_min + h),
    )

    if thickness <= 0:
        H, W = img.shape[:2]
        thickness = max(1, int(round(min(H, W) * 0.0025)))

    cv2.rectangle(
        img,
        (x_min, y_min),
        (x_max, y_max),
        color=box_color,
        thickness=thickness,
    )

    if draw_text:
        font_scale = max(0.4, min(1.6, thickness * 0.6))
        ((text_width, text_height), _) = cv2.getTextSize(
            class_name,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            max(1, thickness - 1),
        )
        cv2.rectangle(
            img,
            (x_min, y_min - int(1.3 * text_height)),
            (x_min + text_width, y_min),
            box_color,
            -1,
        )
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=Colors.TEXT_COLOR.value,
            lineType=cv2.LINE_AA,
        )
    return img


def read_bbox(lpath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file.

    Args:
        lpath: Path to a text file where each row contains
            ``class_id x_center y_center width height``.

    Returns:
        A tuple ``(bboxes, classes)``. ``bboxes`` is an ``N x 4`` float32 array
        containing normalized ``(x_center, y_center, width, height)`` boxes, and
        ``classes`` is an ``N`` int32 array of class ids.
    """
    with open(lpath, "r") as fd:
        labels = fd.readlines()

    gtr = [line.strip().split(" ") for line in labels]
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

"""
Tests for IMUtils.CropRegion.

These tests target the current CropRegion API where crop_white_board performs
board localization and perspective correction in a single global pass.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from IMUtils.CropRegion import (
    _color_mask,
    _estimate_region_hue_hsv,
    _longest_true_run,
    _odd_at_least,
    _order_points_strictly,
    _order_quad,
    _quad_is_convex,
    _side_lengths,
    correct_perspective,
    crop_color_region,
    crop_white_board,
)

from conftest import require_sample


def test_odd_at_least_returns_odd_integer() -> None:
    """
    _odd_at_least should return an odd integer greater than or equal to both
    the requested value and the minimum.
    """
    assert _odd_at_least(2, minimum=3) == 3
    assert _odd_at_least(3, minimum=3) == 3
    assert _odd_at_least(8, minimum=3) == 9
    assert _odd_at_least(1, minimum=9) == 9


def test_longest_true_run_returns_best_length_and_start() -> None:
    """
    _longest_true_run should find the longest contiguous True segment.
    """
    values = np.array([False, True, True, False, True, True, True, False])

    length, start = _longest_true_run(values)

    assert length == 3
    assert start == 4


def test_order_quad_returns_tl_tr_br_bl() -> None:
    """
    _order_quad should normalize unordered points to:
        top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.array(
        [
            [100, 100],
            [20, 100],
            [100, 20],
            [20, 20],
        ],
        dtype=np.float32,
    )

    ordered = _order_quad(pts)

    expected = np.array(
        [
            [20, 20],
            [100, 20],
            [100, 100],
            [20, 100],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(ordered, expected)


def test_order_points_strictly_returns_absolute_spatial_order() -> None:
    """
    _order_points_strictly should return:
        top-left, top-right, bottom-right, bottom-left.

    This helper is used by the new global board warp path.
    """
    pts = np.array(
        [
            [530, 280],
            [90, 300],
            [100, 90],
            [540, 70],
        ],
        dtype=np.float32,
    )

    ordered = _order_points_strictly(pts)
    tl, tr, br, bl = ordered

    assert tl[0] < tr[0]
    assert bl[0] < br[0]
    assert tl[1] < bl[1]
    assert tr[1] < br[1]

    np.testing.assert_allclose(tl, [100, 90])
    np.testing.assert_allclose(tr, [540, 70])
    np.testing.assert_allclose(br, [530, 280])
    np.testing.assert_allclose(bl, [90, 300])


def test_quad_is_convex_accepts_rectangle() -> None:
    """
    _quad_is_convex should accept a simple rectangle.
    """
    quad = np.array(
        [
            [20, 20],
            [120, 20],
            [120, 70],
            [20, 70],
        ],
        dtype=np.float32,
    )

    assert _quad_is_convex(quad) is True


def test_side_lengths_for_rectangle() -> None:
    """
    _side_lengths should return top, bottom, left, and right side lengths.
    """
    quad = np.array(
        [
            [20, 20],
            [120, 20],
            [120, 70],
            [20, 70],
        ],
        dtype=np.float32,
    )

    top, bottom, left, right = _side_lengths(quad)

    assert top == pytest.approx(100)
    assert bottom == pytest.approx(100)
    assert left == pytest.approx(50)
    assert right == pytest.approx(50)


def test_estimate_region_hue_hsv_detects_blue_region(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    _estimate_region_hue_hsv should detect the dominant blue/cyan hue.
    """
    hsv = cv2.cvtColor(synthetic_blue_region_image, cv2.COLOR_BGR2HSV)

    hue = _estimate_region_hue_hsv(hsv)

    assert hue is not None
    assert 70 <= hue <= 120


def test_color_mask_with_background_hint_detects_blue_region(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    _color_mask should return a binary mask that includes the target colored
    region and excludes the dark background.
    """
    mask = _color_mask(
        synthetic_blue_region_image,
        background_bgr=(190, 120, 20),
        tol_h=16,
        tol_s=120,
        tol_v=120,
    )

    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})

    assert mask[120, 210] == 255
    assert mask[20, 20] == 0


def test_color_mask_auto_hue_detects_blue_region(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    _color_mask should also detect the blue/cyan region when background_bgr is
    omitted and auto-hue estimation is used.
    """
    mask = _color_mask(
        synthetic_blue_region_image,
        background_bgr=None,
        tol_h=16,
        tol_s=80,
        tol_v=80,
    )

    assert mask.dtype == np.uint8
    assert mask[120, 210] == 255
    assert mask[20, 20] == 0


def test_crop_white_board_global_warp_returns_expected_contract(
    synthetic_global_board_image: np.ndarray,
) -> None:
    """
    crop_white_board should return:
        warped crop,
        source-space bbox,
        global homography matrix,
        output shape as (out_h, out_w).

    This tests the new single-pass board crop + perspective correction path.
    """
    crop, rect, matrix, out_shape = crop_white_board(
        synthetic_global_board_image,
        image_name="synthetic_global_board",
    )

    x, y, w, h = rect
    out_h, out_w = out_shape

    assert isinstance(crop, np.ndarray)
    assert crop.ndim == 3

    assert matrix.shape == (3, 3)
    assert not np.allclose(matrix, np.eye(3))

    assert crop.shape[:2] == (out_h, out_w)
    assert out_h > 100
    assert out_w > 100

    assert x >= 0
    assert y >= 0
    assert w > 0
    assert h > 0
    assert x + w <= synthetic_global_board_image.shape[1]
    assert y + h <= synthetic_global_board_image.shape[0]

    # Default board ratio is 450 / 220 ~= 2.045.
    assert (out_w / out_h) == pytest.approx(450 / 220, rel=0.20)


def test_crop_white_board_respects_custom_board_ratio(
    synthetic_global_board_image: np.ndarray,
) -> None:
    """
    crop_white_board should use a caller-provided positive board_ratio when
    computing the warped output dimensions.
    """
    custom_ratio = 2.0

    crop, rect, matrix, out_shape = crop_white_board(
        synthetic_global_board_image,
        board_ratio=custom_ratio,
        image_name="synthetic_custom_ratio",
    )

    out_h, out_w = out_shape

    assert crop.shape[:2] == (out_h, out_w)
    assert matrix.shape == (3, 3)
    assert not np.allclose(matrix, np.eye(3))
    assert (out_w / out_h) == pytest.approx(custom_ratio, rel=0.20)


def test_crop_white_board_fallback_on_empty_image(
    synthetic_empty_image: np.ndarray,
) -> None:
    """
    When no foreground contour is detected, crop_white_board should return the
    emergency fallback crop, an identity matrix, and the original output shape.
    """
    H, W = synthetic_empty_image.shape[:2]

    crop, rect, matrix, out_shape = crop_white_board(
        synthetic_empty_image,
        image_name="empty",
    )

    margin_x = int(W * 0.02)
    margin_y = int(H * 0.02)

    assert rect == (
        margin_x,
        margin_y,
        W - (2 * margin_x),
        H - (2 * margin_y),
    )

    assert crop.shape[:2] == (H - 2 * margin_y, W - 2 * margin_x)
    np.testing.assert_allclose(matrix, np.eye(3))
    assert out_shape == (H, W)

def test_crop_white_board_uniform_gray_may_be_processed_as_full_frame() -> None:
    """
    A uniform non-zero gray image can be accepted as a full-frame board-like
    region by the current Otsu-based silhouette detector.

    This documents current behavior rather than treating it as fallback.
    """
    image = np.full((240, 420, 3), fill_value=(30, 30, 30), dtype=np.uint8)

    crop, rect, matrix, out_shape = crop_white_board(
        image,
        image_name="uniform_gray",
    )

    assert rect == (0, 0, 420, 240)
    assert crop.ndim == 3
    assert crop.shape[:2] == out_shape
    assert matrix.shape == (3, 3)


def test_crop_white_board_expand_px_affects_source_rect(
    synthetic_global_board_image: np.ndarray,
) -> None:
    """
    expand_px should affect the returned source-space bbox, while the warped
    crop is controlled by the detected board geometry.
    """
    _, rect_small, _, _ = crop_white_board(
        synthetic_global_board_image,
        expand_px=0,
        image_name="expand_zero",
    )
    _, rect_large, _, _ = crop_white_board(
        synthetic_global_board_image,
        expand_px=20,
        image_name="expand_twenty",
    )

    x0, y0, w0, h0 = rect_small
    x1, y1, w1, h1 = rect_large

    assert x1 <= x0
    assert y1 <= y0
    assert w1 >= w0
    assert h1 >= h0


def test_correct_perspective_rejects_blank_image(
    synthetic_blank_image: np.ndarray,
) -> None:
    """
    correct_perspective remains available. It should reject blank images and
    return the original image, identity matrix, and original size.
    """
    H, W = synthetic_blank_image.shape[:2]

    warped, matrix, out_size = correct_perspective(
        synthetic_blank_image,
        return_matrix=True,
    )

    assert warped.shape == synthetic_blank_image.shape
    np.testing.assert_array_equal(warped, synthetic_blank_image)
    np.testing.assert_allclose(matrix, np.eye(3))
    assert out_size == (W, H)


def test_crop_color_region_with_background_hint(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    crop_color_region should crop the horizontal blue/cyan region when a color
    hint is provided.
    """
    crop, bbox = crop_color_region(
        synthetic_blue_region_image,
        background_bgr=(190, 120, 20),
        margin_px=8,
        min_area_frac=0.03,
    )

    x, y, w, h = bbox

    assert crop.ndim == 3
    assert crop.shape[:2] == (h, w)

    # The crop should contain the central point of the synthetic colored band.
    assert x <= 210 <= x + w
    assert y <= 120 <= y + h

    assert w > 250
    assert h > 40


def test_crop_color_region_auto_hue(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    crop_color_region should crop the horizontal blue/cyan region without a
    color hint, using auto-hue estimation.
    """
    crop, bbox = crop_color_region(
        synthetic_blue_region_image,
        background_bgr=None,
        margin_px=8,
        min_area_frac=0.03,
    )

    x, y, w, h = bbox

    assert crop.ndim == 3
    assert crop.shape[:2] == (h, w)

    assert x <= 210 <= x + w
    assert y <= 120 <= y + h

    assert w > 250
    assert h > 40


def test_crop_color_region_writes_debug_files(
    tmp_path: Path,
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    crop_color_region should write mask and closed-mask debug artifacts when
    debug_dir is provided.
    """
    debug_dir = tmp_path / "crop_debug"

    crop, bbox = crop_color_region(
        synthetic_blue_region_image,
        background_bgr=(190, 120, 20),
        margin_px=8,
        min_area_frac=0.03,
        debug_dir=str(debug_dir),
        debug_prefix="unit",
    )

    assert crop.size > 0
    assert bbox[2] > 0
    assert bbox[3] > 0

    assert (debug_dir / "unit_mask.png").is_file()
    assert (debug_dir / "unit_closed.png").is_file()


def test_real_white_board_sample_can_be_processed(samples_dir: Path) -> None:
    """
    Optional regression test for a real board image.

    Expected file:
        tests/samples/white_board.jpg
    """
    path = require_sample(samples_dir, "white_board.jpg")
    image = cv2.imread(str(path))
    assert image is not None, f"Could not read sample image: {path}"

    crop, rect, matrix, out_shape = crop_white_board(
        image,
        image_name=path.name,
    )

    x, y, w, h = rect
    out_h, out_w = out_shape

    assert crop.ndim == 3
    assert crop.shape[:2] == (out_h, out_w)
    assert matrix.shape == (3, 3)

    assert w > 0
    assert h > 0
    assert x >= 0
    assert y >= 0


def test_real_blue_region_sample_can_be_cropped(samples_dir: Path) -> None:
    """
    Optional regression test for a real blue/cyan region image.

    Expected file:
        tests/samples/blue_region.jpg
    """
    path = require_sample(samples_dir, "blue_region.jpg")
    image = cv2.imread(str(path))
    assert image is not None, f"Could not read sample image: {path}"

    crop, bbox = crop_color_region(
        image,
        background_bgr=None,
    )

    x, y, w, h = bbox

    assert crop.ndim == 3
    assert crop.shape[:2] == (h, w)
    assert w > 0
    assert h > 0
    assert x >= 0
    assert y >= 0

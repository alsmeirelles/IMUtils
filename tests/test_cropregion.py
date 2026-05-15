"""
Tests for IMUtils.CropRegion.

The tests use deterministic synthetic BGR images for CI-safe behavior and
optional real images from tests/samples for local regression testing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from IMUtils.CropRegion import (
    _color_mask,
    _estimate_region_hue_hsv,
    _foreground_bbox,
    _longest_true_run,
    _odd_at_least,
    _order_quad,
    _quad_is_convex,
    _side_lengths,
    correct_perspective,
    crop_color_region,
    crop_white_board,
)

from conftest import require_sample


def test_odd_at_least_returns_odd_value() -> None:
    """
    _odd_at_least should return an odd integer greater than or equal to the
    requested value and minimum.
    """
    assert _odd_at_least(2, minimum=3) == 3
    assert _odd_at_least(7, minimum=3) == 7
    assert _odd_at_least(8, minimum=3) == 9
    assert _odd_at_least(1, minimum=9) == 9


def test_longest_true_run() -> None:
    """
    _longest_true_run should return the length and start index of the longest
    consecutive True segment.
    """
    values = np.array([False, True, True, False, True, True, True, False])

    length, start = _longest_true_run(values)

    assert length == 3
    assert start == 4


def test_order_quad_returns_tl_tr_br_bl() -> None:
    """
    _order_quad should normalize unordered quadrilateral points to:
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


def test_quad_is_convex_accepts_rectangle() -> None:
    """
    _quad_is_convex should accept a simple rectangle.
    """
    quad = np.array(
        [
            [20, 20],
            [100, 20],
            [100, 100],
            [20, 100],
        ],
        dtype=np.float32,
    )

    assert _quad_is_convex(quad) is True


def test_side_lengths_for_rectangle() -> None:
    """
    _side_lengths should return top, bottom, left and right lengths.
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


def test_foreground_bbox_returns_union_of_components() -> None:
    """
    _foreground_bbox should return a bounding box around all foreground
    components above the area threshold.
    """
    mask = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(mask, (20, 30), (60, 70), 255, thickness=-1)
    cv2.rectangle(mask, (120, 20), (160, 80), 255, thickness=-1)

    bbox = _foreground_bbox(mask, min_component_area=20)

    assert bbox is not None
    x, y, w, h = bbox

    assert x <= 20
    assert y <= 20
    assert x + w >= 161
    assert y + h >= 81


def test_estimate_region_hue_hsv_detects_blue_region(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    _estimate_region_hue_hsv should find a blue/cyan dominant hue when the
    image contains a large saturated blue-ish region.
    """
    hsv = cv2.cvtColor(synthetic_blue_region_image, cv2.COLOR_BGR2HSV)

    hue = _estimate_region_hue_hsv(hsv)

    assert hue is not None
    assert 70 <= hue <= 120


def test_color_mask_with_background_hint_detects_blue_region(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    _color_mask should create a binary mask for the target blue region when a
    BGR color hint is provided.
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

    # The center of the synthetic blue region should be detected.
    assert mask[120, 210] == 255

    # The dark background should not be detected.
    assert mask[20, 20] == 0


def test_crop_white_board_returns_expected_region(
    synthetic_white_board_image: np.ndarray,
) -> None:
    """
    crop_white_board should crop around the bright board and return a bbox
    close to the synthetic board position, including expansion padding.
    """
    crop, bbox = crop_white_board(
        synthetic_white_board_image,
        expand_px=10,
        min_area_frac=0.10,
        suppress_glare=False,
    )

    x, y, w, h = bbox

    assert crop.ndim == 3
    assert crop.shape[:2] == (h, w)

    # Original board is approximately x=90..410, y=60..240.
    # With expansion, bbox should be close but not required to be exact because
    # morphology and thresholding may include grid-line edges.
    assert 70 <= x <= 105
    assert 40 <= y <= 75
    assert 300 <= w <= 360
    assert 170 <= h <= 220

    # Crop should be substantially smaller than the whole frame.
    assert w < synthetic_white_board_image.shape[1]
    assert h < synthetic_white_board_image.shape[0]


def test_crop_white_board_fallback_returns_original_for_no_board() -> None:
    """
    crop_white_board should safely return the original image and full-frame bbox
    when no confident white board region is found.
    """
    image = np.full((120, 200, 3), fill_value=(25, 25, 25), dtype=np.uint8)

    crop, bbox = crop_white_board(image, min_area_frac=0.10)

    assert bbox == (0, 0, 200, 120)
    assert crop.shape == image.shape
    np.testing.assert_array_equal(crop, image)


def test_crop_white_board_with_glare_keeps_original_pixels(
    synthetic_white_board_with_glare: np.ndarray,
) -> None:
    """
    crop_white_board uses glare suppression only for detection. The returned
    crop should still contain original glare pixels.
    """
    crop, bbox = crop_white_board(
        synthetic_white_board_with_glare,
        expand_px=10,
        min_area_frac=0.10,
        suppress_glare=True,
    )

    assert bbox != (0, 0, synthetic_white_board_with_glare.shape[1], synthetic_white_board_with_glare.shape[0])
    assert crop.max() == 255


def test_crop_color_region_with_background_hint(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    crop_color_region should crop the horizontal blue region when a color hint
    is provided.
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

    # Synthetic blue region is approximately x=40..380, y=80..160.
    assert 25 <= x <= 50
    assert 65 <= y <= 90
    assert 330 <= w <= 380
    assert 80 <= h <= 110


def test_crop_color_region_auto_hue(
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    crop_color_region should also work without a background_bgr hint by
    estimating the dominant blue/cyan hue.
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

    # Synthetic colored region is approximately x=40..380, y=80..160.
    assert w > 250
    assert h > 40

    # The crop should overlap the central part of the colored band.
    assert y <= 120 <= y + h
    assert x <= 210 <= x + w


def test_crop_color_region_writes_debug_files(
    tmp_path: Path,
    synthetic_blue_region_image: np.ndarray,
) -> None:
    """
    crop_color_region should write debug artifacts when debug_dir is provided.
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


def test_correct_perspective_rejects_blank_image() -> None:
    """
    correct_perspective should return the original image, identity homography,
    and original output size when no valid geometry is detected.
    """
    image = np.full((120, 200, 3), fill_value=(25, 25, 25), dtype=np.uint8)

    warped, matrix, out_size = correct_perspective(image, return_matrix=True)

    assert warped.shape == image.shape
    np.testing.assert_array_equal(warped, image)
    np.testing.assert_allclose(matrix, np.eye(3))
    assert out_size == (200, 120)


def test_correct_perspective_detects_synthetic_quad(
    synthetic_perspective_board_image: np.ndarray,
) -> None:
    """
    correct_perspective should detect the synthetic quadrilateral and return a
    valid warped image with a non-identity homography.
    """
    warped, matrix, out_size = correct_perspective(
        synthetic_perspective_board_image,
        target_aspect_ratio=3 / 2,
        return_matrix=True,
    )

    assert warped.ndim == 3
    assert warped.shape[0] > 16
    assert warped.shape[1] > 16
    assert matrix.shape == (3, 3)
    assert out_size == (warped.shape[1], warped.shape[0])
    assert not np.allclose(matrix, np.eye(3))


def test_real_white_board_sample_can_be_cropped(samples_dir: Path) -> None:
    """
    Optional regression test using a real local white-board image.

    Expected file:
        tests/samples/white_board.jpg
    """
    path = require_sample(samples_dir, "white_board.jpg")
    image = cv2.imread(str(path))
    assert image is not None, f"Could not read sample image: {path}"

    crop, bbox = crop_white_board(image)

    x, y, w, h = bbox
    assert crop.ndim == 3
    assert crop.shape[:2] == (h, w)
    assert w > 0
    assert h > 0


def test_real_white_board_glare_sample_can_be_cropped(samples_dir: Path) -> None:
    """
    Optional regression test for a real board image with glare.

    Expected file:
        tests/samples/white_board_glare.jpg
    """
    path = require_sample(samples_dir, "white_board_glare.jpg")
    image = cv2.imread(str(path))
    assert image is not None, f"Could not read sample image: {path}"

    crop, bbox = crop_white_board(image, suppress_glare=True)

    x, y, w, h = bbox
    assert crop.ndim == 3
    assert crop.shape[:2] == (h, w)
    assert w > 0
    assert h > 0


def test_real_blue_region_sample_can_be_cropped(samples_dir: Path) -> None:
    """
    Optional regression test using a real blue/cyan region image.

    Expected file:
        tests/samples/blue_region.jpg
    """
    path = require_sample(samples_dir, "blue_region.jpg")
    image = cv2.imread(str(path))
    assert image is not None, f"Could not read sample image: {path}"

    crop, bbox = crop_color_region(image, background_bgr=None)

    x, y, w, h = bbox
    assert crop.ndim == 3
    assert crop.shape[:2] == (h, w)
    assert w > 0
    assert h > 0

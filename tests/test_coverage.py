"""
Tests for IMUtils.Coverage.

These tests use synthetic BGR images so coverage behavior is deterministic and
does not depend on private image samples.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from IMUtils.Coverage import (
    BoardDarkAreaResult,
    CoverageResult,
    EmptyBoardBaseline,
    _largest_filled_contour,
    calculate_board_dark_area,
    detect_full_board_mask,
)


@pytest.fixture()
def synthetic_coverage_board() -> np.ndarray:
    """
    Create a bright board on a dark background with one dark rectangular mark.
    """
    image = np.full((160, 220, 3), fill_value=(25, 25, 25), dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (180, 130), (245, 245, 245), thickness=-1)
    cv2.rectangle(image, (80, 60), (120, 90), (20, 20, 20), thickness=-1)
    return image


def test_empty_board_baseline_recommended_dark_pct_uses_median() -> None:
    """
    EmptyBoardBaseline.recommended_dark_pct should expose the robust median
    baseline rather than the mean.
    """
    baseline = EmptyBoardBaseline(
        mean_dark_pct=9.0,
        std_dark_pct=2.0,
        median_dark_pct=4.5,
        min_dark_pct=1.0,
        max_dark_pct=20.0,
        n_images=4,
        values=[1.0, 4.0, 5.0, 20.0],
    )

    assert baseline.recommended_dark_pct == 4.5


def test_coverage_result_dataclass_stores_scalar_outputs() -> None:
    """
    CoverageResult is currently a value object for downstream coverage
    estimates and should preserve all scalar fields exactly.
    """
    result = CoverageResult(
        coverage_pct=12.5,
        raw_dark_pct=14.0,
        baseline_mean_dark_pct=1.0,
        baseline_std_dark_pct=0.5,
        std_penalty=1.5,
        dark_area_px=140,
        board_area_px=1000,
        board_bbox=(10, 20, 30, 40),
        threshold_used=120,
    )

    assert result.coverage_pct == 12.5
    assert result.raw_dark_pct == 14.0
    assert result.baseline_mean_dark_pct == 1.0
    assert result.baseline_std_dark_pct == 0.5
    assert result.std_penalty == 1.5
    assert result.dark_area_px == 140
    assert result.board_area_px == 1000
    assert result.board_bbox == (10, 20, 30, 40)
    assert result.threshold_used == 120
    assert result.board_mask is None
    assert result.dark_mask is None
    assert result.debug_image is None


def test_largest_filled_contour_selects_largest_component() -> None:
    """
    _largest_filled_contour should return a filled mask and bbox for the
    largest external contour.
    """
    mask = np.zeros((80, 100), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (20, 20), 255, thickness=-1)
    cv2.rectangle(mask, (40, 25), (85, 65), 255, thickness=-1)

    filled, bbox = _largest_filled_contour(mask)

    assert filled.dtype == np.uint8
    assert bbox == (40, 25, 46, 41)
    assert filled[45, 60] == 255
    assert filled[10, 10] == 0


def test_largest_filled_contour_returns_full_bbox_for_empty_mask() -> None:
    """
    With no contours, _largest_filled_contour should return an empty mask and
    a full-image fallback bbox.
    """
    mask = np.zeros((12, 18), dtype=np.uint8)

    filled, bbox = _largest_filled_contour(mask)

    assert bbox == (0, 0, 18, 12)
    assert cv2.countNonZero(filled) == 0


def test_detect_full_board_mask_detects_bright_low_saturation_board(
    synthetic_coverage_board: np.ndarray,
) -> None:
    """
    detect_full_board_mask should isolate the bright board region while
    excluding the dark background.
    """
    mask, bbox, debug_image = detect_full_board_mask(
        synthetic_coverage_board,
        min_board_area_frac=0.10,
        close_frac=0.03,
        debug=False,
    )

    x, y, w, h = bbox

    assert mask.shape == synthetic_coverage_board.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert debug_image is None

    assert x <= 40
    assert y <= 30
    assert x + w >= 181
    assert y + h >= 131

    assert mask[80, 110] == 255
    assert mask[10, 10] == 0


def test_detect_full_board_mask_returns_debug_overlay(
    synthetic_coverage_board: np.ndarray,
) -> None:
    """
    When debug=True, detect_full_board_mask should return an image-shaped
    overlay instead of None.
    """
    mask, bbox, debug_image = detect_full_board_mask(
        synthetic_coverage_board,
        min_board_area_frac=0.10,
        debug=True,
    )

    assert mask.shape == synthetic_coverage_board.shape[:2]
    assert bbox[2] > 0
    assert bbox[3] > 0
    assert isinstance(debug_image, np.ndarray)
    assert debug_image.shape == synthetic_coverage_board.shape


def test_detect_full_board_mask_falls_back_to_full_image_when_board_is_too_small() -> None:
    """
    If the largest bright region is below min_board_area_frac, the full image
    should be used as the denominator approximation.
    """
    image = np.full((100, 120, 3), fill_value=(20, 20, 20), dtype=np.uint8)
    cv2.rectangle(image, (45, 35), (65, 55), (245, 245, 245), thickness=-1)

    mask, bbox, _ = detect_full_board_mask(image, min_board_area_frac=0.50)

    assert bbox == (0, 0, 120, 100)
    assert cv2.countNonZero(mask) == 120 * 100


def test_detect_full_board_mask_rejects_invalid_images() -> None:
    """
    detect_full_board_mask should reject missing or non-BGR image inputs.
    """
    with pytest.raises(ValueError, match="valid BGR image"):
        detect_full_board_mask(None)

    with pytest.raises(ValueError, match="valid BGR image"):
        detect_full_board_mask(np.zeros((20, 20), dtype=np.uint8))


def test_calculate_board_dark_area_with_fixed_threshold_counts_dark_pixels(
    synthetic_coverage_board: np.ndarray,
) -> None:
    """
    calculate_board_dark_area should count dark pixels inside the detected
    board and return scalar measurement metadata.
    """
    result = calculate_board_dark_area(
        synthetic_coverage_board,
        fixed_threshold=145,
        threshold_mode="fixed",
        min_board_area_frac=0.10,
        return_debug=False,
    )

    assert isinstance(result, BoardDarkAreaResult)
    assert result.threshold_used == 145
    assert result.board_area_px > 0
    assert result.dark_area_px > 0
    assert result.dark_pct == pytest.approx(
        round(100.0 * result.dark_area_px / result.board_area_px, 4),
    )
    assert result.board_bbox[2] > 0
    assert result.board_bbox[3] > 0
    assert result.board_mask is None
    assert result.dark_mask is None
    assert result.debug_image is None


def test_calculate_board_dark_area_returns_debug_masks(
    synthetic_coverage_board: np.ndarray,
) -> None:
    """
    return_debug=True should include board mask, dark mask, and debug image in
    the result object.
    """
    result = calculate_board_dark_area(
        synthetic_coverage_board,
        threshold_mode="fixed",
        min_board_area_frac=0.10,
        return_debug=True,
    )

    assert result.board_mask is not None
    assert result.dark_mask is not None
    assert result.debug_image is not None
    assert result.board_mask.shape == synthetic_coverage_board.shape[:2]
    assert result.dark_mask.shape == synthetic_coverage_board.shape[:2]
    assert result.debug_image.shape == synthetic_coverage_board.shape


def test_calculate_board_dark_area_otsu_threshold_is_clipped_to_fixed_threshold(
    synthetic_coverage_board: np.ndarray,
) -> None:
    """
    In otsu mode, the selected threshold should be clipped so it does not exceed
    fixed_threshold.
    """
    result = calculate_board_dark_area(
        synthetic_coverage_board,
        fixed_threshold=100,
        threshold_mode="otsu",
        min_board_area_frac=0.10,
    )

    assert 35 <= result.threshold_used <= 100
    assert result.dark_area_px > 0


def test_calculate_board_dark_area_rejects_unknown_threshold_mode(
    synthetic_coverage_board: np.ndarray,
) -> None:
    """
    Unknown threshold modes should fail clearly before returning a partial
    measurement.
    """
    with pytest.raises(ValueError, match="Use 'otsu' or 'fixed'"):
        calculate_board_dark_area(
            synthetic_coverage_board,
            threshold_mode="adaptive",
            min_board_area_frac=0.10,
        )

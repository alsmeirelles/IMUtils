"""
Tests for IMUtils.ImageOps.

These tests validate:
    - pure geometry calculations;
    - resize-with-padding behavior;
    - PIL resizing behavior;
    - BGR/RGB behavior for read/write functions;
    - rotation behavior;
    - optional real image samples from an unversioned folder.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from IMUtils.ImageOps import (
    compute_letterbox_params,
    image_resize,
    image_rotate,
    read_image,
    resize_with_pad,
    write_image,
)

from conftest import require_sample


def test_compute_letterbox_params_landscape_to_square() -> None:
    """
    compute_letterbox_params should preserve aspect ratio and center padding.

    Input:
        original: H=100, W=200
        target:   W=300, H=300

    Expected:
        resized size: W=300, H=150
        vertical padding: top=75, bottom=75
    """
    params = compute_letterbox_params(orig_hw=(100, 200), target_wh=(300, 300))

    assert params.ratio == pytest.approx(1.5)
    assert params.new_size == (300, 150)
    assert params.pad == (0, 75, 0, 75)


def test_compute_letterbox_params_portrait_to_square() -> None:
    """
    compute_letterbox_params should pad horizontally for portrait images.

    Input:
        original: H=200, W=100
        target:   W=300, H=300

    Expected:
        resized size: W=150, H=300
        horizontal padding: left=75, right=75
    """
    params = compute_letterbox_params(orig_hw=(200, 100), target_wh=(300, 300))

    assert params.ratio == pytest.approx(1.5)
    assert params.new_size == (150, 300)
    assert params.pad == (75, 0, 75, 0)


def test_resize_with_pad_returns_expected_shape_and_params(
    bgr_test_image: np.ndarray,
) -> None:
    """
    resize_with_pad should return an image exactly matching the target shape.

    The function receives new_shape as (width, height), while NumPy shape is
    represented as (height, width, channels).
    """
    resized, params = resize_with_pad(
        bgr_test_image,
        new_shape=(300, 300),
        padding_color=(255, 255, 255),
        return_params=True,
    )

    assert resized.shape == (300, 300, 3)
    assert params.new_size == (300, 150)
    assert params.pad == (0, 75, 0, 75)


def test_resize_with_pad_uses_padding_color(
    bgr_test_image: np.ndarray,
) -> None:
    """
    resize_with_pad should fill padded regions with the requested BGR color.

    For a 100x200 image resized into 300x300, top and bottom padding should
    exist. The top-left pixel must be the padding color.
    """
    padding_color = (1, 2, 3)

    resized = resize_with_pad(
        bgr_test_image,
        new_shape=(300, 300),
        padding_color=padding_color,
    )

    assert tuple(resized[0, 0].tolist()) == padding_color
    assert tuple(resized[-1, -1].tolist()) == padding_color


def test_image_resize_by_width_preserves_aspect_ratio(
    rgb_pil_image: Image.Image,
) -> None:
    """
    image_resize should resize by width and compute height proportionally.

    Input image:
        W=200, H=100

    Requested:
        width=100

    Expected:
        W=100, H=50
    """
    resized = image_resize(rgb_pil_image, width=100)

    assert isinstance(resized, Image.Image)
    assert resized.size == (100, 50)


def test_image_resize_by_height_preserves_aspect_ratio(
    rgb_pil_image: Image.Image,
) -> None:
    """
    image_resize should resize by height and compute width proportionally.

    Input image:
        W=200, H=100

    Requested:
        height=50

    Expected:
        W=100, H=50
    """
    resized = image_resize(rgb_pil_image, height=50)

    assert isinstance(resized, Image.Image)
    assert resized.size == (100, 50)


def test_image_resize_without_dimensions_returns_original_object(
    rgb_pil_image: Image.Image,
) -> None:
    """
    When width and height are both None, image_resize should return the
    original image object.
    """
    resized = image_resize(rgb_pil_image)

    assert resized is rgb_pil_image


def test_image_rotate_to_vertical_rotates_landscape_array(
    bgr_test_image: np.ndarray,
) -> None:
    """
    image_rotate should rotate a landscape image to vertical orientation.

    The current ImageOps implementation treats NumPy inputs as BGR and
    returns BGR when rnumpy=True.
    """
    rotated, applied, direction = image_rotate(
        bgr_test_image,
        orientation="v",
        rnumpy=True,
        conditional=True,
    )

    assert applied is True
    assert direction == "ccw"
    assert rotated.shape[:2] == (200, 100)


def test_image_rotate_to_horizontal_does_not_rotate_landscape_array_when_conditional(
    bgr_test_image: np.ndarray,
) -> None:
    """
    image_rotate should not rotate an already-horizontal landscape image when
    orientation='h' and conditional=True.
    """
    rotated, applied, direction = image_rotate(
        bgr_test_image,
        orientation="h",
        rnumpy=True,
        conditional=True,
    )

    assert applied is False
    assert direction is None
    assert rotated.shape == bgr_test_image.shape


def test_write_image_and_read_image_roundtrip_bgr_array(
    tmp_path: Path,
    bgr_test_image: np.ndarray,
) -> None:
    """
    write_image should save a BGR NumPy image, and read_image(..., rnumpy=True)
    should return a BGR NumPy image.

    PNG is used to avoid JPEG compression differences.
    """
    path = tmp_path / "roundtrip.png"

    write_image(bgr_test_image, str(path))
    loaded = read_image(path, rnumpy=True)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == bgr_test_image.shape
    np.testing.assert_array_equal(loaded, bgr_test_image)


def test_read_image_as_pil_returns_pil_image(sample_jpeg_path: Path) -> None:
    """
    read_image(..., rnumpy=False) should return a PIL Image instance.
    """
    image = read_image(sample_jpeg_path, rnumpy=False)

    assert isinstance(image, Image.Image)
    assert image.size == (200, 100)


def test_read_image_as_numpy_returns_bgr_array(sample_jpeg_path: Path) -> None:
    """
    read_image(..., rnumpy=True) should return a BGR NumPy array.

    The fixture writes the image with OpenCV in BGR order, so reading it back
    through ImageOps should preserve the expected BGR channel order.
    """
    image = read_image(sample_jpeg_path, rnumpy=True)

    assert isinstance(image, np.ndarray)
    assert image.shape == (100, 200, 3)

    # JPEG compression may slightly alter values, so use approximate checks.
    mean_b, mean_g, mean_r = image.mean(axis=(0, 1))
    assert mean_b == pytest.approx(10, abs=5)
    assert mean_g == pytest.approx(80, abs=5)
    assert mean_r == pytest.approx(200, abs=5)


def test_real_landscape_sample_can_be_read(samples_dir: Path) -> None:
    """
    Optional integration test using an unversioned real image sample.

    Expected file:
        tests/samples/landscape.jpg

    This test is skipped automatically when the sample is not available.
    """
    path = require_sample(samples_dir, "landscape.jpg")

    image = read_image(path, rnumpy=True)

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.shape[1] > image.shape[0]


def test_real_portrait_sample_can_be_read(samples_dir: Path) -> None:
    """
    Optional integration test using an unversioned real portrait image.

    Expected file:
        tests/samples/portrait.jpg

    This test is skipped automatically when the sample is not available.
    """
    path = require_sample(samples_dir, "portrait.jpg")

    image = read_image(path, rnumpy=True)

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.shape[0] > image.shape[1]


def test_real_exif_orientation_sample_is_transposed(samples_dir: Path) -> None:
    """
    Optional integration test for EXIF orientation handling.

    Expected file:
        tests/samples/exif_orientation_6.jpg

    Use a sample known to have EXIF orientation=6. The exact expected shape
    depends on the image, so this test only verifies that ImageOps can read it
    through the EXIF transpose path and return a valid image.
    """
    path = require_sample(samples_dir, "exif_orientation_6.jpg")

    pil_image = read_image(path, rnumpy=False)
    np_image = read_image(path, rnumpy=True)

    assert isinstance(pil_image, Image.Image)
    assert isinstance(np_image, np.ndarray)
    assert np_image.shape[0] == pil_image.height
    assert np_image.shape[1] == pil_image.width
    assert np_image.shape[2] == 3

def test_real_heic_sample_can_be_read(samples_dir: Path) -> None:
    """
    Optional integration test using an unversioned real HEIC image.

    Expected file:
        tests/samples/real.HEIC

    This test is skipped automatically when the sample is not available.
    """
    path = require_sample(samples_dir, "real.HEIC")

    image = read_image(path, rnumpy=True)

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.shape[0] > image.shape[1]
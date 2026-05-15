"""
Shared pytest fixtures for IMUtils tests.

The tests support two kinds of inputs:

1. Synthetic images generated in memory or inside pytest's temporary directory.
2. Real sample files stored in an unversioned folder.

The real sample folder defaults to:

    tests/samples/

It can be overridden with:

    IMUTILS_TEST_SAMPLES=/path/to/samples pytest
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    """
    Return the folder containing local, unversioned real image samples.

    Priority:
        1. IMUTILS_TEST_SAMPLES environment variable
        2. tests/samples inside the repository

    The fixture does not require the folder to exist. Individual tests that
    need real files should skip when the expected sample is missing.
    """
    env_path = os.getenv("IMUTILS_TEST_SAMPLES")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return Path(__file__).parent / "samples"


@pytest.fixture()
def bgr_test_image() -> np.ndarray:
    """
    Create a deterministic BGR image for OpenCV-style ImageOps functions.

    Shape:
        height=100, width=200, channels=3

    The image uses channel-specific values so tests can detect RGB/BGR
    channel swaps.
    """
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 10   # B
    image[:, :, 1] = 80   # G
    image[:, :, 2] = 200  # R
    return image


@pytest.fixture()
def rgb_pil_image() -> Image.Image:
    """
    Create a deterministic RGB PIL image.

    Shape:
        width=200, height=100
    """
    array = np.zeros((100, 200, 3), dtype=np.uint8)
    array[:, :, 0] = 200  # R
    array[:, :, 1] = 80   # G
    array[:, :, 2] = 10   # B
    return Image.fromarray(array, mode="RGB")


@pytest.fixture()
def sample_jpeg_path(tmp_path: Path, bgr_test_image: np.ndarray) -> Path:
    """
    Write a temporary JPEG image using OpenCV.

    The saved file is useful for testing read_image() and image_rotate()
    with real filesystem paths without depending on external sample files.
    """
    path = tmp_path / "sample.jpg"
    ok = cv2.imwrite(str(path), bgr_test_image)
    assert ok, f"Failed to write temporary test image: {path}"
    return path


def require_sample(samples_dir: Path, filename: str) -> Path:
    """
    Return a real sample path or skip the test when it is unavailable.

    This allows tests to be committed while keeping large/private image
    samples outside Git.
    """
    path = samples_dir / filename
    if not path.is_file():
        pytest.skip(f"Missing optional real sample: {path}")
    return path

@pytest.fixture()
def synthetic_white_board_image() -> np.ndarray:
    """
    Create a synthetic BGR image with a bright white board over a dark background.

    The white board includes thin gray grid lines to approximate real board
    artifacts that crop_white_board should bridge through morphology.
    """
    image = np.full((300, 500, 3), fill_value=(35, 35, 35), dtype=np.uint8)

    # Main white board: x=90..410, y=60..240
    cv2.rectangle(image, (90, 60), (410, 240), (245, 245, 245), thickness=-1)

    # Thin gray grid lines.
    for x in range(130, 410, 40):
        cv2.line(image, (x, 60), (x, 240), (180, 180, 180), thickness=1)
    for y in range(90, 240, 30):
        cv2.line(image, (90, y), (410, y), (180, 180, 180), thickness=1)

    return image


@pytest.fixture()
def synthetic_white_board_with_glare() -> np.ndarray:
    """
    Create a synthetic white board image with a narrow specular glare region.

    The returned crop should still be taken from the original image, not from
    the glare-suppressed detection copy.
    """
    image = np.full((300, 500, 3), fill_value=(35, 35, 35), dtype=np.uint8)
    cv2.rectangle(image, (90, 60), (410, 240), (230, 230, 230), thickness=-1)

    # Strong white glare stripe.
    cv2.ellipse(image, (250, 150), (18, 90), 10, 0, 360, (255, 255, 255), -1)

    return image


@pytest.fixture()
def synthetic_blue_region_image() -> np.ndarray:
    """
    Create a synthetic BGR image with a horizontal blue/cyan region.

    The chosen BGR color is intentionally moderately saturated so it is
    compatible with crop_color_region's auto-hue defaults.
    """
    image = np.full((240, 420, 3), fill_value=(30, 30, 30), dtype=np.uint8)

    # BGR blue/cyan-ish region: x=40..380, y=80..160
    region_bgr = (190, 150, 80)
    cv2.rectangle(image, (40, 80), (380, 160), region_bgr, thickness=-1)

    # Add small dark occlusions to ensure morphology is exercised.
    cv2.rectangle(image, (150, 105), (175, 130), (30, 30, 30), thickness=-1)
    cv2.rectangle(image, (260, 95), (285, 125), (30, 30, 30), thickness=-1)

    return image

@pytest.fixture()
def synthetic_perspective_board_image() -> np.ndarray:
    """
    Create a synthetic image containing a skewed bright quadrilateral.

    correct_perspective should detect the quadrilateral and return a warped
    rectangular output with a non-identity homography.
    """
    image = np.full((300, 500, 3), fill_value=(25, 25, 25), dtype=np.uint8)

    pts = np.array(
        [
            [110, 70],
            [395, 45],
            [420, 225],
            [85, 245],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(image, pts, (235, 235, 235))
    cv2.polylines(image, [pts], isClosed=True, color=(10, 10, 10), thickness=3)

    # Add texture so the Laplacian sanity check does not reject the warp.
    for x in range(130, 390, 45):
        cv2.line(image, (x, 75), (x + 10, 230), (170, 170, 170), thickness=1)
    for y in range(95, 225, 35):
        cv2.line(image, (100, y), (410, y - 15), (170, 170, 170), thickness=1)

    return image
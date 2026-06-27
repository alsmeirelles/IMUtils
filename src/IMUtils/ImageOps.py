# -*- coding: utf-8 -*-
import os.path

import PIL
import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import Tuple
from pathlib import Path

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False

# Local imports
from .Bbox import bbox_convert, draw_bbox
from .Types import LetterboxParams


def compute_letterbox_params(orig_hw: tuple[int, int], target_wh: tuple[int, int]) -> LetterboxParams:
    """
    Compute the scale and symmetric padding needed to letterbox an image.

    The function scales the original image so it fits inside ``target_wh`` while
    preserving aspect ratio, then calculates the left/top/right/bottom padding
    needed to reach the requested output size.

    Args:
        orig_hw: Original image size as ``(height, width)``.
        target_wh: Target canvas size as ``(width, height)``.

    Returns:
        LetterboxParams: Frozen dataclass containing:
            ``ratio``: Scale factor applied to the original dimensions.
            ``new_size``: Resized image size as ``(new_width, new_height)``
            before padding.
            ``pad``: Padding as ``(left, top, right, bottom)``.
    """
    H0, W0 = orig_hw
    Wt, Ht = target_wh
    ratio = float(max(Wt, Ht)) / float(max(W0, H0))
    new_w, new_h = int(W0 * ratio), int(H0 * ratio)
    if new_w > Wt or new_h > Ht:
        ratio = float(min(Wt, Ht)) / float(min(W0, H0))
        new_w, new_h = int(W0 * ratio), int(H0 * ratio)
    dw, dh = max(0, Wt - new_w), max(0, Ht - new_h)
    left, right = dw // 2, dw - (dw // 2)
    top, bottom = dh // 2, dh - (dh // 2)
    return LetterboxParams(ratio, (new_w, new_h), (left, top, right, bottom))


def image_resize(image, width=None, height=None, rotate=False, inter=Image.Resampling.BICUBIC):
    """
    Resize an image while preserving its aspect ratio.

    NumPy inputs are converted to PIL images before resizing. If both ``width``
    and ``height`` are provided, only ``width`` is used to compute the output
    size. If neither dimension is provided, the original image object is
    returned unchanged. When ``rotate`` is true and the source is not square,
    the resized image is passed to :func:`image_rotate`; because
    ``image_rotate`` returns rotation metadata, this branch returns that tuple
    instead of only the resized image.

    Args:
        image: PIL ``Image.Image`` or NumPy array representing the image to
            resize. NumPy arrays are interpreted as RGB channel order by
            ``Image.fromarray``.
        width: Desired output width in pixels. If ``None`` or ``0``, ``height``
            is used and width is computed from the source aspect ratio.
        height: Desired output height in pixels. Used only when ``width`` is
            ``None`` or ``0``.
        rotate: If true, rotate the resized image toward a vertical orientation
            when resizing by height, or toward a horizontal orientation when
            resizing by width.
        inter: PIL resampling filter passed to ``Image.resize``.

    Returns:
        Image.Image | tuple: A resized PIL image when no rotation is applied;
        the original image object when no target dimension is provided; or the
        tuple returned by :func:`image_rotate` when ``rotate`` is true.
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (w, h) = image.size

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None or width == 0:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = image.resize(size=dim, resample=inter)

    # apply rotation
    if rotate and h != w:
        orientation = 'v' if width is None else 'h'
        resized = image_rotate(resized, orientation=orientation)

    # return the resized image
    return resized


def resize_with_pad(image: np.ndarray,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int, int, int] = (255, 255, 255),
                    return_params: bool = False) -> np.ndarray | Tuple[np.ndarray, LetterboxParams]:
    """
    Resize an image to fit inside a target canvas and pad the remaining area.

    The image is resized with OpenCV while preserving aspect ratio. Constant
    padding is then added on each side so the final array exactly matches
    ``new_shape``.

    Args:
        image: NumPy image array in OpenCV-compatible layout. ``image.shape`` is
            expected to start with ``(height, width)``.
        new_shape: Target output size as ``(width, height)``.
        padding_color: Border color passed to OpenCV as a BGR tuple.
        return_params: When true, return the computed
            :class:`LetterboxParams` together with the padded image.

    Returns:
        np.ndarray | tuple[np.ndarray, LetterboxParams]: The resized and padded
        image. If ``return_params`` is true, returns ``(image, params)``, where
        ``params`` describes the scale and padding used.
    """

    params = compute_letterbox_params(image.shape[:2], new_shape)

    image = cv2.resize(image, params.new_size)
    left, top, right, bottom = params.pad
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    if return_params:
        return image, params
    return image


def image_rotate(im, orientation=None, rnumpy=False, conditional=False):
    """
    Rotate an image 90 degrees toward a requested orientation.

    ``orientation='v'`` rotates counter-clockwise when rotation is applied, and
    ``orientation='h'`` rotates clockwise. If ``conditional`` is true, rotation
    only occurs when the current image dimensions do not already match the
    requested orientation. If ``orientation`` is ``None``, the original input is
    returned immediately without metadata.

    Args:
        im: Image to rotate. The current implementation accepts a NumPy array in
            BGR order or a string path. NumPy arrays are converted from BGR to
            RGB before PIL rotation. String paths are loaded with
            :func:`read_image`.
        orientation: Target orientation. Use ``'h'`` for horizontal/landscape,
            ``'v'`` for vertical/portrait, or ``None`` to skip rotation.
        rnumpy: When true, return the rotated image as a BGR NumPy array.
            Otherwise, return a PIL image.
        conditional: When true, skip rotation if the image already has the
            requested orientation.

    Returns:
        Any | tuple[Image.Image | np.ndarray, bool, str | None]: If
        ``orientation`` is ``None``, returns ``im`` unchanged. Otherwise returns
        ``(image, apply_rotation, direction)``, where ``image`` is the rotated
        or original image, ``apply_rotation`` indicates whether a rotation was
        performed, and ``direction`` is ``'ccw'``, ``'cw'``, or ``None``.

    Raises:
        TypeError: If ``im`` is not a NumPy array or string path.
    """

    if orientation is None:
        return im

    if isinstance(im, np.ndarray):
        im = Image.fromarray(im[:, :, ::-1])  # BGR -> RGB for PIL
    elif isinstance(im, str):
        im = read_image(im, False)  # Image.open(im)
    else:
        raise TypeError(f"Unsupported image type: {type(im)}")

    apply_rotation = not conditional or ((orientation == "v" and im.width > im.height) or
                                         (orientation == "h" and im.height > im.width))

    if orientation == 'v' and apply_rotation:
        im = im.rotate(90, expand=True)
        direction = 'ccw'
    elif orientation == 'h' and apply_rotation:
        im = im.rotate(-90, expand=True)
        direction = 'cw'
    else:
        direction = None

    if rnumpy:
        return np.array(im)[:, :, ::-1], apply_rotation, direction  # back to BGR
    else:
        return im, apply_rotation, direction


def visualize(image, bboxes, category_ids=None, category_id_to_name: dict = None, draw_categories=False):
    """
    Draw bounding boxes on an image and display the result with Matplotlib.

    Each bounding box is converted with :func:`bbox_convert`, drawn with
    :func:`draw_bbox`, then shown in a Matplotlib figure. Colors are selected
    from a Matplotlib colormap using the category id.

    Args:
        image: NumPy image array to annotate. The implementation uses
            ``image.copy()`` and ``image.shape``.
        bboxes: Iterable of bounding boxes accepted by
            :func:`bbox_convert`.
        category_ids: Iterable of category ids aligned with ``bboxes``. Each id
            is used to look up the class name and select a display color.
        category_id_to_name: Mapping from category id to display name.
        draw_categories: Whether to draw category text beside each bounding
            box.

    Returns:
        np.ndarray: Copy of ``image`` with bounding boxes drawn on it.

    Raises:
        RuntimeError: If Matplotlib cannot be imported.
    """
    try:
        # import matplotlib
        # matplotlib.use("GTK3Agg")  # Set GTK as the backend only if needed
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "Visualization requires matplotlib with GTK3 support. "
            "Please install it with: pip install matplotlib[gtk3]"
        ) from e

    def get_color_from_matplotlib(index: int, total_colors=10, colormap="tab10"):
        """
        Generate an RGB color from a Matplotlib colormap.

        Args:
            index: Category index used to choose a color.
            total_colors: Number of expected unique colors in the color range.
            colormap: Name of the Matplotlib colormap to sample.

        Returns:
            tuple[int, int, int]: RGB color with channel values from 0 to 255.
        """
        cmap = plt.get_cmap(colormap)  # Load specified colormap
        normalized_index = index / max(1, total_colors - 1)  # Normalize index within colormap range
        color = cmap(normalized_index)  # Get RGBA color from colormap
        return tuple(int(c * 255) for c in color[:3])  # Convert from (0-1) to (0-255)

    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        xyxy, xywh = bbox_convert([bbox], img.shape[:2], verbose=0)
        img = draw_bbox(img, xywh[0], class_name, box_color=get_color_from_matplotlib(category_id),
                        draw_text=draw_categories)  #one box at a time
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    return img

def normalize_illumination_gray(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize illumination in a grayscale image by dividing by a median-blurred background estimate.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    min_dim = min(h, w)

    # Large median blur estimates slow background illumination:
    # glare, shadows, exposure gradient.
    bg_ksize = max(31, int(min_dim * 0.06))
    bg_ksize = bg_ksize if bg_ksize % 2 == 1 else bg_ksize + 1 # use the first odd number
    background = cv2.medianBlur(gray, bg_ksize)

    normalized = cv2.divide(gray, background, scale=255)
    return normalized

def normalize_illumination(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalizes the illumination of a BGR image by adjusting its luminance channel.

    This function processes a BGR image to normalize its illumination, which can help reduce
    uneven lighting effects. The function separates the luminance channel from the image, adjusts
    it using a smoothed background estimation, and reassembles the normalized image in the
    original color format.

    Parameters:
    img_bgr: np.ndarray
        A BGR image with shape (height, width, 3) to be processed.

    Returns:
    np.ndarray
        The resulting BGR image after illumination normalization.
    """
    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)


    bg_ksize = max(31, int(min_dim * 0.06))
    bg_ksize = bg_ksize if bg_ksize % 2 == 1 else bg_ksize + 1
    background = cv2.medianBlur(l, bg_ksize)
    l_norm = cv2.divide(l, background, scale=255)

    lab_norm = cv2.merge([l_norm, a, b])
    img_norm_bgr = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

    return img_norm_bgr

def write_image(img, path: str):
    """
    Save an image to disk.

    NumPy inputs are assumed to be in BGR channel order and are converted to RGB
    before being saved through PIL. PIL ``Image.Image`` inputs are saved
    directly. Other input types are ignored by the current implementation.

    Args:
        img: Image to save, either a NumPy array in BGR order or a PIL image.
        path: Destination file path passed to ``Image.save``.

    Returns:
        None
    """
    if isinstance(img, np.ndarray):
        Image.fromarray(img[:, :, ::-1]).save(path)  # BGR -> RGB before saving PIL
    elif isinstance(img, Image.Image):
        img.save(path)


def read_image(path: str | Path, rnumpy=False, rexif=False) -> np.ndarray | PIL.Image.Image | tuple[np.ndarray | PIL.Image.Image, int]:
    """
    Read an image from disk and apply EXIF orientation correction.

    The image is opened with PIL and normalized with
    ``ImageOps.exif_transpose`` so EXIF orientation is reflected in the returned
    pixels. When NumPy output is requested, HEIF/HEIC images are converted to
    RGB first if HEIF support is available, then channel order is converted from
    RGB to BGR.

    Args:
        path: Path to the image file.
        rnumpy: When true, return a BGR NumPy array. When false, return a PIL
            image.
        rexif: When true, return the EXIF orientation tag. When false, return
            the image without the EXIF orientation tag.

    Returns:
        Image.Image | np.ndarray: EXIF-corrected PIL image, or BGR NumPy array
        if ``rnumpy`` is true.
    """
    path = Path(path)
    im = Image.open(path)

    # Extract EXIF orientation BEFORE transposing (Tag 274 / 0x0112 is Orientation)
    exif = im.getexif()
    orientation_tag = exif.get(0x0112, 1)  # Default to 1 if no EXIF is found

    # Apply the visual transposition
    im = ImageOps.exif_transpose(im)

    if rnumpy:
        if HAS_HEIF and path.suffix.lower() in [".heic", ".heif"]:
            im = im.convert("RGB")
        rgb = np.array(im)[:, :, ::-1] # BGR numpy array
        if rexif:
            return rgb, orientation_tag
        return rgb
    else:
        if rexif:
            return im, orientation_tag
        return im

if __name__ == "__main__":

    import argparse

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Apply image transformations to a file or a dataset.')
    parser.add_argument('-root', dest='root', type=str, default='.',
                        help='Path to folder containing the dataset root.', required=False)
    parser.add_argument('-outroot', dest='outroot', type=str, default='',
                        help='Path to folder where new altered dataset will be.', required=False)
    parser.add_argument('-file', dest='file', type=str, default='',
                        help='Path to image file to resize/visualize.', required=False)
    parser.add_argument('-im_size', dest='image_size', type=int, nargs="+", default=(None, None),
                        help='Image size tuple (Default (None,None)).')
    parser.add_argument('-ho', action='store_true', default=False, dest='horizontal',
                        help='Rotate image to horizontal orientation.')
    parser.add_argument('-ve', action='store_true', default=False, dest='vertical',
                        help='Rotate image to vertical orientation.')
    parser.add_argument('-cpu', dest='cpu', type=int, default=1,
                        help='Number of processes workers .')
    parser.add_argument('-v', action='count', default=0, dest='verbose',
                        help='Amount of verbosity (more \'v\'s means more verbose).')
    config, unparsed = parser.parse_known_args()

    config.image_size = tuple(config.image_size)

    if not config.outroot:
        config.outroot = os.path.join(os.path.dirname(config.outroot), "resized")

    if not os.path.isdir(config.outroot):
        os.mkdir(config.outroot)

    orientation = "h" if config.horizontal else "v" if config.vertical else ""
    rotate = True if (orientation == "h" or orientation == "v") else False
    r_img = image_resize(image=config.file,
                         width=config.image_size[0],
                         height=config.image_size[1],
                         rotate=rotate)
    rotated = image_rotate(im=config.file, orientation=orientation)

# -*- coding: utf-8 -*-
import os.path
import numpy as np
import cv2
from PIL import Image
from typing import Tuple


# Local imports
from .Bbox import bbox_convert, draw_bbox
from .Types import LetterboxParams

def compute_letterbox_params(orig_hw: tuple[int, int], target_wh: tuple[int, int]) -> LetterboxParams:
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
    Resize keeping aspect ratio. Should keep in mind that if the image is rotated, requested
    dimensions will be inverted.
    
    @param image: PIL Image or NDARRAY
    @param width: Resize to this width, adjust height accordingly
    @param height: Resize to the height, adjust width accordingly
    @param rotate: Rotates before resizing to match to the defined axis (horizontal or vertical)
    @param inter: Interpolation mode, given by OpenCV
    @return: Resized image as NDARRAY
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
                    return_params:bool = False) -> np.ndarray | Tuple[np.ndarray, LetterboxParams]:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
        params: Resized parameters (optional)
    """

    params = compute_letterbox_params(image.shape[:2], new_shape)


    image = cv2.resize(image, params.new_size)
    left, top, right, bottom = params.pad
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    if return_params:
        return image, params
    return image


def image_rotate(im, orientation = None, rnumpy = False, conditional = False):
    """
    Rotate image 90 degrees, depending on its shape. If the longest dim is width, make height the longest
    and vice versa
    @param im: Pillow Image, NDARRAY or string path
    @param orientation | STR: h for horizontal or v for vertical
    @param rnumpy: Boolean, return NDARRAY
    @param conditional: Boolean, only rotate if orientation is different from current dimensions. IE: if width is the
    largest dimension and orientation is h, do nothing
    @return: same as im or rotated image, apply_rotation (tells if rotation was applied)
    """

    if orientation is None:
        return im

    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    elif isinstance(im, str):
        im = Image.open(im)

    apply_rotation = not conditional or ((orientation == "v" and im.width > im.height) or
                                         (orientation == "h" and im.height > im.width))

    if orientation == 'v' and apply_rotation:
        im = im.rotate(90, expand=True)
    elif orientation == 'h' and apply_rotation:
        im = im.rotate(-90, expand=True)

    if rnumpy:
        return np.array(im), apply_rotation
    else:
        return im, apply_rotation


def visualize(image, bboxes, category_ids=None, category_id_to_name:dict=None, draw_categories=False):
    """
    Visualize bounding boxes on the image.

    @param image: PIL Image or NDARRAY
    @param bboxes: Bounding boxes as NDARRAY or list
    @param category_ids: Class IDs
    @param category_id_to_name: Dictionary, mapping from class ID to name
    @param draw_categories: Boolean, draw categories text along bboxes
    """
    try:
        import matplotlib
        matplotlib.use("GTK3Agg")  # Set GTK as the backend only if needed
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "Visualization requires matplotlib with GTK3 support. "
            "Please install it with: pip install matplotlib[gtk3]"
        ) from e

    def get_color_from_matplotlib(index: int, total_colors=10, colormap="tab10"):
        """
        Generates an RGB color from a Matplotlib colormap.

        :param index: Integer index.
        :param total_colors: Total expected unique colors.
        :param colormap: Matplotlib colormap.
        :return: Tuple (R, G, B) with values in range (0-255).
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


def write_image(img, path:str):
    if isinstance(img, np.ndarray):
        Image.fromarray(img).save(path)
    elif isinstance(img, Image.Image):
        img.save(path)

def read_image(path:str, rnumpy=False):
    """
    Read an image from path
    @param path: Path to image
    @param rnumpy: Boolean, return NDARRAY
    """

    im = Image.open(path)
    if rnumpy:
        return np.array(im)
    else:
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
    resized = image_resize(image=config.file,
                 width=config.image_size[0],
                 height=config.image_size[1],
                 rotate=rotate)
    rotated = image_rotate(im=config.file, orientation=orientation)

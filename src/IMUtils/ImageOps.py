# -*- coding: utf-8 -*-
import os.path
import numpy as np
import matplotlib
matplotlib.use("GTK3Agg")  # Set GTK as the backend
import matplotlib.pyplot as plt
from PIL import Image

# Local imports
from .Bbox import bbox_convert, draw_bbox


def get_color_from_matplotlib(index:int, total_colors=10, colormap="tab10"):
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

def resize_with_pad(image: np.array,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])

    if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
        ratio = float(min(new_shape)) / min(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])

    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0] if new_shape[0] > new_size[0] else 0
    delta_h = new_shape[1] - new_size[1] if new_shape[1] > new_size[1] else 0
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
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
    parser.add_argument('-vis', action='store_true', default=False, dest='visualize',
                        help='Rotate image to vertical orientation.')
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

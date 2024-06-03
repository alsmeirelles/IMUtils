# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image

def image_resize(image, width=None, height=None, rotate=False, inter=cv2.INTER_AREA):
    """
    Resize keeping aspect ratio
    @param image: Image as NDARRAY
    @param width: Resize to this width, adjust height accordingly
    @param height: Resize to the height, adjust width accordingly
    @param rotate: Rotates before resizing to match to the defined axis (horizontal or vertical)
    @param inter: Interpolation mode, given by OpenCV
    @return: Resized image as NDARRAY
    """

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    if rotate and h != w:
        image = image_rotate(image, vert=True) if w > h else image_rotate(image, horiz=True)

    # check to see if the width is None
    if width is None:
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
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def image_rotate(im, vert=False, horiz=False, rnumpy = True):
    """
    Rotate image 90 degrees, depending on its shape. If the longest dim is width, make height the longest
    and vice-versa
    @param im: Pillow Image, NDARRAY or string path
    @param vert: Boolean, make vertical.
    @param horiz: Boolean, make horizontal
    @param rnumpy: Boolean, return NDARRAY
    @return: same as im, rotated image
    """

    if not isinstance(im, Image.Image):
        im = Image.fromarray(im)

    if vert:
        im = im.rotate(90, expand=True)
    elif horiz:
        im = im.rotate(-90, expand=True)

    if rnumpy:
        return np.array(im)
    else:
        return im

if __name__ == "__main__":

    import argparse

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Train an AD model from Anomalib.')
    parser.add_argument('-root', dest='root', type=str, default='/TrueVision/2-Datasets/BeefAnom',
                        help='Path to folder containing the dataset root.', required=False)
    parser.add_argument('-im_size', dest='image_size', type=int, nargs="+", default=(224, 224),
                        help='Image size tuple (Default (224,224)).')
    parser.add_argument('-h', action='store_true', default=False, dest='horizontal',
                        help='Rotate image to horizontal orientation.')
    parser.add_argument('-ve', action='store_true', default=False, dest='vertical',
                        help='Rotate image to vertical orientation.')
    parser.add_argument('-v', action='count', default=0, dest='verbose',
                        help='Amount of verbosity (more \'v\'s means more verbose).')
    config, unparsed = parser.parse_known_args()

    config.image_size = tuple(config.image_size)

    if config.image_size != (224, 224):
        config.should_resize = True
    else:
        config.should_resize = False
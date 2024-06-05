# -*- coding: utf-8 -*-
import os.path
import numpy as np
from PIL import Image

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
    resized = image.resize(size=dim, resample=inter)

    # apply rotation
    if rotate and h != w:
        orientation = 'h' if width is None else 'h'
        resized = image_rotate(resized, orientation=orientation)

    # return the resized image
    return resized

def image_rotate(im, orientation=None, rnumpy = False):
    """
    Rotate image 90 degrees, depending on its shape. If the longest dim is width, make height the longest
    and vice-versa
    @param im: Pillow Image, NDARRAY or string path
    @param orientation | STR: h for horizontal or v for vertical
    @param rnumpy: Boolean, return NDARRAY
    @return: same as im, rotated image
    """

    if orientation is None:
        return im

    if not isinstance(im, Image.Image):
        im = Image.fromarray(im)

    if orientation == 'v':
        im = im.rotate(90, expand=True)
    elif orientation == 'h':
        im = im.rotate(-90, expand=True)

    if rnumpy:
        return np.array(im)
    else:
        return im

def multiprocess_resize():
    pass

if __name__ == "__main__":

    import argparse

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Apply image transformations to a file or a dataset.')
    parser.add_argument('-root', dest='root', type=str, default='/TrueVision/2-Datasets/BeefAnom',
                        help='Path to folder containing the dataset root.', required=False)
    parser.add_argument('-outroot', dest='outroot', type=str, default='',
                        help='Path to folder where new altered dataset will be.', required=False)
    parser.add_argument('-file', dest='file', type=str, default='',
                        help='Path to image file to resize.', required=False)
    parser.add_argument('-im_size', dest='image_size', type=int, nargs="+", default=(224, 224),
                        help='Image size tuple (Default (224,224)).')
    parser.add_argument('-h', action='store_true', default=False, dest='horizontal',
                        help='Rotate image to horizontal orientation.')
    parser.add_argument('-ve', action='store_true', default=False, dest='vertical',
                        help='Rotate image to vertical orientation.')
    parser.add_argument('-cpu', dest='cpu', type=int, default=1,
                        help='Number of processes workers .')
    parser.add_argument('-v', action='count', default=0, dest='verbose',
                        help='Amount of verbosity (more \'v\'s means more verbose).')
    config, unparsed = parser.parse_known_args()

    config.image_size = tuple(config.image_size)

    if config.image_size != (224, 224):
        config.should_resize = True
    else:
        config.should_resize = False

    if not config.outroot:
        config.outroot = os.path.join(os.path.dirname(config.outroot), "resized")

    if not os.path.isdir(config.outroot):
        os.mkdir(config.outroot)


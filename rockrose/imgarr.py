"""
"""

import numpy as np

from PIL import Image

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

import matplotlib.pyplot as plt


# #### #### #
# use Image

def im_open(fp, mode='r'):
    im = Image.open(fp, mode=mode)
    return im

def im_resize(im, *args, **kwargs):
    im = im.resize(*args, **kwargs)
    return im

def im_convert(im, *args, **kwargs):
    """
    L = R * 299/1000 + G * 587/1000 + B * 114/1000
    """
    im = im.convert(*args, **kwargs)
    return im


def im_test_1():  # -o-
    im = im_open('../data/gymfdt_2.jpg')

    irr = np.array(im)
    print irr.shape  # (570, 798, 3)

    #rrr = np.rep
    plt.imshow(im)
    plt.show()


def im_test_2():  # -o-
    im = im_open('../data/gymfdt_2.jpg')

    irr = np.array(im)
    print irr.shape

    im = im.resize((80, 80)).convert('L')

    irr = np.array(im)
    print irr.shape

    #rrr = np.repeat(irr, 3, axis=0)
    iro = irr.reshape(1, irr.shape[0], irr.shape[1])
    print iro.shape
    rrr = np.repeat(iro, 3, axis=0)
    print rrr.shape
    rrri = np.swapaxes(np.swapaxes(rrr, 0, 1), 1, 2)
    print rrri.shape

    '''
    (120, 120, 3)
    (80, 80)
    (1, 80, 80)
    (3, 80, 80)
    (80, 80, 3)
    '''

    #plt.imshow(im)
    #plt.imshow(irr)
    plt.imshow(rrri)
    plt.show()


# #### #### #
# use skimage


def ski_cl_rgb2gray(*args, **kwargs):
    """
    Y = 0.2125 R + 0.7154 G + 0.0721 B
    """
    orr = skimage.color.rgb2gray(*args, **kwargs)
    return orr


def ski_tr_resize(*args, **kwargs):
    orr = skimage.transform.resize(*args, **kwargs)
    return orr


def im_test_2_2():  # -o-
    im = im_open('../data/gymfdt_2.jpg')

    #im = im.resize((80, 80)).convert('L')

    irr = np.array(im)
    print irr.shape

    #irr = ski_cl_rgb2gray(irr)
    irr = ski_tr_resize(irr, (80, 80))
    print irr.shape
    irr = ski_cl_rgb2gray(irr)
    print irr.shape

    #rrr = np.repeat(irr, 3, axis=0)
    iro = irr.reshape(1, irr.shape[0], irr.shape[1])
    print iro.shape
    rrr = np.repeat(iro, 3, axis=0)
    print rrr.shape
    rrri = np.swapaxes(np.swapaxes(rrr, 0, 1), 1, 2)
    print rrri.shape

    '''
    (120, 120, 3)
    (80, 80, 3)
    (80, 80)
    (1, 80, 80)
    (3, 80, 80)
    (80, 80, 3)
    '''

    #plt.imshow(im)
    #plt.imshow(irr)
    plt.imshow(rrri)
    plt.show()


if __name__ == '__main__':
    #im_test_1()
    #im_test_2()

    im_test_2_2()

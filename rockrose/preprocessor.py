"""
Input data and output reward preprocessor.

# TODO:
* use imgarr.py
"""

import numpy as np

import skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer


class RRPreprocessorBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def process(self, x_t, s_t=None):
        pass

    def process_reward(self, r_t, *args, **kwargs):
        return r_t


class RRPreprImgGrayN(RRPreprocessorBase):
    """Output N gray frames.
    """
    def __init__(self, channels=4, out_size=(84, 84),
                 #out_range=(0, 255),
                 out_range=None,
                 *args, **kwargs):
        super(RRPreprImgGrayN, self).__init__()

        self.channels = channels
        self.out_size = out_size
        self.out_range = out_range

    def process(self, x_t, s_t=None):

        if s_t is None:
            x_t = skimage.color.rgb2gray(x_t)
            x_t = skimage.transform.resize(x_t, self.out_size)
            if self.out_range is not None:
                x_t = skimage.exposure.rescale_intensity(
                    x_t, out_range=self.out_range)
            x_ts = []
            for i in range(self.channels):
                x_ts.append(x_t)
            s_t = np.stack(x_ts, axis=0)
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
            return s_t

        else:
            x_t1 = skimage.color.rgb2gray(x_t)
            x_t1 = skimage.transform.resize(x_t1, self.out_size)
            if self.out_range is not None:
                x_t1 = skimage.exposure.rescale_intensity(
                    x_t1, out_range=self.out_range)
            x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
            s_t1 = np.append(x_t1, s_t[:, :self.channels - 1, :, :], axis=1)
            return s_t1


class RRPreprImgGrayA(RRPreprocessorBase):
    """Output One gray frame.
    """
    def __init__(self, channels=4, out_size=(84, 84),
                 #out_range=(0, 255),
                 out_range=None,
                 *args, **kwargs):
        super(RRPreprImgGrayN, self).__init__()

        self.channels = channels
        self.out_size = out_size
        self.out_range = out_range

    def process(self, x_t, s_t=None):

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(84,84))
        if self.out_range is not None:
            x_t1 = skimage.exposure.rescale_intensity(
                x_t1, out_range=self.out_range)
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        s_t1 = x_t1
        return s_t1


class RRPreprImgRGB(RRPreprocessorBase):
    """Output 3 RGB frames.
    """
    def __init__(self, channels=3, out_size=(84, 84),
                 #out_range=(0, 255),
                 out_range=None,
                 *args, **kwargs):
        super(RRPreprImgRGB, self).__init__()

        self.channels = channels
        self.out_size = out_size
        self.out_range = out_range

    def process(self, x_t, s_t=None):

        x_t = skimage.transform.resize(x_t, self.out_size)  # (84, 84, 3)
        if self.out_range is not None:
            x_t = skimage.exposure.rescale_intensity(
                x_t, out_range=self.out_range)
        x_t = np.swapaxes(np.swapaxes(x_t, 1, 2), 0, 1)  # (3, 84, 84)
        s_t = x_t
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
        return s_t


class RRPreprImgGrayN4R(RRPreprocessorBase):
    """Output N gray frames with reverse order.
    """
    def __init__(self, channels=4, out_size=(84, 84),
                 #out_range=(0, 255),
                 out_range=None,
                 *args, **kwargs):
        super(RRPreprImgGrayN4R, self).__init__()

        self.channels = channels
        self.out_size = out_size
        self.out_range = out_range

    def process(self, x_t, s_t=None):

        if s_t is None:
            x_t = skimage.color.rgb2gray(x_t)
            x_t = skimage.transform.resize(x_t, self.out_size)
            if self.out_range is not None:
                x_t = skimage.exposure.rescale_intensity(
                    x_t, out_range=self.out_range)
            x_ts = []
            for i in range(self.channels):
                x_ts.append(x_t)
            s_t = np.stack(x_ts, axis=0)
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
            return s_t

        else:
            x_t1 = skimage.color.rgb2gray(x_t)
            x_t1 = skimage.transform.resize(x_t1, self.out_size)
            if self.out_range is not None:
                x_t1 = skimage.exposure.rescale_intensity(
                    x_t1, out_range=self.out_range)

            s_t1 = np.empty((self.channels, self.out_size[0], self.out_size[1]))
            s_t1[:self.channels-1, ...] = s_t.reshape(
                (s_t.shape[1], s_t.shape[2], s_t.shape[3])).copy()[:self.channels-1, ...]
            s_t1[self.channels-1] = x_t1

            s_t1 = s_t1.reshape((1, s_t1.shape[0], s_t1.shape[1], s_t1.shape[2]))

            return s_t1

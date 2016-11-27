"""Segment class that holds all important attributes of a segment defined by
superpixels
"""

import cv2
import numpy as np
from scipy import ndimage


class Segment(object):
    def __init__(self, index):
        self.__index = index

        self.__left = None
        self.__top = None
        self.__bottom = None
        self.__right = None

        self.__count = 0
        self.__image = None
        self.__mask = None

    @property
    def index(self):
        return self.__index

    @property
    def left(self):
        return self.__left

    @property
    def top(self):
        return self.__top

    @property
    def right(self):
        return self.__right

    @property
    def bottom(self):
        return self.__bottom

    @property
    def count(self):
        return self.__count

    @property
    def image(self):
        return self.__image

    @property
    def mask(self):
        return self.__mask

    @property
    def width(self):
        return 1 + self.right - self.left

    @property
    def height(self):
        return 1 + self.bottom - self.top

    @property
    def center(self):
            center = ndimage.measurements.center_of_mass(self.mask)
            return (self.left + center[1], self.top + center[0])

    @property
    def mean(self):
        return cv2.mean(self.image, self.mask)

    @staticmethod
    def generate(image, superpixels):
        superpixels = np.array(superpixels)
        image = np.array(image)
        segment_values = np.unique(superpixels)
        segments = {i: Segment(i) for i in segment_values}

        for y, row in enumerate(superpixels):
            for x, value in enumerate(row):
                s = segments[value]

                s.__left = x if s.left is None else min(s.left, x)
                s.__top = y if s.top is None else min(s.top, y)
                s.__right = x if s.right is None else max(s.right, x)
                s.__bottom = y if s.bottom is None else max(s.bottom, y)
                s.__count += 1

        for index in segments:
            s = segments[index]

            s.__image = image[s.top:s.top+s.height, s.left:s.left+s.width]

            sliced = superpixels[s.top:s.top+s.height, s.left:s.left+s.width]
            s.__mask = np.zeros(sliced.shape, dtype=np.uint8)
            s.__mask[sliced == s.index] = 255

        return segments

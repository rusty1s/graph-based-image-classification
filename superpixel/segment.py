"""A segment defined by a superpixel calculation of an image
"""

import cv2
import numpy as np
from scipy import ndimage


class Segment(object):
    def __init__(self, index):
        self.__index = index

        self.__left = float('inf')
        self.__top = float('inf')
        self.__bottom = float('-inf')
        self.__right = float('-inf')

        self.__count = 0
        self.__image = []
        self.__mask = []

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

                s.__left = min(s.left, x)
                s.__top = min(s.top, y)
                s.__right = max(s.right, x)
                s.__bottom = max(s.bottom, y)
                s.__count += 1

        for index in segments:
            s = segments[index]

            slice_x = slice(s.left, s.right + 1)
            slice_y = slice(s.top, s.bottom + 1)

            s.__image = image[slice_y, slice_x]

            sliced_superpixels = superpixels[slice_y, slice_x]
            s.__mask = np.zeros(sliced_superpixels.shape, dtype=np.uint8)
            s.__mask[sliced_superpixels == s.index] = 255

        return segments

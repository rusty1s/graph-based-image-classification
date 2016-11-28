"""A segment defined by a superpixel calculation of an image
"""

import cv2
import numpy as np
from scipy import ndimage
import pickle


class Segment(object):
    def __init__(self, id):
        self.__id = id

        self.__left = float('inf')
        self.__top = float('inf')
        self.__bottom = float('-inf')
        self.__right = float('-inf')

        self.__count = 0
        self.__image = []
        self.__mask = []

        self.__neighbors = set()

    @property
    def id(self):
        """The identifier of the segment."""
        return self.__id

    @property
    def left(self):
        """The left coordinate of the bounding box of th segment."""

        return self.__left

    @property
    def top(self):
        """The top coordinate of the bounding box of th segment."""

        return self.__top

    @property
    def right(self):
        """The right coordinate of the bounding box of th segment."""

        return self.__right

    @property
    def bottom(self):
        """The bottom coordinate of the bounding box of th segment."""

        return self.__bottom

    @property
    def count(self):
        """The amount of pixels that are in the segment."""

        return self.__count

    @property
    def image(self):
        """The sliced image of the segment in the shape of its bounding box."""

        return self.__image

    @property
    def mask(self):
        """The mask of the segment with the shape of its bounding box. A 255
        means that the pixel is in the segment."""

        return self.__mask

    @property
    def neighbors(self):
        """The set of spatial neighbors of the segment."""

        return self.__neighbors

    @property
    def width(self):
        """The width of the bounding box of the segment."""

        return 1 + self.right - self.left

    @property
    def height(self):
        """The height of the bounding box of the segment."""

        return 1 + self.bottom - self.top

    @property
    def center(self):
        """The center of the segment."""

        center = ndimage.measurements.center_of_mass(self.mask)
        return (self.left + center[1], self.top + center[0])

    @property
    def mean(self):
        """The mean color of the segment."""

        return cv2.mean(self.image, self.mask)

    @staticmethod
    def generate(image, superpixels):
        """A dictionary of all the segments found in the superpixel
        presentation with its identifier as key."""

        superpixels = np.array(superpixels)
        image = np.array(image)
        segment_values = np.unique(superpixels)
        segments = {i: Segment(i) for i in segment_values}

        width = len(superpixels[0])
        height = len(superpixels)

        for y, row in enumerate(superpixels):
            for x, value in enumerate(row):
                s = segments[value]

                s.__left = min(s.left, x)
                s.__top = min(s.top, y)
                s.__right = max(s.right, x)
                s.__bottom = max(s.bottom, y)
                s.__count += 1

                # calculate and update neighbors
                slice_x = Segment.__get_1x1_slice(x, width)
                slice_y = Segment.__get_1x1_slice(y, height)
                s.neighbors.update(superpixels[slice_y, slice_x].flatten())

        for id in segments:
            s = segments[id]

            # remove itself from neighborhood
            s.neighbors.discard(id)

            slice_x = slice(s.left, s.right + 1)
            slice_y = slice(s.top, s.bottom + 1)

            s.__image = image[slice_y, slice_x]

            sliced_superpixels = superpixels[slice_y, slice_x]
            s.__mask = np.zeros(sliced_superpixels.shape, dtype=np.uint8)
            s.__mask[sliced_superpixels == s.id] = 255

        return segments

    @staticmethod
    def write(segments, image_name, suffix=None):
        """Writes the generated dictionary of segments to disk."""

        suffix = '' if suffix is None else '-{}'.format(suffix)
        name = '{}_segments{}.pkl'.format(image_name, suffix)

        with open(name, 'wb') as output:
            pickle.dump(segments, output, -1)

    @staticmethod
    def read(name):
        """Reads the written dictionary of segments from disk."""

        with open('{}.pkl'.format(name), 'rb') as input:
            return pickle.load(input)

    @staticmethod
    def __get_1x1_slice(index, maximum):
        return slice(max(0, index - 1), min(maximum, index + 2))

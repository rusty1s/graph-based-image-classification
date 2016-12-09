"""A segment defined by a superpixel calculation of an image
"""

import cv2
import numpy as np
from scipy import ndimage

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle


class Segment(object):
    def __init__(self, id):
        self.__id = id
        self.__order = None

        self.__left = float('inf')
        self.__top = float('inf')
        self.__bottom = float('-inf')
        self.__right = float('-inf')

        self.__count = 0
        self.__covered = None

        self.__image = []
        self.__mask = []

        self.__neighbors = set()

    @property
    def id(self):
        """The identifier of the segment. Corresponds to the the value in the
        superpixel representation."""

        return self.__id

    @property
    def order(self):
        """The order of the segment. Corresponds to the scan line order of the
        segments."""

        return self.__order

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
    def covered(self):
        """The amount of pixels that are in the segment in relation to the
        amount of pixels in the image."""

        return self.__covered

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
        presentation with its identifier in the superpixel representation as
        the key."""

        # Convert arguments to 2D numpy arrays.
        superpixels = np.array(superpixels)
        image = np.array(image)

        # Create the dictionary of empty segments (except identifier).
        segment_values = np.unique(superpixels)
        segments = {i: Segment(i) for i in segment_values}

        # Image respectively superpixels dimensions.
        width = len(superpixels[0])
        height = len(superpixels)

        order = 0

        # Iterate over each pixel once and collect as much information about
        # the segments.
        for y, row in enumerate(superpixels):
            for x, value in enumerate(row):
                s = segments[value]

                # Set the order and increment (if not already set).
                order = s.__compute_order(order)

                # Compute the bounding box.
                s.__left = min(s.left, x)
                s.__top = min(s.top, y)
                s.__right = max(s.right, x)
                s.__bottom = max(s.bottom, y)

                # Increment pixel count of segment.
                s.__count += 1

                # Calculate and update neighbors.
                slice_x = Segment.__get_1x1_slice(x, width)
                slice_y = Segment.__get_1x1_slice(y, height)
                s.neighbors.update(superpixels[slice_y, slice_x].flatten())

        # Iterate over each segment once and extend the collection information.
        for s in segments.values():
            # Remove itself from neighborhood.
            s.neighbors.discard(s.id)

            # Calculate covered area.
            s.__covered = float(s.count)/(width * height)

            # Slice the image to the segments bounding box.
            slice_x = slice(s.left, s.right + 1)
            slice_y = slice(s.top, s.bottom + 1)

            s.__image = image[slice_y, slice_x]

            # Compute the mask.
            s.__mask = Segment.__compute_mask(superpixels[slice_y, slice_x],
                                              s.id)

        return segments

    @staticmethod
    def write(segments, image_name, suffix=None):
        """Writes the generated dictionary of segments to disk."""

        # Compute the correct file name.
        suffix = '' if suffix is None else '-{}'.format(suffix)
        file_name = '{}_segments{}.pkl'.format(image_name, suffix)

        with open(file_name, 'wb') as output:
            pickle.dump(segments, output, -1)

    @staticmethod
    def read(file_name):
        """Reads the written dictionary of segments from disk. The file name
        shouldn't have a file extension."""

        with open('{}.pkl'.format(file_name), 'rb') as input:
            return pickle.load(input)

    @staticmethod
    def __get_1x1_slice(index, maximum):
        """Returns the 1x1 neighborhood of `index` exlucding values lower than
        0 and greater than or equal to maximum."""

        return slice(max(0, index - 1), min(maximum, index + 2))

    @staticmethod
    def __compute_mask(image, value):
        """Returns the mask of an image, where everything not equals `value` is
        0 and everything equals `value` is 255."""

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[image == value] = 255
        return mask

    def __compute_order(self, order):
        """Sets the order of the segment if not already defined and increments
        it."""

        if self.order is None:
            self.__order = order
            return order + 1

        return order

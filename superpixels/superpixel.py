import cv2
import numpy as np
from scipy import ndimage


class Superpixel(object):
    """Data class that holds all the information of a single superpixel."""

    def __init__(self, id=None, order=None, left=0, top=0, neighbors=None,
                 image=np.empty((0, 0, 3), dtype=np.float),
                 mask=np.empty((0, 0, 1), dtype=np.uint8)):

        # The identifier of the superpixel.
        self.id = id

        # The order of the superpixel. Corresponds to the scan line order of
        # the superpixels.
        self.order = order

        # The left coordinate of the bounding box of the superpixel.
        self.left = left

        # The top coordinate of the bounding box of the superpixel.
        self.top = top

        # The set of spatial neighbors of the superpixel saved by their id.
        self.neighbors = set() if neighbors is None else neighbors

        # The sliced image of the superpixel in the shape of its bounding box.
        self.image = image

        # The mask of the superpixel with the shape of its bounding box. A `0`
        # means that the pixel is outside of the superpixel, otherwise inside.
        self.mask = mask

    @property
    def width(self):
        """The width of the bounding box of the superpixel."""

        return self.mask.shape[1]

    @property
    def height(self):
        """The height of the bounding box of the superpixel."""

        return self.mask.shape[0]

    @property
    def count(self):
        """The amount of pixels that are inside the superpixel."""

        return np.count_nonzero(self.mask)

    @property
    def relative_center(self):
        """The relative center of the superpixel in the superpixels bounding
        box."""

        if self.width == 0 or self.height == 0:
            return (0, 0)

        # Calculate the center and reverse the order of the resulting tuple.
        return ndimage.measurements.center_of_mass(self.mask)[::-1]

    @property
    def absolute_center(self):
        """The absolute center of the superpixel."""

        relative_center = self.relative_center
        return (self.left + relative_center[0], self.top + relative_center[1])

    @property
    def mean(self):
        """The mean color of the superpixel."""

        # Calculate mean color and remove the alpha channel
        return cv2.mean(self.image, self.mask)[:-1]

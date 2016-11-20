import numpy as np
from scipy import ndimage


class Segment(object):
    def __init__(self, index):
        self.index = index

        self.left = None
        self.top = None

        # initialize with 0, so we have an easier time calculating th values
        self.width = 0
        self.height = 0

        self.mask = None
        self.count = None
        self.mean = None
        self.center = None
        self.neighbors = None

    @staticmethod
    def generate(superpixels):
        superpixels = np.array(superpixels)
        segment_values = np.unique(superpixels)
        segments = {i: Segment(i) for i in segment_values}

        # compute top/left
        for y, row in enumerate(superpixels):
            for x, v in enumerate(row):
                s = segments[v]

                s.left = x if s.left is None else min(s.left, x)
                s.top = y if s.top is None else min(s.top, y)

        # compute width/height
        for y, row in enumerate(superpixels):
            for x, v in enumerate(row):
                s = segments[v]

                s.width = max(s.width, x + 1 - s.left)
                s.height = max(s.height, y + 1 - s.top)

        # compute mask and center
        for i in segments:
            s = segments[i]
            sliced = superpixels[s.top:s.top+s.height, s.left:s.left+s.width]
            s.mask = np.zeros(sliced.shape, dtype=np.uint8)
            s.mask[sliced == s.index] = 255
            s.center = ndimage.measurements.center_of_mass(s.mask)

        return segments

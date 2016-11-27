"""Segment class that holds all important attributes of a segment defined by
superpixels
"""

import numpy as np
from scipy import ndimage


class Segment(object):
    def __init__(self, index):
        self.index = index

        # bounding box
        self.left = None
        self.top = None
        self.right = None
        self.bottom = None
        self.width = 0
        self.height = 0

        self.count = 0  # amount of pixels
        self.share = 0  # percentual amount of pixels

        self.mask = None

        self.center = None
        self.mean = None

        self.neighbors = set()

    @staticmethod
    def generate(superpixels):
        # compute the image dimensions
        height = len(superpixels)
        width = len(superpixels[0])
        count = width * height

        superpixels = np.array(superpixels)
        segment_values = np.unique(superpixels)
        segments = {i: Segment(i) for i in segment_values}

        # compute bounding box
        for y, row in enumerate(superpixels):
            for x, value in enumerate(row):
                s = segments[value]

                s.left = x if s.left is None else min(s.left, x)
                s.top = y if s.top is None else min(s.top, y)
                s.right = x if s.right is None else max(s.right, x)
                s.bottom = y if s.bottom is None else max(s.bottom, y)

                s.count += 1

        # compute for every segment
        for i in segments:
            s = segments[i]

            s.width = 1 + s.right - s.left
            s.height = 1 + s.bottom - s.top

            sliced = superpixels[s.top:s.top+s.height, s.left:s.left+s.width]
            s.mask = np.zeros(sliced.shape, dtype=np.uint8)
            s.mask[sliced == s.index] = 255

            center = ndimage.measurements.center_of_mass(s.mask)
            s.center = (s.left + center[1], s.top + center[0])

            s.share = s.count/float(count)

        return segments

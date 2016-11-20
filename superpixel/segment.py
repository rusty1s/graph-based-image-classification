import numpy as np


class Segment(object):
    def __init__(self, index, left=None, top=None, width=1, height=1):
        self.index = index
        self.left = left
        self.top = top
        self.width = width
        self.height = height

        # compute later
        self.mask = None
        self.mean = None
        self.center = None
        self.neighbors = None

    @staticmethod
    def generate(superpixel):
        superpixel = np.array(superpixel)
        segment_values = np.unique(superpixel)
        segments = {i: Segment(i) for i in segment_values}

        # compute top/left
        for y, row in enumerate(superpixel):
            for x, v in enumerate(row):
                s = segments[v]

                s.left = x if s.left is None else min(s.left, x)
                s.top = y if s.top is None else min(s.top, y)

        # compute width/height
        for y, row in enumerate(superpixel):
            for x, v in enumerate(row):
                s = segments[v]

                s.width = max(s.width, x + 1 - s.left)
                s.height = max(s.height, y + 1 - s.top)

        for i in segments:
            s = segments[i]
            sliced = superpixel[s.top:s.top+s.height, s.left:s.left+s.width]
            s.mask = np.zeros(sliced.shape, dtype=np.uint8)
            s.mask[sliced == s.index] = 255

        return segments

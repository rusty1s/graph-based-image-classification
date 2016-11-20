class Segment(object):
    def __init__(self, index, start, shape, mean, neighbors):
        self.index = index
        self.start = start
        self.shape = shape
        self.neighbors = neighbors
        self.mean = (0, 0, 0)
        self.center = (0, 0)

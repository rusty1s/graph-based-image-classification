from grapher import Grapher

from .feature_extraction import feature_extraction, NUM_FEATURES
from .adjacency import adjacencies


def SegmentationGrapher(Grapher):

    def __init__(self, algorithm, adjacency_name):
        self._algorithm = algorithm
        self._adjacency_name = adjacency_name

    @property
    def num_node_channels(self):
        return NUM_FEATURES

    @property
    def num_adjacency_matrices(self):
        return 1

    def create_graph(self, image):
        segmentation = self._algorithm(image)

        nodes = feature_extraction(segmentation, image)
        adjacency = adjacencies[self._adjacency_name](segmentation)

        return nodes, adjacency

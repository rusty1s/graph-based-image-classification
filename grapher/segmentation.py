from __future__ import absolute_import

from segmentation import feature_extraction, NUM_FEATURES,\
                         adjacency_euclidean_distance

from .grapher import Grapher


class SegmentationGrapher(Grapher):

    def __init__(self, segmentation_algorithm, adjacency_algorithm):
        self._segmentation_algorithm = segmentation_algorithm
        self._adjacency_algorithm = adjacency_algorithm

    @property
    def num_node_channels(self):
        return NUM_FEATURES

    @property
    def num_adjacency_matrices(self):
        return 1

    def create_graph(self, image):
        segmentation = self._segmentation_algorithm(image)

        nodes = feature_extraction(segmentation, image)
        adjacency = self._adjacency_algorithm(segmentation)

        return nodes, adjacency

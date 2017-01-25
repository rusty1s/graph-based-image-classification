from __future__ import absolute_import

import tensorflow as tf

from segmentation import feature_extraction, NUM_FEATURES
from segmentation import adjacency_euclidean_distance

from .grapher import Grapher


class SegmentationGrapher(Grapher):
    """A graph generator by segmenting input images."""

    def __init__(self, segment, adjacency_from_segmentation):
        """Creates a graph generator by segmenting input images.

        Args:
            segment: A segmentation algorithm that takes a sinle input image.
            adjacency_from_segmentation: An adjacency generation algorithm
              that produces an adjacency matrix based on a given segmentation.
        """

        self._segment = segment
        self._adjacency_from_segmentation = adjacency_from_segmentation

    @property
    def num_node_channels(self):
        """The number of corresponding channels for each node in the graph.

        Returns:
            A number.
        """

        return NUM_FEATURES

    @property
    def num_edge_channels(self):
        """The number of corresponding channels for each edge in the graph.

        Returns:
            A number.
        """

        return 1

    def create_graph(self, image):
        """Generates a graph based on the passed data. Note that the number of
          nodes can vary depending on the number of segments the segmentation
          algorithm produces.

        Args:
            image: The image.

        Returns:
            nodes: A numpy array that holds the channels for each node in the
              shape [num_nodes, num_node_channels].
            adjacencies: An numpy array that holds the (multiple) adjacency
              matrices of the graph in the shape
              [num_nodes, num_nodes, num_edge_channels].
        """

        segmentation = self._segment(image)

        # Compute the nodes and adjacency matrix based on the segmentation.
        nodes = feature_extraction(segmentation, image)
        adjacency = self._adjacency_from_segmentation(segmentation)

        # Till now, we only consider one adjacency matrix.
        return nodes, tf.expand_dims(adjacency, axis=2)

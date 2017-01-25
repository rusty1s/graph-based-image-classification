from __future__ import absolute_import

import tensorflow as tf

from segmentation import feature_extraction, NUM_FEATURES
from segmentation.algorithm import json_generators as segmentations
from segmentation import adjacencies

from .grapher import Grapher


class SegmentationGrapher(Grapher):
    """A graph generator by segmenting input images."""

    def __init__(self, segment, adjacencies_from_segmentation):
        """Creates a graph generator by segmenting input images.

        Args:
            segment: A segmentation algorithm that takes a sinle input image.
            adjacencies_from_segmentation: An array of adjacency generation
              algorithms which each produces an adjacency matrix based on a
              computed segmentation.
        """

        self._segment = segment
        self._adjacencies_from_segmentation = adjacencies_from_segmentation

    @classmethod
    def create(cls, obj):
        segmentation = obj['segmentation']
        adj = obj['adjacencies_from_segmentation']

        return cls(
            segmentations[segmentation['name']](segmentation),
            [adjacencies[i] for i in adj])

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

        return len(self._adjacencies_from_segmentation)

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

        # Compute the nodes and adjacency matrices based on the segmentation.
        nodes = feature_extraction(segmentation, image)
        adjacency = self._adjacency_from_segmentation[0](segmentation)

        # Till now, we only consider one adjacency matrix.
        return nodes, tf.expand_dims(adjacency, axis=2)

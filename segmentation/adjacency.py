import tensorflow as tf
import networkx as nx
import numpy as np
from skimage.future.graph import RAG


CONNECTIVITY = 2


def adjacency_unweighted(segmentation, connectivity=CONNECTIVITY):
    """Computes the adjacency matrix of the Region Adjacency Graph.

    Given an segmentation, this method constructs the constructs the
    corresponding Region Adjacency Graphh (RAG). Each node in the RAG
    represents a set of pixels with the same label in `segmentation`. An edge
    between two nodes exist if the nodes are spatial connected.

    Args:
        segmentation: The segmentation.
        connectivity: Integer. Pixels with a squared distance less than
          `connectivity` from each other are considered adjacent (optional).

    Returns:
        An adjacent matrix with shape [num_segments, num_segments].
    """

    def _adjacency(segmentation):
        graph = RAG(segmentation, connectivity=connectivity)

        # Simply return the unweighted adjacency matrix of the computed graph.
        return nx.to_numpy_matrix(graph, dtype=np.float32)

    return tf.py_func(
        _adjacency, [segmentation], tf.float32, stateful=False,
        name='adjacency_unweighted')


def adjacency_euclidean_distance(segmentation, connectivity=CONNECTIVITY):
    """Computes the adjacency matrix of the Region Adjacency Graph using the
    euclidian distance between the centroids of adjacent segments.

    Given an segmentation, this method constructs the constructs the
    corresponding Region Adjacency Graphh (RAG). Each node in the RAG
    represents a set of pixels with the same label in `segmentation`. An edge
    between two nodes exist if the nodes are spatial connected. The weight
    between two adjacent regions represents represents how nearby tow segments
    are.

    Args:
        segmentation: The segmentation.
        connectivity: Integer. Pixels with a squared distance less than
          `connectivity` from each other are considered adjacent (optional).

    Returns:
        An adjacent matrix with shape [num_segments, num_segments].
    """

    def _adjacency_euclidean_distance(segmentation):
        graph = RAG(segmentation, connectivity=connectivity)

        # Initialize the node's data for computing their centroids.
        for n in graph:
            graph.node[n].update({'count': 0,
                                  'centroid': np.zeros((2), dtype=np.float32)})

        # Run through each segmentation pixel and add the pixel's coordinates
        # to the centroid data.
        for index in np.ndindex(segmentation.shape):
            current = segmentation[index]
            graph.node[current]['count'] += 1
            graph.node[current]['centroid'] += index

        # Centroid is the sum of all pixel coordinates / pixel count.
        for n in graph:
            graph.node[n]['centroid'] = (graph.node[n]['centroid'] /
                                         graph.node[n]['count'])

        # Run through each edge and calculate the euclidian distance based on
        # the two node's centroids.
        for n1, n2, d in graph.edges_iter(data=True):
            diff = graph.node[n1]['centroid'] - graph.node[n2]['centroid']
            d['weight'] = np.linalg.norm(diff)

        # Return graph as adjacency matrix.
        return nx.to_numpy_matrix(graph, dtype=np.float32)

    return tf.py_func(
        _adjacency_euclidean_distance, [segmentation], tf.float32,
        stateful=False, name='adjacency_euclid_distance')

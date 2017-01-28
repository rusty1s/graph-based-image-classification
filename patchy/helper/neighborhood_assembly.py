import tensorflow as tf
import networkx as nx
import numpy as np


def neighborhoods_weights_to_root(adjacency, sequence, size):
    def _neighborhoods_weights_to_root(adjacency, sequence):
        graph = nx.from_numpy_matrix(adjacency)

        neighborhoods = np.zeros((sequence.shape[0], size), dtype=np.int32)
        neighborhoods.fill(-1)

        for i, n in enumerate(sequence):
            if n < 0:
                continue

            shortest = nx.single_source_dijkstra_path_length(graph, n).items()
            shortest = sorted(shortest, key=lambda v: v[1])
            shortest = shortest[:size]

            for j, k in enumerate(shortest):
                neighborhoods[i][j] = k[0]

        return neighborhoods

    return tf.py_func(_neighborhoods_weights_to_root, [adjacency, sequence],
                      tf.int32, stateful=False,
                      name='neighborhoods_weights_to_root')


def neighborhoods_nearest_scanline(adjacency, sequence, size):
    with tf.name_scope('neighborhoods_nearest_scanline',
                       values=[adjacency, sequence, size]):

        neighborhoods = neighborhoods_weights_to_root(
            adjacency, sequence, size)

        def _sort(array):
            return np.sort(array)

        return tf.py_func(_sort, [neighborhoodse], tf.int32, stateful=False,
                          name='sort')


neighborhood_assemblies = {'weights_to_root': neighborhoods_weights_to_root,
                           'nearest_scanline': neighborhoods_nearest_scanline}

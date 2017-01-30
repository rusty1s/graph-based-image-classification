from datetime import datetime

from six.moves import xrange
import tensorflow as tf
import networkx as nx
import numpy as np


def neighborhoods_weights_to_root(adjacency, sequence, size):
    def _neighborhoods_weights_to_root(adjacency, sequence):
        graph = nx.from_numpy_matrix(adjacency)

        neighborhoods = np.zeros((sequence.shape[0], size), dtype=np.int32)
        neighborhoods.fill(-1)

        for i in xrange(0, sequence.shape[0]):
            n = sequence[i]
            if n < 0:
                break

            shortest = nx.single_source_dijkstra_path_length(graph, n).items()
            shortest = sorted(shortest, key=lambda v: v[1])
            shortest = shortest[:size]

            for j in xrange(0, min(size, len(shortest))):
                neighborhoods[i][j] = shortest[j][0]

        return neighborhoods

    return tf.py_func(_neighborhoods_weights_to_root, [adjacency, sequence],
                      tf.int32, stateful=False,
                      name='neighborhoods_weights_to_root')


def neighborhoods_grid_spiral(adjacency, sequence, size):
    def _neighborhoods_grid_spiral(adjacency, sequence):
        graph = nx.from_numpy_matrix(adjacency)

        neighborhoods = np.zeros((sequence.shape[0], size), dtype=np.int32)
        neighborhoods.fill(-1)

        # Note: This method just works properly on planar graphs where nodes
        # are placed in a grid like layout and are weighted by distance.
        #
        # Add root to arr => [root]
        # Find nearest neighbor x to root
        # Add x => arr = [root, x]
        # Find nearest neighbor y with n(x, y) and min w(x,y) + w(root, y)
        # that is not already in arr.
        # set x = y
        # repeat until arr.length == size

        for i in xrange(0, sequence.shape[0]):
            root = sequence[i]
            if root < 0:
                break

            # Add root node to the beginning of the neighborhood.
            neighborhoods[i][0] = root
            x = root

            ws = nx.single_source_dijkstra_path_length(graph, root)
            ws = list(ws.items())

            for j in xrange(1, size):
                if x == -1:
                    break

                y = -1
                weight = float('inf')
                for _, n, d, in graph.edges_iter(x, data=True):
                    if n in neighborhoods[i]:
                        continue

                    w = ws[n][1] + d['weight']
                    if w < weight:
                        y = n
                        weight = w

                neighborhoods[i][j] = y
                x = y

        return neighborhoods

    return tf.py_func(_neighborhoods_grid_spiral, [adjacency, sequence],
                      tf.int32, stateful=False,
                      name='neighborhoods_grid_spiral')


neighborhood_assemblies = {'weights_to_root': neighborhoods_weights_to_root,
                           'grid_spiral': neighborhoods_grid_spiral}

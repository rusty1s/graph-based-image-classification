import tensorflow as tf
import networkx as nx
import numpy as np


def neighborhoods_weights_to_root(adjacency, sequence, size):
    def _neighborhoods_weights_to_root(adjacency, sequence):
        graph = nx.from_numpy_matrix(adjacency)

        neighborhoods = np.zeros((sequence.shape[0], size), dtype=np.int32)
        neighborhoods.fill(-1)

        for i, n in enumerate(sequence):
            # Pass if we iterate over an invalid node.
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

        for i, root in enumerate(sequence):
            # Pass if we iterate over an invalid node.
            if root < 0:
                continue

            # Add root node to the beginning of the neighborhood.
            neighborhoods[i][0] = root

            # Find the nearest neighbor x to root.
            x = -1
            weight = float('inf')
            for _, n, d in graph.edges_iter(nbunch=[root], data=True):
                if d['weight'] < weight:
                    x = n
                    weight = d['weight']

            neighborhoods[i][1] = x

            for j in range(2, size):
                if x == -1:
                    pass

                y = -1
                weight = float('inf')
                for _, n, d, in graph.edges_iter(nbunch=[x], data=True):
                    length = nx.shortest_path_length(graph, root, n, 'weight')
                    if d['weight'] + length < weight and\
                       n not in neighborhoods[i]:
                        y = n
                        weight = d['weight'] + length

                neighborhoods[i][j] = y
                x = y

        return neighborhoods

    return tf.py_func(_neighborhoods_grid_spiral, [adjacency, sequence],
                      tf.int32, stateful=False,
                      name='neighborhoods_grid_spiral')


neighborhood_assemblies = {'weights_to_root': neighborhoods_weights_to_root,
                           'grid_spiral': neighborhoods_grid_spiral}

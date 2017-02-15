from __future__ import division

import numpy as np
import scipy.sparse as sp

adjacency = np.array([
                      [[0, 0],   [-3, -1], [-1, 2], [2, 2], [4, 0], [1, -3]],
                      [[3, 1],   [0, 0],   [0, 0],  [0, 0], [0, 0], [0, 0]],
                      [[1, -2],  [0, 0],   [0, 0],  [0, 0], [0, 0], [0, 0]],
                      [[-2, -2], [0, 0],   [0, 0],  [0, 0], [0, 0], [0, 0]],
                      [[-4, 0],  [0, 0],   [0, 0],  [0, 0], [0, 0], [0, 0]],
                      [[-1, 3],  [0, 0],   [0, 0],  [0, 0], [0, 0], [0, 0]],
                     ], np.float32)

adjacency_single = np.array([
                             [0, 3, 1, 2, 4, 1, 0],
                             [3, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [2, 0, 0, 0, 0, 0, 0],
                             [4, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                            ], np.float32)


# Central question: Do we need to sigma our inputs if we normalize them anyway?

def scale_invariant_adj(adj):
    # We normalize our graph by making it scale invariant, so that the weight
    # of a neighbors node is max one. Note that this transforms an undirected
    # graph to an directed graph.
    r = adj.max(1).toarray()

    with np.errstate(divide='ignore'):
        r_inv = (1 / r).flatten()
        r_inv[np.isinf(r_inv)] = 0  # Correct x/0 results.

    return sp.diags(r_inv).dot(adj)


def add_self_loops(adj, value=1.0):
    return adj + value * sp.eye(adj.shape[0])


def partition_adj(distance_adj, angle_adj, num_partitions=1):
    pass


def invert_adj(adj, sigma=1.0):
    return adj


def preprocess_adj(distance_adj, angle_adj, sigma=1.0, num_partitions=1):
    # Preprocess distance adjacency.
    adj = scale_invariant_adj(distance_adj)
    adj = invert_adj(adj, sigma)
    adj = add_self_loop(adj, gaussian(0, sigma))

    # Derive partitions.
    return partition_adj(adj, angle_adj, num_partitions)


def gaussian(x, sigma=1.0):
    sigma = sigma * sigma
    return 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-x * x / (2 * sigma))


def normalize_adj(adj):
    d = np.array(adj.sum(1))

    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0  # Correct x/0 results.

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


# print(add_self_loops(sp.coo_matrix(adjacency_single), 10).toarray())
# print(normalize_adj(sp.coo_matrix(adjacency_single)).toarray())
# print(scale_invariant_adj(sp.coo_matrix(adjacency_single)))

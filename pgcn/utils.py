from __future__ import division

import numpy as np
import scipy.sparse as sp


adj = np.array([
                [0, 3, 1, 4, 1, 1, 1],
                [3, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
               ], np.float32)


def _scale_invariant_adj(adj):
    # We normalize our graph by making it scale invariant, so that the weight
    # of a neighbors node is max one. Note that this transforms an undirected
    # graph to a directed graph.
    r = adj.max(1).toarray()

    with np.errstate(divide='ignore'):
        r_inv = (1 / r).flatten()
        r_inv[np.isinf(r_inv)] = 0  # Correct x/0 results.

    return sp.diags(r_inv).dot(adj)


def _partition_adj(distance_adj, rad_adj, num_partitions=1):
    max_rad = 2 * np.pi
    i = max_rad / num_partitions

    # What does this op need todo?
    # For every node, we inspect its angle_adj.
    return distance_adj


def preprocess_adj(distance_adj, rad_adj, sigma=1.0, num_partitions=1):
    # Preprocess distance adjacency.
    adj = _scale_invariant_adj(distance_adj)

    # Apply gaussian elementwise.
    coef = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    adj.data = coef * np.exp(- adj.data ** 2 / (2 * sigma ** 2))

    # Add self loops with value of gaussian(0).
    adj = adj + coef * sp.eye(adj.shape[0])

    # Derive partitions.
    return _partition_adj(adj, rad_adj, num_partitions)


def normalize_adj(adj):
    d = np.array(adj.sum(1))

    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0  # Correct x/0 results.

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


# print(gaussian(0))
# print(add_self_loops(sp.coo_matrix(adj), 10).toarray())
print(preprocess_adj(sp.coo_matrix(adj), None).toarray())
# print(scale_invariant_adj(sp.coo_matrix(adjacency_single)))

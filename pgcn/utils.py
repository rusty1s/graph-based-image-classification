from __future__ import division

import numpy as np
import scipy.sparse as sp


def _scale_invariant_adj(adj):
    # We normalize our graph by making it scale invariant, so that the weight
    # of a neighbors node is max one. Note that this transforms an undirected
    # graph to a directed graph.

    with np.errstate(divide='ignore'):
        # Calculate max fraction matrix R with R_ii = 1 / max(A_ij).
        r = adj.max(1).toarray()
        r_inv = (1 / r).flatten()
        r_inv[np.isinf(r_inv)] = 0  # Correct x/0 results.
        r_mat_inv = sp.diags(r_inv)

    # Calculate R * A.
    return r_mat_inv.dot(adj)


def _partition_adj(adj_dist, adj_rad, num=1, start_deg=0.0):
    # TODO: Efficiency, maybe reduce adj_dist too.
    shape = (num * adj_dist.shape[0], adj_dist.shape[1])
    adjs = sp.coo_matrix(shape)

    max_rad = 2 * np.pi
    i = max_rad / num
    act_rad = max_rad - i + (start_deg % i)

    def _extract_greater(value, adj_dist, adj_rad):
        mask = adj_rad > value
        adj = adj_dist.multiply(mask)
        return adj, adj_dist - adj, adj_rad - adj_rad.multiply(mask)

    def _add_adj(adj, adjs, index):
        adj = adj.tocoo()
        row = adj.row + index * adj.shape[0]
        return adjs + sp.coo_matrix((adj.data, (row, adj.col)), shape)

    # We need to run backwards to efficiently extract values > 0 from the
    # adjacency matrix.
    for j in reversed(range(num)):
        adj, adj_dist, adj_rad = _extract_greater(act_rad, adj_dist, adj_rad)
        adjs = _add_adj(adj, adjs, j)
        act_rad -= i

    # If the start radian is greater than 0, we need to collect the last nodes
    # and add them to the last adjacency matrix in the adjacency collection.
    if adj_rad.count_nonzero() > 0:
        adj, _ = _extract_greater(0, adj_dist, adj_rad)
        adjs = _add_adj(adj, adjs, num-1)

    return adjs


def preprocess_adj(adj_dist, adj_rad, sigma=1.0, num_partitions=2,
                   start_angle=0.0):
    # Preprocess distance adjacency.
    adj = _scale_invariant_adj(adj_dist)

    # Apply gaussian elementwise.
    coef = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    adj.data = coef * np.exp(- adj.data ** 2 / (2 * sigma ** 2))

    # Add self loops with value of gaussian(0).
    # adj = adj + coef * sp.eye(adj.shape[0])

    # Derive partitions.
    return _partition_adj(adj, adj_rad, num_partitions)


def normalize_adj(adj):
    with np.errstate(divide='ignore'):
        # Calculate D^(-1/2).
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0  # Correct x/0 results.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Calculate D^(-1/2) * A * D^(-1/2).
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

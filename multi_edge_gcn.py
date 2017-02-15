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
                             [0, 4, 2, 2, 2, 2],
                             [3, 0, 0, 0, 0, 0],
                             [-1, 0, 0, 0, 0, 0],
                             [-2, 0, 0, 0, 0, 0],
                             [-4, 0, 0, 0, 0, 0],
                             [-1, 0, 0, 0, 0, 0],
                            ], np.float32)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adjacency_single.sum(1))
    print(rowsum)
    d_inv_sqrt = np.power(rowsum, -1)
    print(d_inv_sqrt)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    print(d_mat_inv_sqrt.todense())
    return d_mat_inv_sqrt.dot(adj).tocoo()


print(normalize_adj(adjacency_single).todense())

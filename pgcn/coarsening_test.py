from unittest import TestCase

import numpy as np
from numpy.testing import *
import scipy.sparse as sp

from .coarsening import coarsen


class UtilsTest(TestCase):

    def test_coarsening(self):
        adj = [[0, 1, 2, 0],
               [1, 0, 0, 3],
               [2, 0, 0, 1],
               [0, 3, 1, 0]]

        adj = sp.coo_matrix(np.array(adj))

        graphs, perms = coarsen(adj, levels=1)
        print(graphs[1].toarray())

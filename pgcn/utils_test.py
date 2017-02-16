from unittest import TestCase

import numpy as np
from numpy.testing import *
import scipy.sparse as sp

from .utils import (scale_invariant_adj,
                    self_loop_adj,
                    normalize_adj,
                    partition_adj,
                    gaussian)


class UtilsTest(TestCase):

    def test_scale_invariant_adj(self):
        adj = [[0.0, 1.0, 0.0],
               [1.0, 0.0, 2.0],
               [0.0, 0.5, 0.0]]

        expected = [[0.0, 1.0, 0.0],
                    [0.5, 0.0, 1.0],
                    [0.0, 1.0, 0.0]]

        adj = scale_invariant_adj(sp.coo_matrix(np.array(adj)))
        assert_equal(adj.toarray(), expected)

    def test_self_loop_adj(self):
        adj = [[0.0, 1.0, 0.0],
               [1.0, 0.0, 1.0],
               [0.0, 1.0, 0.0]]

        expected = [[0.5, 1.0, 0.0],
                    [1.0, 0.5, 1.0],
                    [0.0, 1.0, 0.5]]

        adj = self_loop_adj(sp.coo_matrix(np.array(adj)), value=0.5)
        assert_equal(adj.toarray(), expected)

    def test_normalize_adj(self):
        adj = np.array([[0.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0]])

        d_sqrt = np.array([[1.0, 0.0, 0.0],
                           [0.0, 2.0 ** -0.5, 0.0],
                           [0.0, 0.0, 1.0]])

        assert_equal(normalize_adj(sp.coo_matrix(adj)).toarray(),
                     d_sqrt.dot(adj).dot(d_sqrt))

    def test_partition_adj(self):
        pass

    def test_gaussian(self):
        self.assertEqual(gaussian(0), 0.3989422804014327)
        self.assertEqual(gaussian(1), 0.24197072451914337)
        self.assertEqual(gaussian(1, sigma=2.0), 0.17603266338214976)
        self.assertEqual(gaussian(1), gaussian(-1))

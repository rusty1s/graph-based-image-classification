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

        d_sqrt = np.array([[1.0, 0.0,         0.0],
                           [0.0, 2.0 ** -0.5, 0.0],
                           [0.0, 0.0,         1.0]])

        assert_equal(normalize_adj(sp.coo_matrix(adj)).toarray(),
                     d_sqrt.dot(adj).dot(d_sqrt))

    def test_partition_adj(self):
        adj_dist = ([[0.0, 1.0, 2.0],
                     [3.0, 0.0, 4.0],
                     [5.0, 6.0, 0.0]])

        adj_deg = ([[0.0,   45.0,  90.0],
                    [225.0, 0.0,   135.0],
                    [270.0, 315.0, 0.0]])

        adjs = partition_adj(sp.coo_matrix(np.array(adj_dist)),
                             sp.coo_matrix(np.array(adj_deg)).deg2rad())

        assert_equal(adjs.toarray(), adj_dist)

        adjs = partition_adj(sp.coo_matrix(np.array(adj_dist)),
                             sp.coo_matrix(np.array(adj_deg)).deg2rad(),
                             num=4)

        assert_equal(adjs.toarray(), [[0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 2.0],
                                      [0.0, 0.0, 4.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [3.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [5.0, 6.0, 0.0]])

        adjs = partition_adj(sp.coo_matrix(np.array(adj_dist)),
                             sp.coo_matrix(np.array(adj_deg)).deg2rad(),
                             num=4, start_rad=np.pi * 0.25)

        assert_equal(adjs.toarray(), [[0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 4.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [3.0, 0.0, 0.0],
                                      [5.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 6.0, 0.0]])

    def test_gaussian(self):
        self.assertEqual(round(gaussian(0), 4), 0.3989)
        self.assertEqual(round(gaussian(1), 4), 0.242)
        self.assertEqual(round(gaussian(1, sigma=2.0), 4), 0.176)
        self.assertEqual(gaussian(1), gaussian(-1))

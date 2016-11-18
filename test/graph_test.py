from __future__ import print_function

from nose.tools import *
from pynauty import Graph


def test_graph():
    g = Graph(2)
    g.connect_vertex(0, [1])
    g.connect_vertex(1, [0])

    assert_equal(g.number_of_vertices, 2)
    assert_equal(g.directed, False)
    assert_equal(g.adjacency_dict[0], [1])
    assert_equal(g.adjacency_dict[1], [0])
    assert_equal(g.vertex_coloring, [])

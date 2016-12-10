from nose.tools import *

import cv2
import networkx as nx

from .labeling import (betweenness_centrality, order)


def test_betweenness_centrality():
    # Create test graph (https://www.youtube.com/watch?v=UNDWoKE9s1w).
    graph = nx.Graph()

    graph.add_node('Ben')
    graph.add_node('Anna')
    graph.add_node('Cara')
    graph.add_node('Dana')
    graph.add_node('Evan')
    graph.add_node('Frank')

    graph.add_edge('Ben', 'Anna')
    graph.add_edge('Anna', 'Cara')
    graph.add_edge('Cara', 'Dana')
    graph.add_edge('Cara', 'Evan')
    graph.add_edge('Dana', 'Evan')
    graph.add_edge('Dana', 'Frank')
    graph.add_edge('Evan', 'Frank')

    labeling = betweenness_centrality(graph)

    # => ['Cara', 'Anna', { 'Evan', 'Dana' }, { 'Ben', 'Frank' }]
    #
    # **Problem:**
    #
    # betweenness centrality labeling is not unique, if there are two nodes
    # with the same betweenness centrality value, that is Evan == Dana == 1.5
    # and Ben == Frank == 0.0.

    assert_equals(len(labeling), 6)
    assert_equals(labeling[0], 'Cara')
    assert_equals(labeling[1], 'Anna')
    assert_in(labeling[2], ['Evan', 'Dana'])
    assert_in(labeling[3], ['Evan', 'Dana'])
    assert_true(labeling[2] != labeling[3])
    assert_in(labeling[4], ['Ben', 'Frank'])
    assert_in(labeling[5], ['Ben', 'Frank'])
    assert_true(labeling[4] != labeling[5])


def test_order():
    graph = nx.Graph()

    graph.add_node('Ben', {'order': 4})
    graph.add_node('Anna', {'order': 1})
    graph.add_node('Cara', {'order': 0})
    graph.add_node('Dana', {'order': 3})
    graph.add_node('Evan', {'order': 2})
    graph.add_node('Frank', {'order': 5})

    labeling = order(graph)

    assert_equals(len(labeling), 6)
    assert_equals(labeling[0], 'Cara')
    assert_equals(labeling[1], 'Anna')
    assert_equals(labeling[2], 'Evan')
    assert_equals(labeling[3], 'Dana')
    assert_equals(labeling[4], 'Ben')
    assert_equals(labeling[5], 'Frank')

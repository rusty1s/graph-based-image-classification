from nose.tools import *

import cv2
import networkx as nx

from .labeling import (betweenness_centrality, order)

from superpixels import (SuperpixelGraph, extract_superpixels,
                         image_to_slic_zero)


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
    def node_mapping(superpixel):
        return {'order': superpixel.order}

    def edge_mapping(from_superpixel, to_superpixel):
        return {}

    image = cv2.imread('./superpixels/test.png')
    assert_is_not_none(image)

    superpixels = extract_superpixels(image, image_to_slic_zero(image, 4))

    graph = SuperpixelGraph(superpixels, node_mapping, edge_mapping)

    labeling = order(graph)

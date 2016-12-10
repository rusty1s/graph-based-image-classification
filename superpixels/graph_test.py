from nose.tools import *

import cv2

from .graph import SuperpixelGraph
from .extract import extract_superpixels
from .slic import image_to_slic_zero


def node_mapping(superpixel):
    return {
        'color': superpixel.mean,
        'order': superpixel.order,
    }


def edge_mapping(from_superpixel, to_superpixel):
    c1 = from_superpixel.absolute_center
    c2 = to_superpixel.absolute_center

    return {'weight': (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2}


def test_graph():
    image = cv2.imread('./superpixels/test.png')
    assert_is_not_none(image)

    superpixels = extract_superpixels(image, image_to_slic_zero(image, 4))

    graph = SuperpixelGraph(superpixels, node_mapping, edge_mapping)

    assert_false(graph.is_directed())

    assert_equals(graph.number_of_nodes(), 4)
    assert_equals(graph.number_of_edges(), 6)

    assert_equals(len(graph.node[0]), 2)
    assert_equals(graph.node[0]['color'], (0.0, 0.0, 255.0))
    assert_equals(graph.node[0]['order'], 0)
    assert_equals(len(graph.node[1]), 2)
    assert_equals(graph.node[1]['color'], (255.0, 0.0, 0.0))
    assert_equals(graph.node[1]['order'], 1)
    assert_equals(len(graph.node[2]), 2)
    assert_equals(graph.node[2]['color'], (0.0, 255.0, 0.0))
    assert_equals(graph.node[2]['order'], 2)
    assert_equals(len(graph.node[3]), 2)
    assert_equals(graph.node[3]['color'], (0.0, 0.0, 0.0))
    assert_equals(graph.node[3]['order'], 3)

    assert_equal(len(graph[0][1]), 1)
    assert_equal(graph[0][1]['weight'], 100.0)
    assert_equal(len(graph[0][2]), 1)
    assert_equal(graph[0][2]['weight'], 100.0)
    assert_equal(len(graph[1][3]), 1)
    assert_equal(graph[1][3]['weight'], 100.0)
    assert_equal(len(graph[2][3]), 1)
    assert_equal(graph[2][3]['weight'], 100.0)
    assert_equal(len(graph[0][3]), 1)
    assert_equal(graph[0][3]['weight'], 200.0)
    assert_equal(len(graph[1][2]), 1)
    assert_equal(graph[1][2]['weight'], 200.0)

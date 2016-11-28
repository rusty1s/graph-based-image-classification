from nose.tools import *
from superpixel import Segment
from graph import graph_from_segments

image = [
        [0,   10,  20,  30],
        [40,  50,  60,  70],
        [80,  90,  100, 110],
        [120, 130, 140, 150],
        ]

superpixels = [
        [1, 1, 2, 1],
        [1, 1, 1, 1],
        [3, 3, 3, 1],
        [3, 4, 4, 4],
        ]


def node_mapping(segment):
    return {'color': segment.mean}


def edge_mapping(from_segment, to_segment):
    c1 = from_segment.center
    c2 = to_segment.center

    return {'weight': (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2}


def test_graph():
    segments = Segment.generate(image, superpixels)

    G = graph_from_segments(segments, node_mapping, edge_mapping)

    assert_false(G.is_directed())

    assert_equal(G.number_of_nodes(), 4)
    assert_equal(G.number_of_edges(), 4)

    assert_equal(len(G.node[1]), 1)
    assert_equal(G.node[1]['color'][0], 46.25)
    assert_equal(len(G.node[2]), 1)
    assert_equal(G.node[2]['color'][0], 20)
    assert_equal(len(G.node[3]), 1)
    assert_equal(G.node[3]['color'][0], 97.5)
    assert_equal(len(G.node[4]), 1)
    assert_equal(G.node[4]['color'][0], 140)

    assert_equal(len(G[1][2]), 1)
    assert_equal(G[1][2]['weight'], 0.375**2 + 0.75**2)
    assert_equal(len(G[1][3]), 1)
    assert_equal(G[1][3]['weight'], 0.875**2 + 1.5**2)
    assert_equal(len(G[1][4]), 1)
    assert_equal(G[1][4]['weight'], 0.375**2 + 2.25**2)
    assert_equal(len(G[3][4]), 1)
    assert_equal(G[3][4]['weight'], 1.25**2 + 0.75**2)

from nose.tools import *

import networkx as nx

from .normalization import normalize
from .neighborhood_assembly import assemble_neighborhood
from .labeling import order


def test_neighborhood_assembly():
    # Create test graph with custom ordering.
    graph = nx.Graph()

    graph.add_node('Ben', {'order': 4})
    graph.add_node('Anna', {'order': 1})
    graph.add_node('Cara', {'order': 0})
    graph.add_node('Dana', {'order': 3})
    graph.add_node('Evan', {'order': 2})
    graph.add_node('Frank', {'order': 5})

    graph.add_edge('Ben', 'Anna')
    graph.add_edge('Anna', 'Cara')
    graph.add_edge('Cara', 'Dana')
    graph.add_edge('Cara', 'Evan')
    graph.add_edge('Dana', 'Evan')
    graph.add_edge('Dana', 'Frank')
    graph.add_edge('Evan', 'Frank')

    neighborhood = assemble_neighborhood(graph, 'Cara', 6)

    normalization = normalize(graph, neighborhood, 'Cara', order, 2)

    assert_equals(len(normalization), 2)
    assert_equals(normalization[0], 'Cara')
    assert_equals(normalization[1], 'Anna')

    normalization = normalize(graph, neighborhood, 'Cara', order, 7)

    assert_equals(len(normalization), 7)
    assert_equals(normalization[0], 'Cara')
    assert_equals(normalization[1], 'Anna')
    assert_equals(normalization[2], 'Evan')
    assert_equals(normalization[3], 'Dana')
    assert_equals(normalization[4], 'Ben')
    assert_equals(normalization[5], 'Frank')
    assert_is_none(normalization[6])

    neighborhood = assemble_neighborhood(graph, 'Ben', 4)

    normalization = normalize(graph, neighborhood, 'Ben', order, 4)

    assert_equals(len(normalization), 4)
    assert_equals(normalization[0], 'Ben')
    assert_equals(normalization[1], 'Anna')
    assert_equals(normalization[2], 'Cara')
    assert_equals(normalization[3], 'Evan')

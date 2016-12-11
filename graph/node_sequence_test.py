from nose.tools import *

import networkx as nx

from .node_sequence import node_sequence
from .labeling import order


def test_node_sequence():
    # Create test graph with custom ordering.
    graph = nx.Graph()

    graph.add_node('Ben', {'order': 4})
    graph.add_node('Anna', {'order': 1})
    graph.add_node('Cara', {'order': 0})
    graph.add_node('Dana', {'order': 3})
    graph.add_node('Evan', {'order': 2})
    graph.add_node('Frank', {'order': 5})

    sequence = node_sequence(graph, order, 2, 2)

    assert_equals(len(sequence), 2)
    assert_equals(sequence[0], 'Cara')
    assert_equals(sequence[1], 'Evan')

    sequence = node_sequence(graph, order, 1, 4)

    assert_equals(len(sequence), 4)
    assert_equals(sequence[0], 'Cara')
    assert_equals(sequence[1], 'Anna')
    assert_equals(sequence[2], 'Evan')
    assert_equals(sequence[3], 'Dana')

    sequence = node_sequence(graph, order, 3, 5)

    assert_equals(len(sequence), 5)
    assert_equals(sequence[0], 'Cara')
    assert_equals(sequence[1], 'Dana')
    assert_is_none(sequence[2])
    assert_is_none(sequence[3])
    assert_is_none(sequence[4])

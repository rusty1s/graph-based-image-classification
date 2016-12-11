from nose.tools import *

import networkx as nx

from .node_sequence import node_sequence
from .labeling import order

# Create test graph (https://www.youtube.com/watch?v=UNDWoKE9s1w).
graph = nx.Graph()

graph.add_node('Ben', {'order': 4})
graph.add_node('Anna', {'order': 1})
graph.add_node('Cara', {'order': 0})
graph.add_node('Dana', {'order': 3})
graph.add_node('Evan', {'order': 2})
graph.add_node('Frank', {'order': 5})


def test_node_sequence():
    sequence = node_sequence(order, graph, 2, 2)

    assert_equals(len(sequence), 2)
    assert_equals(sequence[0], 'Cara')
    assert_equals(sequence[1], 'Evan')

    sequence = node_sequence(order, graph, 1, 4)

    assert_equals(len(sequence), 4)
    assert_equals(sequence[0], 'Cara')
    assert_equals(sequence[1], 'Anna')
    assert_equals(sequence[2], 'Evan')
    assert_equals(sequence[3], 'Dana')

    sequence = node_sequence(order, graph, 3, 5)

    assert_equals(len(sequence), 5)
    assert_equals(sequence[0], 'Cara')
    assert_equals(sequence[1], 'Dana')
    assert_is_none(sequence[2])
    assert_is_none(sequence[3])
    assert_is_none(sequence[4])

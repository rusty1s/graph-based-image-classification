from nose.tools import *

import networkx as nx

from .node_sequence import node_sequence
from .labeling import betweenness_centrality

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

# Betweenness centrality:
# => ['Cara', 'Anna', { 'Evan', 'Dana' }, { 'Ben', 'Frank' }]


def test_node_sequence():
    sequence = node_sequence(betweenness_centrality, graph, 2, 2)

    assert_equals(len(sequence), 2)
    assert_equals(sequence[0], 'Cara')
    assert_in(sequence[1], ['Evan', 'Dana'])

    sequence = node_sequence(betweenness_centrality, graph, 1, 4)

    assert_equals(len(sequence), 4)
    assert_equals(sequence[0], 'Cara')
    assert_equals(sequence[1], 'Anna')
    assert_in(sequence[2], ['Evan', 'Dana'])
    assert_in(sequence[3], ['Evan', 'Dana'])
    assert_true(sequence[2] != sequence[3])

    sequence = node_sequence(betweenness_centrality, graph, 3, 5)

    assert_equals(len(sequence), 5)
    assert_equals(sequence[0], 'Cara')
    assert_in(sequence[1], ['Evan', 'Dana'])
    assert_is_none(sequence[2])
    assert_is_none(sequence[3])
    assert_is_none(sequence[4])

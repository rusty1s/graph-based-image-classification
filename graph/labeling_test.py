from nose.tools import *

import networkx as nx

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


def test_betweenness_centrality():
    labeling = betweenness_centrality(graph)

    assert_equals(len(labeling), 6)
    assert_equals(labeling[0], 'Cara')
    assert_equals(labeling[1], 'Anna')
    assert_equals(labeling[2], 'Evan')
    assert_equals(labeling[3], 'Dana')
    assert_equals(labeling[4], 'Ben')
    assert_equals(labeling[5], 'Frank')

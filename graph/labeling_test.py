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
# Betweenness centrality:

    # Problem betweenness centrality labeling is not unique, if there are two
    # nodes with the same betweenness centrality value, that is Evan == Dana ==
    # 1.5 and Ben == Frank == 0.0.
    #
    # => ['Cara', 'Anna', { 'Evan', 'Dana' }, { 'Ben', 'Frank' }]
    assert_equals(len(labeling), 6)
    assert_equals(labeling[0], 'Cara')
    assert_equals(labeling[1], 'Anna')
    assert_in(labeling[2], ['Evan', 'Dana'])
    assert_in(labeling[3], ['Evan', 'Dana'])
    assert_true(labeling[2] != labeling[3])
    assert_in(labeling[4], ['Ben', 'Frank'])
    assert_in(labeling[5], ['Ben', 'Frank'])
    assert_true(labeling[4] != labeling[5])

from nose.tools import *

import networkx as nx

from .neighborhood_assembly import neighborhood_assembly


def test_neighborhood_assembly():
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

    neighborhood = neighborhood_assembly(graph, 'Cara', 0)

    assert_equals(len(neighborhood), 1)
    assert_in('Cara', neighborhood)

    neighborhood = neighborhood_assembly(graph, 'Cara', 1)

    assert_equals(len(neighborhood), 1)
    assert_in('Cara', neighborhood)

    neighborhood = neighborhood_assembly(graph, 'Cara', 2)

    assert_equals(len(neighborhood), 4)
    assert_in('Cara', neighborhood)
    assert_in('Anna', neighborhood)
    assert_in('Dana', neighborhood)
    assert_in('Evan', neighborhood)

    neighborhood = neighborhood_assembly(graph, 'Ben', 2)

    assert_equals(len(neighborhood), 2)
    assert_in('Ben', neighborhood)
    assert_in('Anna', neighborhood)

    neighborhood = neighborhood_assembly(graph, 'Ben', 3)

    assert_equals(len(neighborhood), 3)
    assert_in('Ben', neighborhood)
    assert_in('Anna', neighborhood)
    assert_in('Cara', neighborhood)

    neighborhood = neighborhood_assembly(graph, 'Ben', 4)

    assert_equals(len(neighborhood), 5)
    assert_in('Ben', neighborhood)
    assert_in('Anna', neighborhood)
    assert_in('Cara', neighborhood)
    assert_in('Dana', neighborhood)
    assert_in('Evan', neighborhood)

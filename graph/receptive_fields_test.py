from nose.tools import *
from numpy import testing as np_test

import networkx as nx

from .receptive_fields import receptive_fields
from .labeling import order


def test_receptive_fields():
    # Create test graph with custom ordering.
    graph = nx.Graph()

    graph.add_node('Ben', {'order': 4, 'red': 1.0, 'green': 2.0, 'blue': 3.0})
    graph.add_node('Anna', {'order': 1, 'red': 4.0, 'green': 5.0, 'blue': 6.0})
    graph.add_node('Cara', {'order': 0, 'red': 7.0, 'green': 8.0, 'blue': 9.0})
    graph.add_node('Dana', {'order': 3, 'red': 10.0, 'green': 11.0,
                            'blue': 12.0})
    graph.add_node('Evan', {'order': 2, 'red': 13.0, 'green': 14.0,
                            'blue': 15.0})
    graph.add_node('Frank', {'order': 5, 'red': 16.0, 'green': 17.0,
                             'blue': 18.0})

    graph.add_edge('Ben', 'Anna')
    graph.add_edge('Anna', 'Cara')
    graph.add_edge('Cara', 'Dana')
    graph.add_edge('Cara', 'Evan')
    graph.add_edge('Dana', 'Evan')
    graph.add_edge('Dana', 'Frank')
    graph.add_edge('Evan', 'Frank')

    def node_features(attributes):
        return [attributes['red'], attributes['green'], attributes['blue']]

    fields = receptive_fields(graph, order, 1, 6, 4, node_features, 3)

    assert_equals(fields.shape, (3, 6, 4))

    # The first neighborhood nodes on the second axis are centered around
    # `Cara`. The neighborhood of `Cara` is ['Cara', 'Anna', 'Evan', 'Dana'].
    # The feature vectors for `red` should be: [7.0, 4.0, 13.0, 10.0].
    np_test.assert_array_equal(fields[0][0], [7.0, 4.0, 13.0, 10.0])

    # The third neighborhood nodes on the second axis are centered around
    # `Evan`. The neighborhood of `Evan` is ['Evan', 'Cara', 'Dana', 'Frank'].
    # The feature vectors for `green` should be [14.0, 8.0, 11.0, 17.0].
    np_test.assert_array_equal(fields[1][2], [14.0, 8.0, 11.0, 17.0])

    # The second neighborhood nodes on the second axis are centered around
    # `Anna`. The neighborhood of `Anna` is ['Anna', 'Cara', 'Ben', 'Evan'].
    # The feature vectors for `blue` should be [6.0, 9.0, 3.0, 15.0].
    np_test.assert_array_equal(fields[2][1], [6.0, 9.0, 3.0, 15.0])

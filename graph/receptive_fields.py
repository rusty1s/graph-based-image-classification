import networkx as nx
import numpy as np

from .node_sequence import node_sequence
from .neighborhood_assembly import assemble_neighborhood
from .normalization import normalize


def receptive_fields(graph, labeling, stride, width, receptive_field_size,
                     node_features, node_features_size):

    # Create the initial receptive fields
    receptive_fields = np.zeros((node_features_size, width,
                                 receptive_field_size))

    # Select a fixed-length sequence of nodes from the graph. A node in `nodes`
    # can have the value `None` for padding purposes.
    nodes = node_sequence(graph, labeling, stride, width)

    for y, main_node in enumerate(nodes):
        if main_node is None:
            continue

        # Create a fixed-length sequence of neighborhood nodes. A node in
        # `normalization` can have the value `None` for padding purposes.
        neighborhood = assemble_neighborhood(graph, main_node,
                                             receptive_field_size)

        normalization = normalize(graph, neighborhood, main_node, labeling,
                                  receptive_field_size)

        for x, node in enumerate(normalization):
            if node is None:
                continue

            # Get the node_features.
            features = node_features(graph.node[node])

            for i, feature in enumerate(features):
                receptive_fields[i][y][x] = feature

    return receptive_fields

import networkx as nx


# Node sequence selection is the process of identifying, for each input graph,
# a sequence of nodes for which receptive fields are created.
#
# 1. The vertices of the input graph are sorted with respect to a given graph
#    labeling.
# 2. The resulting node sequence is traversed using a given stride and for each
#    visited node we construct a receptive field, until exactly `width`
#    receptive fields have been created. The stride determines the distance,
#    relative to the selected node sequence, between two consectutive nodes for
#    which a receptive field is created. If the number of nodes is smaller than
#    `width`, the algorithm creates all-zero receptive fields for padding
#    purposes.
def node_sequence(labeling, graph, stride, width):
    """Identifies and returns for a graph and a graph labeling the sequence of
    nodes for which a receptive fields are created."""

    if stride <= 0:
        raise Exception('Stride must be greater than zero.')

    sort = labeling(graph)
    filtered = [v for i, v in enumerate(sort) if i % stride == 0]

    if len(filtered) >= width:
        return filtered[0:width]
    else:
        filtered.extend([None for _ in range(0, width - len(filtered))])
        return filtered

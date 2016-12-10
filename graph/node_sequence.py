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
def select_node_sequence(labeling, graph, stride, width, receptive_field_size):

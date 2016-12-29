import tensorflow as tf

from converter import Converter


class PatchySan(Converter):

    def __init__(self, grapher, num_nodes, node_labeling, node_stride,
                 neighborhood_size, neighborhood_labeling, num_node_channels):

        if node_labeling not in labelings:
            raise ValueError(
                'Could not find labeling "{}".'.format(node_labeling))

        if neighborhood_labeling not in labelings:
            raise ValueError(
                'Could not find labeling "{}".'.format(neighborhood_labeling))

        self._grapher = grapher
        self._num_nodes = num_nodes
        self._node_labeling = node_labeling
        self._node_stride = node_stride
        self._neighborhood_size = neighborhood_size
        self._neighborhood_labeling = neighborhood_labeling
        self._num_node_channels = num_node_channels

    @property
    def shape(self):
        return [
            self._num_nodes,
            self._neighborhood_size,
            self._num_node_channels
        ]

    def convert(self, data):
        nodes, adjacent = self._grapher.create_graph(data)

        sequence = labelings[self._node_labeling](adjacent)
        sequence = node_sequence(sequence, self._num_nodes, self._node_stride)

        neighborhoods = neighborhood_assembly(
            sequence, adjacent, self._neighborhood_size,
            labelings[self._neighborhood_labeling])





        return data

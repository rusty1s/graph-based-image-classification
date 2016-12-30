import tensorflow as tf

from converter import Converter
from .patchy_san.labeling import labelings
from .patchy_san.node_sequence import node_sequence
from .patchy_san.neighborhood_assembly import neighborhoods_assembly
from .patchy_san.receptive_field import receptive_fields

from superpixel.algorithm import slico_generator


slic = slico_generator(120)


class PatchySan(Converter):

    def __init__(self, grapher, num_nodes, node_labeling, node_stride,
                 neighborhood_size, neighborhood_labeling):

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

    @property
    def shape(self):
        return [
            self._num_nodes,
            self._neighborhood_size,
            # self._grapher.node_channels_length,
            1
        ]

    def convert(self, data):
        # s = slic(data)
        # nodes = tf.strided_slice(s, [0, 0], [100, 8], [1, 1])
        # nodes = tf.cast(nodes, tf.float32)

        # # nodes = tf.zeros([100, 8])
        # adjacent = tf.zeros([100, 100])
        # return tf.pad(data, [[]])
        nodes, adjacent = self._grapher.create_graph(data)
        # nodes = tf.zeros([self._num_nodes])
        # result = tf.strided_slice(nodes, [0], [self._num_nodes], [1])
        # result = tf.map_fn(lambda x: tf.zeros([self._neighborhood_size, 8]), result)
        # return result

        sequence = labelings[self._node_labeling](adjacent)
        sequence = node_sequence(sequence, self._num_nodes, self._node_stride)

        neighborhoods = neighborhoods_assembly(
            sequence, adjacent, self._neighborhood_size,
            labelings[self._neighborhood_labeling])

        return receptive_fields(neighborhoods, nodes, 1)
                                # self._grapher.node_channels_length)

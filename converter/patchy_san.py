import tensorflow as tf
import networkx as nx
import numpy as np

from .converter import Converter


class PatchySan(Converter):

    def __init__(self, grapher, num_nodes, node_labeling, node_stride,
                 neighborhood_size, neighborhood_labeling, num_node_channels):

        # TODO throw errors if labelings not in labelings dict

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

    @property
    def params(self):
        return {
            'num_nodes': self._num_nodes,
            'node_labeling': self._node_labeling,
            'node_stride': self._node_stride,
            'neighborhood_size': self._neighborhood_size,
            'neighborhood_labeling': self._neighborhood_labeling,
            'num_node_channels': self._num_node_channels,
        }

    def convert(self, data):
        nodes, adjacent = self._grapher.create_graph(data)

        sequence = labelings[self._node_labeling](adjacent)
        sequence = node_sequence(sequence, self._num_nodes, self._node_stride)
        neighborhoods = assemble_neighborhoods(
            sequence, adjacent, self._neighborhood_size,
            labelings[self._neighborhood_labeling])

        # TODO receptive fields
        # this is relative easy, all we need to do is create another mapper
        # function that replaces itself with the corresponding node features or
        # with zero vectors if value is -1.

        return data


def betweenness_centrality(adjacent):
    def _betweenness_centrality(adjacent):
        graph = nx.Graph(adjacent)

        labeling = nx.betweenness_centrality(graph, normalized=False)
        labeling = list(labeling.items())
        labeling = sorted(labeling, key=lambda v: v[1], reverse=True)
        labeling = [v[0] for v in labeling]

        return np.array(labeling, dtype=np.float32)

    labeling = tf.py_func(_betweenness_centrality, [adjacent], tf.float32,
                          stateful=False, name='betweenness_centrality')
    return tf.cast(labeling, tf.int32)


labelings = {
    'betweenness_centrality': betweenness_centrality,
}


def node_sequence(sequence, width, stride):
    # Stride the sequence based on the given stride width.
    size = sequence.get_shape()[0].value
    sequence = tf.strided_slice(sequence, [0], [size], [stride])

    # No more entries than we want.
    sequence = tf.strided_slice(sequence, [0], [width], [1])

    # Pad with zeros if we need to.
    size = sequence.get_shape()[0].value

    if size < width:  # TODO if weg, muss auch ohne gehen
        sequence = tf.add(sequence, tf.ones_like(sequence))
        sequence = tf.pad(sequence, [[0, width-size]])
        sequence = tf.sub(sequence, tf.ones_like(sequence))

    return sequence


def assemble_neighborhoods(sequence, adjacent, size, labeling):
    assemble = assemble_neighborhood(size, adjacent)

    sequence = tf.reshape([-1, 1])
    neighborhoods = tf.map_fn(assemble, t)

    return neighborhoods


def assemble_neighborhood(index, adjacent, size, labeling):
    # this shall return a function assemble(index) with a fixed size
    # this is neighborhood assembly where we already compute a fixed size
    # neighborhood via labeling and distances
    def _assemble(index):
        return index

    return _assemble

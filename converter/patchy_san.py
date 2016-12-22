import tensorflow as tf
import networkx as nx
import numpy as np

from .converter import Converter

from patchy import (receptive_fields, betweenness_centrality, order)
from superpixels import (image_to_slic_zero, extract_superpixels,
                         create_superpixel_graph)


def convert_image_to_field(image):
    rep = image_to_slic_zero(image, 100)
    superpixels = extract_superpixels(image, rep)

    graph = create_superpixel_graph(superpixels, node_mapping, edge_mapping)
    fields = receptive_fields(graph, order, 1, 100, 10, betweenness_centrality,
                              node_features, 8)
    fields = fields.astype(np.float32)
    return fields


def node_mapping(superpixel):
    return superpixel.get_attributes()


def edge_mapping(from_superpixel, to_superpixel):
    return {}


def node_features(node_attributes):
    return [
        node_attributes['red'],
        node_attributes['green'],
        node_attributes['blue'],
        node_attributes['count'],
        node_attributes['y'],
        node_attributes['x'],
        node_attributes['height'],
        node_attributes['width'],
    ]


class PatchySan(Converter):

    def __init__(self, num_nodes, node_labeling, node_stride, num_neighborhood,
                 neighborhood_labeling, node_channels):

        # TODO throw errors if labelings not in labelings dict

        self._num_nodes = num_nodes
        self._node_labeling = node_labeling
        self._node_stride = node_stride
        self._num_neighborhood = num_neighborhood
        self._neighborhood_labeling = neighborhood_labeling
        self._node_channels = node_channels

    @property
    def shape(self):
        return [
            self._num_nodes,
            self._num_neighborhood,
            8,
            # len(self._node_channels),
        ]

    @property
    def params(self):
        return {
            'num_nodes': self._num_nodes,
            'node_labeling': self._node_labeling,
            'node_stride': self._node_stride,
            'num_neighborhood': self._num_neighborhood,
            'neighborhood_labeling': self._neighborhood_labeling,
            'node_channels': self._node_channels,
        }

    def convert(self, data):
        image = tf.cast(data, tf.int32)
        field = tf.py_func(convert_image_to_field, [image], tf.float32,
                           stateful=False, name='GRAPH')
        return field


def node_sequence(sequence, width, stride=1):
    # Stride the sequence based on the given stride width.
    sequence = tf.strided_slice(sequence, [0], [-1], [stride])

    # No more entries than we want.
    sequence = tf.strided_slice(sequence, [0], [width], [1])

    # Pad with zeros if we need to.
    size = sequence.get_shape()[0].value

    if size < width:  # TODO if weg, muss auch ohne gehen
        sequence = tf.pad(sequence, [[0, width-size]])

    return sequence


# def betweenness_centrality(graph):
#     result = nx.betweenness_centrality(graph, normalized=False)
#     result = list(result.items())
#     result = sorted(result, key=lambda v: v[1], reverse=True)
#     result = [v[0] for v in result]

#     return tf.constant(result)


# def order(graph):
#     result = nx.get_node_attributes(graph, 'order')
#     result = list(result.items())
#     result = sorted(result, key=lambda v: v[1])
#     result = [v[0] for v in result]

#     return tf.constant(result)

# labelings = {
#     'betweenness_centraliy': betweenness_centrality,
#     'order': order,
# }

import tensorflow as tf
import networkx as nx
import numpy as np

from .converter import Converter

# from patchy import (receptive_fields, betweenness_centrality, order)
# from superpixels import (image_to_slic_zero, extract_superpixels,
#                          create_superpixel_graph)
# from superpixel.algorithm import (slic, slico)


# def convert_image_to_field(image):
#     rep = image_to_slic_zero(image, 100)
#     superpixels = extract_superpixels(image, rep)

#     graph = create_superpixel_graph(superpixels, node_mapping, edge_mapping)
#    fields = receptive_fields(graph, order, 1, 100, 10,
#    betweenness_centrality,
#                               node_features, 8)
#     fields = fields.astype(np.float32)
#     return fields


# def node_mapping(superpixel):
#     return superpixel.get_attributes()


# def edge_mapping(from_superpixel, to_superpixel):
#     return {}


# def node_features(node_attributes):
#     return [
#         node_attributes['red'],
#         node_attributes['green'],
#         node_attributes['blue'],
#         node_attributes['count'],
#         node_attributes['y'],
#         node_attributes['x'],
#         node_attributes['height'],
#         node_attributes['width'],
#     ]


class PatchySan(Converter):

    def __init__(self, grapher, num_nodes, node_labeling, node_stride,
                 num_neighborhood, neighborhood_labeling, num_node_channels):

        # TODO throw errors if labelings not in labelings dict

        self._grapher = grapher
        self._num_nodes = num_nodes
        self._node_labeling = node_labeling
        self._node_stride = node_stride
        self._num_neighborhood = num_neighborhood
        self._neighborhood_labeling = neighborhood_labeling
        self._num_node_channels = num_node_channels

    @property
    def shape(self):
        return [
            self._num_nodes,
            self._num_neighborhood,
            self._num_node_channels
        ]

    @property
    def params(self):
        return {
            'num_nodes': self._num_nodes,
            'node_labeling': self._node_labeling,
            'node_stride': self._node_stride,
            'num_neighborhood': self._num_neighborhood,
            'neighborhood_labeling': self._neighborhood_labeling,
            'num_node_channels': self._num_node_channels,
        }

    def convert(self, image):
        graph = self._grapher.create_graph(image)

        sequence = labelings[self._node_labeling](graph)
        sequence = node_sequence(sequence, self._num_nodes, self._node_stride)
        # s = slico(image, 100)
        # TODO to graph
        # TODO to data
        return image
        # s = tf.reshape(s, [24, 24, 1])
        # s = tf.cast(s, tf.float32)
        # return s

        # image = tf.cast(data, tf.int32)
        # field = tf.py_func(convert_image_to_field, [image], tf.float32,
        #                    stateful=False, name='GRAPH')
        # return field


def node_sequence(sequence, width, stride):
    # Stride the sequence based on the given stride width.
    size = sequence.get_shape()[0].value
    sequence = tf.strided_slice(sequence, [0], [size], [stride])

    # No more entries than we want.
    sequence = tf.strided_slice(sequence, [0], [width], [1])

    # Pad with zeros if we need to.
    size = sequence.get_shape()[0].value

    if size < width:  # TODO if weg, muss auch ohne gehen
        sequence = tf.pad(sequence, [[0, width-size]])

    return sequence


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

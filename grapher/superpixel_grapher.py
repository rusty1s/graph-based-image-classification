import tensorflow as tf
import networkx as nx
import numpy as np

from superpixel import (Superpixel, extract)

from .grapher import Grapher


def _node_channels(superpixel):
    color = superpixel.mean
    center = superpixel.relative_center_in_bounding_box

    return [
        color[0],
        color[1],
        color[2],
        superpixel.count,
        center[1],
        center[0],
        superpixel.height,
        superpixel.width,
    ]


class SuperpixelGrapher(Grapher):

    def __init__(self, superpixel_generator):
        self._superpixel_generator = superpixel_generator

    @property
    def node_channels_length(self):
        return len(_node_channels(Superpixel()))

    def create_graph(self, image):
        def _create_graph(image, superpixel_representation):
            superpixels = extract(image, superpixel_representation)

            graph = nx.Graph()

            def _node_mapping(superpixel):
                return {'features': _node_channels(superpixel)}

            for s in superpixels:
                graph.add_node(s.id, _node_mapping(s))

                for id in s.neighbors:
                    graph.add_edge(s.id, id)

            nodes = list(nx.get_node_attributes(graph, 'features').items())
            nodes = sorted(nodes, key=lambda v: v[0])
            nodes = [v[1] for v in nodes]
            nodes = np.array(nodes, dtype=np.float32)

            adjacent = nx.to_numpy_matrix(graph)
            adjacent = adjacent.astype(np.float32)

            return nodes, adjacent

        superpixel_representation = self._superpixel_generator(image)
        return tf.py_func(_create_graph, [image, superpixel_representation],
                          [tf.float32, tf.float32], stateful=False,
                          name='superpixel_grapher')

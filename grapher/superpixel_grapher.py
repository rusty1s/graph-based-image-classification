import tensorflow as tf
import networkx as nx
import numpy as np

from .grapher import Grapher
from .superpixel.extract import extract


class SuperpixelGrapher(Grapher):

    def __init__(self, superpixel_generator):
        self._superpixel_generator = superpixel_generator

    def create_graph(self, image):
        def _create_graph(image, superpixel_representation):
            superpixels = extract(image, superpixel_representation)

            graph = nx.Graph()

            def _node_mapping(superpixel):
                return {
                    'features': superpixel.features
                }

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

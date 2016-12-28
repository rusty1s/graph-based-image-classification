import tensorflow as tf
import networkx as nx
import numpy as np

from .extract import extract_superpixels
# returns a 1d tensor with all the infos of the segment
# returns an 2d tensor with the adjacent matrix


def create_graph(image, superpixels):
    def _create_graph(image, superpixels):
        superpixels = extract_superpixels(image, superpixels)

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

    return tf.py_func(_create_graph, [image, superpixels],
                      [tf.float32, tf.float32], stateful=False,
                      name='create_graph')

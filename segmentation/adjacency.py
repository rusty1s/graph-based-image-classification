import tensorflow as tf
import networkx as nx
import numpy as np
from skimage.future.graph import RAG


CONNECTIVITY = 2


def adjacency(segmentation, connectivity=CONNECTIVITY):
    def _adjacency(segmentation):
        graph = RAG(segmentation, connectivity=connectivity)
        return nx.to_numpy_matrix(graph, dtype=np.float32)

    return tf.py_func(
        _adjacency, [segmentation], tf.float32, stateful=False,
        name='adjacency')


def adjacency_euclid_distance(segmentation, connectivity=CONNECTIVITY):
    def _adjacency_euclid_distance(segmentation):
        graph = RAG(segmentation, connectivity=connectivity)

        for n in graph:
            graph.node[n].update({'count': 0,
                                  'centroid': np.zeros((2), dtype=np.float32)})

        for index in np.ndindex(segmentation.shape):
            current = segmentation[index]
            graph.node[current]['count'] += 1
            graph.node[current]['centroid'] += index

        for n in graph:
            graph.node[n]['centroid'] = (graph.node[n]['centroid'] /
                                         graph.node[n]['count'])

        for n1, n2, d in graph.edges_iter(data=True):
            diff = graph.node[n1]['centroid'] - graph.node[n2]['centroid']
            d['weight'] = np.linalg.norm(diff)

        return nx.to_numpy_matrix(graph, dtype=np.float32)

    return tf.py_func(
        _adjacency_euclid_distance, [segmentation], tf.float32,
        stateful=False, name='adjacency_euclid_distance')

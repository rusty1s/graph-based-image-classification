import tensorflow as tf
import networkx as nx
import numpy as np


def identity(adjacency, labels=None):
    return _labels_default(labels, adjacency)


def betweenness_centrality(adjacency, labels=None):
    def _betweeness_centrality(adjacency, labels):
        graph = nx.from_numpy_matrix(adjacency)

        labeling = nx.betweenness_centrality(graph, normalized=False)
        labeling = list(labeling.items())
        labeling = sorted(labeling, key=lambda n: n[1], reverse=True)

        return np.array([labels[n[0]] for n in labeling])

    labels = _labels_default(labels, adjacency)
    return tf.py_func(_betweeness_centrality, [adjacency, labels], tf.int32,
                      stateful=False, name='betweenness_centrality')


def _labels_default(labels, adjacency):
    if labels is None:
        return tf.range(0, tf.shape(adjacency)[0], dtype=tf.int32)
    else:
        return labels


labelings = {'identity': identity,
             'betweenness_centrality': betweenness_centrality}

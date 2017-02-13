from six.moves import xrange

import tensorflow as tf
import networkx as nx
import numpy as np
import pynauty as nauty


def scanline(adjacency, labels=None):
    with tf.name_scope('scanline', values=[adjacency, labels]):
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


def canonical(adjacency, labels=None):
    def _canonical(adjacency, labels):
        count = adjacency.shape[0]
        adjacency_dict = {}

        for i in xrange(count):
            adjacency_dict[i] = list(np.nonzero(adjacency[i])[0])

        graph = nauty.Graph(count, adjacency_dict=adjacency_dict)
        labeling = nauty.canonical_labeling(graph)

        labeling = [labels[i] for i in labeling]

        return np.array(labeling, np.int32)

    labels = _labels_default(labels, adjacency)
    return tf.py_func(_canonical, [adjacency, labels], tf.int32,
                      stateful=False, name='canonical')


def _labels_default(labels, adjacency):
    if labels is None:
        return tf.range(0, tf.shape(adjacency)[0], dtype=tf.int32)
    else:
        return labels


labelings = {'scanline': scanline,
             'betweenness_centrality': betweenness_centrality,
             'canonical': canonical}

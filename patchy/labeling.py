import tensorflow as tf
import networkx as nx
import numpy as np


def identity(adjacency, labels=None):
    return _labels_default(labels, adjacency)


def betweenness_centrality(adjacency, labels=None):
    def _betweeness_centrality(adjacency, labels):
        graph = nx.from_numpy_matrix(adjacency)

        
    # result = nx.betweenness_centrality(graph, normalized=False)
    # result = list(result.items())
    # result = sorted(result, key=lambda v: v[1], reverse=True)

    # return [v[0] for v in result]
        # for n in graph:
        #     graph.node[n].update({'label': n})

        labeling = nx.betweenness_centrality(graph, normalized=False)
        labeling = np.array(list(labeling.items()))


        print(np.sort(labeling, axis=0))


        return np.zeros((2, 2), dtype=np.float32)

    labels = _labels_default(labels, adjacency)
    return tf.py_func(
        _betweeness_centrality, [adjacency, labels], tf.float32,
        stateful=False, name='betweenness_centrality')


def weight_to_first_label(adjacency, labels=None):
    pass


def _labels_default(labels, adjacency):
    if labels is None:
        return tf.range(0, tf.shape(adjacency)[0], dtype=tf.int32)
    else:
        return labels

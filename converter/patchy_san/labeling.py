import tensorflow as tf
import networkx as nx
import numpy as np


def betweenness_centrality(adjacent):
    def _betweenness_centrality(adjacent):
        graph = nx.Graph(adjacent)

        labeling = nx.betweenness_centrality(
            graph, normalized=False, weight='weight')
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

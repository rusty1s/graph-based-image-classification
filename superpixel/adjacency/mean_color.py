import tensorflow as tf
import networkx as nx
import numpy as np
from skimage.future import graph


def mean_color_adjacency(image, segmentation):
    def _mean_color_adjacency(image, segmentation):
        rag = graph.rag_mean_color(image, segmentation)
        adjacency = nx.to_numpy_matrix(rag, dtype=np.float32, weight=None)
        mean_colors = nx.to_numpy_matrix(rag, dtype=np.float32)
        return adjacency, mean_colors

    neighbors, mean_color = tf.py_func(
        _mean_color_adjacency, [image, segmentation], [tf.float32, tf.float32],
        stateful=False, name='mean_color_adjacency')

    return tf.cast(neighbors, tf.uint8), mean_color

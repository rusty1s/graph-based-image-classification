import tensorflow as tf
import networkx as nx


def neighborhoods_assembly(sequence, adjacent, size, labeling):
    def _neighborhoods_assembly(sequence, adjacent, size, labeling):
        pass


    # graph = nx.Graph(adjacent)

    def _assemble(node):
        return tf.zeros([size], dtype=tf.int32)

    neighborhoods = tf.map_fn(_assemble, sequence, dtype=tf.int32,
                              name='neighborhood_assembly')

    return neighborhoods

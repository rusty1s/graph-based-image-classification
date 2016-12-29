import tensorflow as tf


def receptive_field(neighborhood, nodes, num_features):
    def _replace(node):
        def _zero():
            return tf.zeros([num_features])

        def _nonzero():
            result = tf.strided_slice(nodes, [node], [node+1], [1])
            return tf.reshape(result, [-1])

        return tf.cond(node > -1, _nonzero, _zero)

    return tf.map_fn(_replace, neighborhood, dtype=tf.float32)


def receptive_fields(neighborhoods, nodes, num_features):
    def _receptive_field(neighborhood):
        return receptive_field(neighborhood, nodes, num_features)

    return tf.map_fn(_receptive_field, neighborhoods, dtype=tf.float32)

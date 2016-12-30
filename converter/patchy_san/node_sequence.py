import tensorflow as tf


def node_sequence(sequence, width, stride):
    # Stride the sequence based on the given stride width.
    sequence = tf.strided_slice(sequence, [0], [width*stride], [stride])

    # Pad right with -1 if we need to.
    padding = tf.ones([width], dtype=tf.int32)
    padding = tf.negative(padding)
    sequence = tf.concat(0, [sequence, padding])

    return tf.strided_slice(sequence, [0], [width], [1])

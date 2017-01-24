import tensorflow as tf


def node_sequence(sequence, width, stride):
    with tf.name_scope('node_sequence', values=[sequence, width, stride]):
        # Stride the sequence based on the given stride width.
        sequence = tf.strided_slice(sequence, [0], [width*stride], [stride])

        # Pad right with -1 if the sequence length is lower than width.
        padding = tf.ones([width - tf.shape(sequence)[0]], dtype=tf.int32)
        padding = tf.negative(padding)
        sequence = tf.concat(0, [sequence, padding])

    return sequence

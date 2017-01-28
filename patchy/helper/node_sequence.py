import tensorflow as tf


def node_sequence(sequence, width, stride):
    """Normalizes a given sequence to have a fixed width by striding over the
    sequence. The returned sequence is padded with -1 if its length is lower
    than the requested width.

    Args:
        sequence: A 1d tensor.
        width: The length of the returned sequence.
        stride: The distance between two selected nodes.

    Returns:
        A 1d tensor.
    """

    with tf.name_scope('node_sequence', values=[sequence, width, stride]):
        # Stride the sequence based on the given stride size.
        sequence = tf.strided_slice(sequence, [0], [width*stride], [stride])

        # Pad right with -1 if the sequence length is lower than width.
        padding = tf.ones([width - tf.shape(sequence)[0]], dtype=tf.int32)
        padding = tf.negative(padding)
        sequence = tf.concat(0, [sequence, padding])

    return sequence

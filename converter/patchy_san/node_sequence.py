def node_sequence(sequence, width, stride):
    # Stride the sequence based on the given stride width.
    size = sequence.get_shape()[0].value
    sequence = tf.strided_slice(sequence, [0], [size], [stride])

    # No more entries than we want.
    sequence = tf.strided_slice(sequence, [0], [width], [1])

    # Pad with zeros if we need to.
    size = sequence.get_shape()[0].value

    if size < width:  # TODO if weg, muss auch ohne gehen
        sequence = tf.add(sequence, tf.ones_like(sequence))
        sequence = tf.pad(sequence, [[0, width-size]])
        sequence = tf.sub(sequence, tf.ones_like(sequence))

    return sequence

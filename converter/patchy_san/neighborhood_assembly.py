def assemble_neighborhoods(sequence, adjacent, size, labeling):
    assemble = assemble_neighborhood(size, adjacent)

    sequence = tf.reshape([-1, 1])
    neighborhoods = tf.map_fn(assemble, t)

    return neighborhoods


def assemble_neighborhood(index, adjacent, size, labeling):
    # this shall return a function assemble(index) with a fixed size
    # this is neighborhood assembly where we already compute a fixed size
    # neighborhood via labeling and distances
    def _assemble(index):
        return index

    return _assemble

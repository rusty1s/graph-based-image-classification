import tensorflow as tf
# returns a 1d tensor with all the infos of the segment
# returns an 2d tensor with the adjacent matrix

# count
# red
# green
# blue
# width
# height
NUM_CHANNELS = 6


def create_graph(image, superpixels):
    one_dim = tf.reshape(superpixels, [-1])
    y, idx, count = tf.unique_with_counts(one_dim)

    # print(y.eval())
    # print(y.get_shape())
    # num_nodes = y.get_shape()[0].value
    # print(num_nodes)
    nodes = tf.zeros([y.get_shape()[0].value, NUM_CHANNELS])
    adjacent = tf.zeros([4, 4])

    return nodes, adjacent

    # Add count to nodes
    # Superpixels is an

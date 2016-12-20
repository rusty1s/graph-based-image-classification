def variable_with_weight_decay(name, shape, stddev, wd):
    pass


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    """

    labels = tf.cast(labels, tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')
